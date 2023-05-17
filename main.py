import time
import os
import json
import random
import torch
import torchvision.transforms.functional as FT
import torch.backends.cudnn as cudnn
from torchvision.utils import make_grid
from torch.utils.tensorboard import SummaryWriter
from models import Generator, Discriminator, TruncatedVGG19
from torch import nn
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from models import SRResNet,Generator
from datasets import SRDataset
from PIL import Image



# Data set parameters
data_folder = './data/' # path to data folder
crop_size = 96 # High resolution image crop size
scaling_factor = 4 # magnification ratio

# Generator model parameters (same as SRResNet)
large_kernel_size_g = 9 # kernel size for the first and last convolutional layers
small_kernel_size_g = 3 # kernel size for the middle layer of convolution
n_channels_g = 64 # number of channels in the middle layer
n_blocks_g = 16 # number of residual modules
srresnet_checkpoint = "./results/checkpoint_srresnet.pth" # Pre-trained SRResNet model to initialize

# Discriminator model parameters
kernel_size_d = 3 # kernel size of all convolution modules
n_channels_d = 64 # number of channels in the first convolutional module, doubling the number of channels in every subsequent module
n_blocks_d = 8 # number of convolution modules
fc_size_d = 1024 # number of connections in the fully connected layer

# learning parameters
batch_size = 128 # batch size
start_epoch = 1 # iteration start position
epochs = 1 # number of iterative rounds
checkpoint = None # SRGAN pre-trained model, if none, fill in None
workers = 16 # number of threads to load data
vgg19_i = 5 # The i-th pooling layer of the VGG19 network
vgg19_j = 4 # jth convolutional layer of the VGG19 network
beta = 1e-3 # discriminant loss multiplier
lr = 1e-4 # learning rate

# device parameters
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ngpu = 1 # number of gpus to run on
cudnn.benchmark = True # accelerate for convolution
writer = SummaryWriter() # Real-time monitoring Use the command tensorboard --logdir runs to view


def training():
    """
    Train
    """
    global checkpoint,start_epoch,writer

    # Model initialization
    generator = Generator(large_kernel_size=large_kernel_size_g,
                              small_kernel_size=small_kernel_size_g,
                              n_channels=n_channels_g,
                              n_blocks=n_blocks_g,
                              scaling_factor=scaling_factor)

    discriminator = Discriminator(kernel_size=kernel_size_d,
                                    n_channels=n_channels_d,
                                    n_blocks=n_blocks_d,
                                    fc_size=fc_size_d)

    # Optimizer initialization
    optimizer_g = torch.optim.Adam(params=filter(lambda p: p.requires_grad,generator.parameters()),lr=lr)
    optimizer_d = torch.optim.Adam(params=filter(lambda p: p.requires_grad,discriminator.parameters()),lr=lr)

    # The truncated VGG19 network is used to calculate the loss function
    truncated_vgg19 = TruncatedVGG19(i=vgg19_i, j=vgg19_j)
    truncated_vgg19.eval()

    # Loss Function
    content_loss_criterion = nn.MSELoss()
    adversarial_loss_criterion = nn.BCEWithLogitsLoss()

    generator = generator.to(device)
    discriminator = discriminator.to(device)
    truncated_vgg19 = truncated_vgg19.to(device)
    content_loss_criterion = content_loss_criterion.to(device)
    adversarial_loss_criterion = adversarial_loss_criterion.to(device)
    
    # Load pre-training Model
    srresnetcheckpoint = torch.load(srresnet_checkpoint)
    generator.net.load_state_dict(srresnetcheckpoint['model'])

    if checkpoint is not None:
        checkpoint = torch.load(checkpoint)
        start_epoch = checkpoint['epoch'] + 1
        generator.load_state_dict(checkpoint['generator'])
        discriminator.load_state_dict(checkpoint['discriminator'])
        optimizer_g.load_state_dict(checkpoint['optimizer_g'])
        optimizer_d.load_state_dict(checkpoint['optimizer_d'])


    # dataloaders
    train_dataset = SRDataset(data_folder,split='train',
                              crop_size=crop_size,
                              scaling_factor=scaling_factor,
                              lr_img_type='imagenet-norm',
                              hr_img_type='imagenet-norm')
    train_loader = torch.utils.data.DataLoader(train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=workers,
        pin_memory=True) 

    for epoch in range(start_epoch, epochs+1):
        
        if epoch == int(epochs / 2):
            adjust_learning_rate(optimizer_g, 0.1)
            adjust_learning_rate(optimizer_d, 0.1)

        generator.train()
        discriminator.train()

        losses_c = AverageMeter()
        losses_a = AverageMeter()
        losses_d = AverageMeter()

        n_iter = len(train_loader)


        for i, (lr_imgs, hr_imgs) in enumerate(train_loader):

            # send data to GPU
            lr_imgs = lr_imgs.to(device)
            hr_imgs = hr_imgs.to(device)


            sr_imgs = generator(lr_imgs)
            sr_imgs = convert_image(
                sr_imgs, source='[-1, 1]',
                target='imagenet-norm')

            sr_imgs_in_vgg_space = truncated_vgg19(sr_imgs)
            hr_imgs_in_vgg_space = truncated_vgg19(hr_imgs).detach()

            content_loss = content_loss_criterion(sr_imgs_in_vgg_space,hr_imgs_in_vgg_space)

            sr_discriminated = discriminator(sr_imgs)
            adversarial_loss = adversarial_loss_criterion(
                sr_discriminated, torch.ones_like(sr_discriminated))

            # Loss
            perceptual_loss = content_loss + beta * adversarial_loss

            # backward.
            optimizer_g.zero_grad()
            perceptual_loss.backward()

            # update generater
            optimizer_g.step()

            #record loss
            losses_c.update(content_loss.item(), lr_imgs.size(0))
            losses_a.update(adversarial_loss.item(), lr_imgs.size(0))

            # discriminater
            hr_discriminated = discriminator(hr_imgs)
            sr_discriminated = discriminator(sr_imgs.detach())

            # loss
            adversarial_loss = adversarial_loss_criterion(sr_discriminated, torch.zeros_like(sr_discriminated)) + \
                            adversarial_loss_criterion(hr_discriminated, torch.ones_like(hr_discriminated))

            # backward
            optimizer_d.zero_grad()
            adversarial_loss.backward()

            # update discriminater
            optimizer_d.step()

            # record loss
            losses_d.update(adversarial_loss.item(), hr_imgs.size(0))

            # monitor grid
            if i==(n_iter-2):
                writer.add_image('SRGAN/epoch_'+str(epoch)+'_1', make_grid(lr_imgs[:4,:3,:,:].cpu(), nrow=4, normalize=True),epoch)
                writer.add_image('SRGAN/epoch_'+str(epoch)+'_2', make_grid(sr_imgs[:4,:3,:,:].cpu(), nrow=4, normalize=True),epoch)
                writer.add_image('SRGAN/epoch_'+str(epoch)+'_3', make_grid(hr_imgs[:4,:3,:,:].cpu(), nrow=4, normalize=True),epoch)
 
        # release memory
        del lr_imgs, hr_imgs, sr_imgs, hr_imgs_in_vgg_space, sr_imgs_in_vgg_space, hr_discriminated, sr_discriminated
        print("第 " + str(epoch) + " 个epoch结束")
        # monitor loss
        writer.add_scalar('SRGAN/Loss_c', losses_c.val, epoch) 
        writer.add_scalar('SRGAN/Loss_a', losses_a.val, epoch)    
        writer.add_scalar('SRGAN/Loss_d', losses_d.val, epoch)    

        # save model
        torch.save({
            'epoch': epoch,
            'generator': generator.state_dict(),
            'discriminator': discriminator.state_dict(),
            'optimizer_g': optimizer_g.state_dict(),
            'optimizer_g': optimizer_g.state_dict(),
        }, 'results/checkpoint_srgan.pth')

    writer.close()


if __name__ == '__main__':
    def create_data_lists(train_folders, test_folders, min_size, output_folder):
        """
        Creates a list of training and test set files.
        Parameters train_folders: collection of training folders; the images in each folder will be merged into a single image list file
        Parameters test_folders: collection of test folders; each folder will form a single image list file
        parameter min_size: minimum tolerance value for image width and height
        parameter output_folder: the final list of files to be generated, in json format
        """
        train_images = list()
        for d in train_folders:
            for i in os.listdir(d):
                img_path = os.path.join(d, i)
                img = Image.open(img_path, mode='r')
                if img.width >= min_size and img.height >= min_size:
                    train_images.append(img_path)

        with open(os.path.join(output_folder, 'train_images.json'), 'w') as j:
            json.dump(train_images, j)

        for d in test_folders:
            test_images = list()
            test_name = d.split("/")[-1]
            for i in os.listdir(d):
                img_path = os.path.join(d, i)
                img = Image.open(img_path, mode='r')
                if img.width >= min_size and img.height >= min_size:
                    test_images.append(img_path)
            with open(os.path.join(output_folder, test_name + '_test_images.json'), 'w') as j:
                json.dump(test_images, j)


    create_data_lists(train_folders=['./data/DIV2K/DIV2K_train_HR',
                                     './data/DIV2K/DIV2K_valid_HR'],
                      test_folders=['./data/Set5',
                                    './data/Set14'],
                      min_size=100,
                      output_folder='./data/')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # constent
    rgb_weights = torch.FloatTensor([65.481, 128.553, 24.966]).to(device)
    imagenet_mean = torch.FloatTensor([0.485, 0.456,
                                       0.406]).unsqueeze(1).unsqueeze(2)
    imagenet_std = torch.FloatTensor([0.229, 0.224,
                                      0.225]).unsqueeze(1).unsqueeze(2)
    imagenet_mean_cuda = torch.FloatTensor(
        [0.485, 0.456, 0.406]).to(device).unsqueeze(0).unsqueeze(2).unsqueeze(3)
    imagenet_std_cuda = torch.FloatTensor(
        [0.229, 0.224, 0.225]).to(device).unsqueeze(0).unsqueeze(2).unsqueeze(3)


    def convert_image(img, source, target):
        """
        Convert the image format.

        :parameter img: input image
        :parameter source: the data source format, there are 3 types
                       (1) 'pil' (PIL image)
                       (2) '[0, 1]'
                       (3) '[-1, 1]'
        :parameter target: data target format, 5 types
                       (1) 'pil' (PIL image)
                       (2) '[0, 1]'
                       (3) '[-1, 1]'
                       (4) 'imagenet-norm' (normalised by the mean and variance of the imagenet dataset)
                       (5) 'y-channel' (luminance channel Y, using YCbCr colour space, used to calculate PSNR and SSIM)
        : return: converted image
        """

        # convert image data to [0, 1]
        if source == 'pil':
            img = FT.to_tensor(img)

        elif source == '[0, 1]':
            pass

        elif source == '[-1, 1]':
            img = (img + 1.) / 2.


        if target == 'pil':
            img = FT.to_pil_image(img)

        elif target == '[0, 255]':
            img = 255. * img

        elif target == '[0, 1]':
            pass

        elif target == '[-1, 1]':
            img = 2. * img - 1.

        elif target == 'imagenet-norm':
            if img.ndimension() == 3:
                img = (img - imagenet_mean) / imagenet_std
            elif img.ndimension() == 4:
                img = (img - imagenet_mean_cuda) / imagenet_std_cuda

        elif target == 'y-channel':
            img = torch.matmul(255. * img.permute(0, 2, 3, 1)[:, 4:-4, 4:-4, :],
                               rgb_weights) / 255. + 16.

        return img


    class ImageTransforms(object):

        def __init__(self, split, crop_size, scaling_factor, lr_img_type,
                     hr_img_type):
            """
            :parameter split: 'train' or 'test'
            :parameter crop_size: High resolution image crop size
            :parameter scaling_factor: magnification ratio
            :parameter lr_img_type: low-resolution image pre-processing method
            :parameter hr_img_type: high-resolution image preprocessing method
            """
            self.split = split.lower()
            self.crop_size = crop_size
            self.scaling_factor = scaling_factor
            self.lr_img_type = lr_img_type
            self.hr_img_type = hr_img_type

            assert self.split in {'train', 'test'}

        def __call__(self, img):
            """
            Cropping and downsampling the image to form a low resolution image
            :parameter img: image read by the PIL library
            :return: low and high resolution images of a particular form
            """

            if self.split == 'train':
                # random cut
                left = random.randint(1, img.width - self.crop_size)
                top = random.randint(1, img.height - self.crop_size)
                right = left + self.crop_size
                bottom = top + self.crop_size
                hr_img = img.crop((left, top, right, bottom))
            else:
                x_remainder = img.width % self.scaling_factor
                y_remainder = img.height % self.scaling_factor
                left = x_remainder // 2
                top = y_remainder // 2
                right = left + (img.width - x_remainder)
                bottom = top + (img.height - y_remainder)
                hr_img = img.crop((left, top, right, bottom))

            lr_img = hr_img.resize((int(hr_img.width / self.scaling_factor),
                                    int(hr_img.height / self.scaling_factor)),
                                   Image.BICUBIC)


            lr_img = convert_image(lr_img, source='pil', target=self.lr_img_type)
            hr_img = convert_image(hr_img, source='pil', target=self.hr_img_type)

            return lr_img, hr_img


    class AverageMeter(object):

        def __init__(self):
            self.reset()

        def reset(self):
            self.val = 0
            self.avg = 0
            self.sum = 0
            self.count = 0

        def update(self, val, n=1):
            self.val = val
            self.sum += val * n
            self.count += n
            self.avg = self.sum / self.count


    def clip_gradient(optimizer, grad_clip):
        """
        Discard gradients to prevent them from exploding during computation.

        :parameter optimizer: optimizer whose gradient will be truncated
        :parameter grad_clip: truncated value
        """
        for group in optimizer.param_groups:
            for param in group['params']:
                if param.grad is not None:
                    param.grad.data.clamp_(-grad_clip, grad_clip)


    def save_checkpoint(state, filename):
        """
        save result
        """

        torch.save(state, filename)


    def adjust_learning_rate(optimizer, shrink_factor):

        for param_group in optimizer.param_groups:
            param_group['lr'] = param_group['lr'] * shrink_factor


    training()

    large_kernel_size = 9  # kernel size for the first and last convolutional layers
    small_kernel_size = 3  # kernel size for the middle convolution layer
    n_channels = 64  # number of channels in the middle layer
    n_blocks = 16  # number of residual modules
    scaling_factor = 4  # magnification ratio
    ngpu = 1  # number of GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Test set directory
    data_folder = "./data/"
    test_data_names = ["Set5", "Set14"]

    # Pre-trained models
    srgan_checkpoint = "./results/checkpoint_srgan.pth"

    # Load model SRResNet or SRGAN
    checkpoint = torch.load(srgan_checkpoint)
    generator = Generator(large_kernel_size=large_kernel_size,
                          small_kernel_size=small_kernel_size,
                          n_channels=n_channels,
                          n_blocks=n_blocks,
                          scaling_factor=scaling_factor)
    generator = generator.to(device)
    generator.load_state_dict(checkpoint['generator'])

    generator.eval()
    model = generator

    for test_data_name in test_data_names:

        # DataLoader
        test_dataset = SRDataset(data_folder,
                                 split='test',
                                 crop_size=0,
                                 scaling_factor=4,
                                 lr_img_type='imagenet-norm',
                                 hr_img_type='[-1, 1]',
                                 test_data_name=test_data_name)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=1,
                                                  pin_memory=True)

        # record PSNR SSIM
        PSNRs = AverageMeter()
        SSIMs = AverageMeter()

        # record time
        start = time.time()

        with torch.no_grad():
            for i, (lr_imgs, hr_imgs) in enumerate(test_loader):
                lr_imgs = lr_imgs.to(device)
                hr_imgs = hr_imgs.to(device)

                # forward
                sr_imgs = model(lr_imgs)

                # calculate PSNR  SSIM
                sr_imgs_y = convert_image(sr_imgs, source='[-1, 1]', target='y-channel').squeeze(
                    0)
                hr_imgs_y = convert_image(hr_imgs, source='[-1, 1]', target='y-channel').squeeze(
                    0)
                psnr = peak_signal_noise_ratio(hr_imgs_y.cpu().numpy(), sr_imgs_y.cpu().numpy(),
                                               data_range=255.)
                ssim = structural_similarity(hr_imgs_y.cpu().numpy(), sr_imgs_y.cpu().numpy(),
                                             data_range=255.)
                PSNRs.update(psnr, lr_imgs.size(0))
                SSIMs.update(ssim, lr_imgs.size(0))

        # output PSNR and SSIM
        print('PSNR  {psnrs.avg:.3f}'.format(psnrs=PSNRs))
        print('SSIM  {ssims.avg:.3f}'.format(ssims=SSIMs))

    print("\n")
