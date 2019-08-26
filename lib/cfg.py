import argparse
import torch.backends.cudnn as cudnn
import numpy as np
period = 30

parser = argparse.ArgumentParser(description='pix2pix-warping-PyTorch-implementation')
parser.add_argument('--continue_train', type=int, default=5, help='the number of starting train')
parser.add_argument('--checkpoint_dir', required=False, default='unet_256_kalman_with_losspixel1', help='unet_affine_temp')
parser.add_argument('--mode', required=True, default='train', help='unet_affine_temp')
parser.add_argument('--num_layer', type=int, default=3, help='number of layers for cascading')
parser.add_argument('--batchSize', type=int, default=16, help='training batch size')
parser.add_argument('--nEpochs', type=int, default=80, help='number of epochs to train for')
parser.add_argument('--input_nc', type=int, default=period + 1, help='input image channels')
parser.add_argument('--output_nc', type=int, default=2, help='output image channels')
parser.add_argument('--ngf', type=int, default=64, help='generator filters in first conv layer')
parser.add_argument('--ndf', type=int, default=32, help='discriminator filters in first conv layer')
parser.add_argument('--lr', type=float, default=0.0001, help='Learning Rate. Default=0.002')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
parser.add_argument('--cuda', default=True, action='store_true', help='use cuda?')
parser.add_argument('--threads', type=int, default=16, help='number of threads for data loader to use')
parser.add_argument('--seed', type=int, default=123, help='random seed to use. Default=123')
parser.add_argument('--lamd', type=int, default=10, help='weight on L1 term in objective')
parser.add_argument('--input_size', type=int, default=256, help='size of input images for networks')
parser.add_argument('--use_gan', type=bool, default=False, help='train with gan or not')
parser.add_argument('--start_gan', type=int, default=40, help='epoch of starting gan')
parser.add_argument('--path_feature', default='../../feature/', help='path of feature for train')
parser.add_argument('--path_affine', default='../../affine640/', help='path of affine for train')
parser.add_argument('--path_image', default='../../image256_rgb/', help='path of image for train')
parser.add_argument('--path_adjacent', default='../../feature_adjacent/', help='path of adjacent homograpy for train')
parser.add_argument('--number_feature', type=int, default=400, help='number of feature points for train')
parser.add_argument('--period_D', type=int, default=3, help='period for discriminator 2*period_D+1')
parser.add_argument('--balance_gd', type=float, default=0.1,help='balance of generator loss and discriminator loss (2 is ok)')
parser.add_argument('--block', type=int, default=16, help='block*block for shapeloss')
parser.add_argument('--shapeloss', type=bool, default=True, help='whether to use shapeloss')
parser.add_argument('--shapeloss_weight', type=float, default=0.001, help='weight of shapeloss')
parser.add_argument('--use_BN', type=bool, default=False, help='whether to use batchnorm')
parser.add_argument('--visdom_port', type=int, default=8009, help='visdom port')
parser.add_argument('--decreaselr', type=int, default=20, help='visdom port')

opt = parser.parse_args()


index_sample = np.append(np.arange(-period // 2, 0, 1), np.arange(0, period // 2 + 1, 1))
index_sample_discriminator=np.append(np.arange(-opt.period_D, 0, 1), np.arange(0, opt.period_D+1, 1))


train_files = [1, 2, 3, 5, 7, 9, 10, 11, 13, 14, 15, 17, 18, 19, 20, 21, 22, 24, 25, 26, 28, 30, 33, 35, 37, 38, 41,
               42, 43, 44, 45, 46, 47, 48, 50, 51, 54, 55, 58, 59, 60, 53]
val_files = [6, 12, 16, 23, 32, 40, 49, 61]
test_files = [4, 8, 34, 39, 52, 27, 29, 57]

cudnn.benchmark = True
