import argparse
import torch.backends.cudnn as cudnn

period = 31

parser = argparse.ArgumentParser(description='pix2pix-warping-PyTorch-implementation')
parser.add_argument('--continue_train', type=int, default=0, help='the number of starting train')
parser.add_argument('--dataset', required=False, default='unet_256_kalman', help='unet_affine_temp')
parser.add_argument('--train', type=bool, default=True, help='unet_affine_temp')
parser.add_argument('--dir_logs', default='logs', help='logs for tensorboard')
parser.add_argument('--num_layer', type=int, default=3, help='number of layers for cascading')
parser.add_argument('--batchSize', type=int, default=6, help='training batch size')
parser.add_argument('--testBatchSize', type=int, default=2, help='testing batch size')
parser.add_argument('--nEpochs', type=int, default=35, help='number of epochs to train for')
parser.add_argument('--input_nc', type=int, default=period + 1, help='input image channels')
parser.add_argument('--output_nc', type=int, default=2, help='output image channels')
parser.add_argument('--ngf', type=int, default=64, help='generator filters in first conv layer')
parser.add_argument('--ndf', type=int, default=32, help='discriminator filters in first conv layer')
parser.add_argument('--lr', type=float, default=0.0002, help='Learning Rate. Default=0.002')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
parser.add_argument('--cuda', default=True, action='store_true', help='use cuda?')
parser.add_argument('--threads', type=int, default=8, help='number of threads for data loader to use')
parser.add_argument('--seed', type=int, default=123, help='random seed to use. Default=123')
parser.add_argument('--lamd', type=int, default=10, help='weight on L1 term in objective')
parser.add_argument('--input_size', type=int, default=256, help='size of input images for networks')
parser.add_argument('--use_gan', type=bool, default=True, help='train with gan or not')
parser.add_argument('--gpu_ids', type=int, default=[0], help='gpu devices')
parser.add_argument('--path_feature', default='feature512/', help='path of feature for train')
parser.add_argument('--path_affine', default='affine640/', help='path of affine for train')
parser.add_argument('--path_image', default='image512_whole/', help='path of image for train')
parser.add_argument('--path_adjacent', default='feature_adjacent256/', help='path of adjacent homograpy for train')
parser.add_argument('--number_feature', type=int, default=400, help='number of feature points for train')
parser.add_argument('--period_D', type=int, default=3, help='period for discriminator 2*period_D+1')
parser.add_argument('--balance_gd', type=float, default=0.1,help='balance of generator loss and discriminator loss (2 is ok)')
parser.add_argument('--affine_weight', type=float, default=0.1, help='weight of affine loss')
parser.add_argument('--start_gan', type=int, default=10, help='epoch of starting gan')
parser.add_argument('--start_loss_affine', type=int, default=0, help='epoch of starting loss_affine')

opt = parser.parse_args()


index_sample = [31,25,19,13,7]#np.append(np.arange(-period // 2, 0, 1), np.arange(0, period // 2 + 1, 1))
index_sample_discriminator=[5,4,3,2,1]#np.append(np.arange(-opt.period_D, 0, 1), np.arange(0, opt.period_D+1, 1))


train_files = [1, 2, 3, 5, 7, 9, 10, 11, 13, 14, 15, 17, 18, 19, 20, 21, 22, 24, 25, 26, 28, 30, 33, 35, 37, 38, 41,
               42, 43, 44, 45, 46, 47, 48, 50, 51, 54, 55, 58, 59, 60, 53]
val_files = [6]#, 12]#, 16, 23, 32, 40, 49, 61]
test_files = [4, 8, 34, 39, 52, 27, 29, 57]

cudnn.benchmark = True