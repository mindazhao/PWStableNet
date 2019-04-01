from __future__ import print_function
import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
# from torch.utils.data import DataLoader
from torch.utils.data import Dataset, DataLoader
from networks_cascading import define_G, define_D, GANLoss
import torch.backends.cudnn as cudnn
import random
import cv2
import numpy as np
from tensorboardX import SummaryWriter
import torchvision.utils as vutils
from vgg import GeneratorLoss
import utils
import itertools
import torch.nn.functional as functional
import time
#os.environ["CUDA_VISIBLE_DEVICES"] = "0"

period = 30

parser = argparse.ArgumentParser(description='pix2pix-warping-PyTorch-implementation')
parser.add_argument('--continue_train',type=int,default=0,help='the number of starting train')
parser.add_argument('--dataset', required=False, default='unet_256_kalman', help='unet_affine_temp')
parser.add_argument('--train', type=bool, default=True, help='unet_affine_temp')
parser.add_argument('--dir_logs', default='logs', help='logs for tensorboard')
parser.add_argument('--num_layer', type=int, default=3, help='number of layers for cascading')
parser.add_argument('--batchSize', type=int, default=8, help='training batch size')
parser.add_argument('--testBatchSize', type=int, default=2, help='testing batch size')
parser.add_argument('--nEpochs', type=int, default=60, help='number of epochs to train for')
parser.add_argument('--input_nc', type=int, default=period + 1, help='input image channels')
parser.add_argument('--output_nc', type=int, default=2, help='output image channels')
parser.add_argument('--ngf', type=int, default=32, help='generator filters in first conv layer')
parser.add_argument('--ndf', type=int, default=64, help='discriminator filters in first conv layer')
parser.add_argument('--lr', type=float, default=0.0002, help='Learning Rate. Default=0.002')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
parser.add_argument('--cuda', default=True, action='store_true', help='use cuda?')
parser.add_argument('--threads', type=int, default=16, help='number of threads for data loader to use')
parser.add_argument('--seed', type=int, default=123, help='random seed to use. Default=123')
parser.add_argument('--lamd', type=int, default=10, help='weight on L1 term in objective')
parser.add_argument('--input_size',type=int,default=256,help='size of input images for networks')
parser.add_argument('--use_gan',type=bool,default=True,help='train with gan or not')
parser.add_argument('--gpu_ids',type=int,default=[],help='gpu devices')
parser.add_argument('--path_feature',default='feature512/',help='path of feature for train')
parser.add_argument('--path_image',default='image512_whole/',help='path of image for train')
parser.add_argument('--path_adjacent',default='feature_adjacent256/',help='path of adjacent homograpy for train')
parser.add_argument('--number_feature',type=int,default=400,help='number of feature points for train')
parser.add_argument('--period_D',type=int,default=3,help='period for discriminator 2*period_D+1')
parser.add_argument('--balance_gd', type=float, default=1, help='balance of generator loss and discriminator loss')
opt = parser.parse_args()
print(opt)

if opt.use_gan:
    criterionGAN = GANLoss()


index_sample = np.append(np.arange(-period // 2, 0, 1), np.arange(0, period // 2 + 1, 1))
index_sample_discriminator=np.append(np.arange(-opt.period_D, 0, 1), np.arange(0, opt.period_D+1, 1))

feature_all_all = []
train_files = [1]#, 2, 3, 5, 7, 9, 10, 11, 13, 14, 15, 17, 18, 19, 20, 21, 22, 24, 25, 26, 28, 30, 33, 35, 37, 38, 41,
               #42, 43, 44, 45, 46, 47, 48, 50, 51, 54, 55, 58, 59, 60, 53]
val_files = [6]#, 12]#, 16, 23, 32, 40, 49, 61]
test_files = [4, 8, 34, 39, 52, 27, 29, 57]


if opt.cuda and not torch.cuda.is_available():
    raise Exception("No GPU found, please run without --cuda")

cudnn.benchmark = True


print('===> Building model')
# netG = Net()
netG = define_G(opt.input_nc, opt.output_nc, opt.ngf, 'normal', 0.02, opt.gpu_ids)
netD = define_D(2*opt.period_D+1, opt.ndf, 'n_layers', n_layers_D=5, norm='batch', use_sigmoid=False)
#netD2 = define_D(3 + 3, opt.ndf, 'n_layers', n_layers_D=5, norm='batch', use_sigmoid=False, gpu_ids=[0])

# setup optimizer
generator_criterion=GeneratorLoss()
torch.manual_seed(opt.seed)
if opt.cuda:
    netG = netG.cuda()
    netD = netD.cuda()
    #netD2 = netD2.cuda()
    generator_criterion = generator_criterion.cuda()
    torch.cuda.manual_seed_all(opt.seed)
    if opt.use_gan:
        criterionGAN.cuda()


def list_random(files):
    list_random = []

    for video_id in range(0, len(files)):
        feature_path = opt.path_feature + str(files[video_id]) + '.avi.txt'
        f = open(feature_path, "r")

        numframe = list(map(int, f.readline().split()))[0]
        for i in range(0 + period // 2, numframe - period // 2 - 1, 1):
            video_frame = [video_id, i]
            list_random.append(video_frame)
        f.close()
    return list_random


def image_store(files):#store all images and features in memory for accelarating loading speed.
    list_unstable = []
    list_stable = []
    list_feature=[]
    list_adjacent=[]

    for video_id in range(0, len(files)):
        print('image_store---' + '[' +str(video_id)+ '/'+ str(len(files))+' ]')
        feature_path = opt.path_feature + str(files[video_id]) + '.avi.txt'
        adjacent_path = opt.path_adjacent + str(files[video_id]) + '.avi.txt'
        f = open(feature_path, "r")
        f_adjacent = open(adjacent_path, "r")
        list_unstable_clip = []
        list_stable_clip = []
        feature_clip = []
        adjacent_clip = []
        numframe = list(map(int, f.readline().split()))[0]
        f_adjacent.readline()
        for i in range(0, numframe):

            image_path_unstable = opt.path_image+'unstable/' + str(files[video_id]) + '.avi/'
            image_path_stable = opt.path_image+'stable/' + str(files[video_id]) + '.avi/'
            image_path_unstable_one = image_path_unstable + str(i) + '.png'
            image_path_stable_one = image_path_stable + str(i) + '.png'

            image_unstable = cv2.imread(image_path_unstable_one)
            size = (opt.input_size, opt.input_size)
            shrink = cv2.resize(image_unstable, size, interpolation=cv2.INTER_AREA)
            shrink = cv2.cvtColor(shrink, cv2.COLOR_BGR2RGB)
            list_unstable_clip.append(shrink)

            image_stable = cv2.imread(image_path_stable_one)
            size = (opt.input_size, opt.input_size)
            shrink = cv2.resize(image_stable, size, interpolation=cv2.INTER_AREA)
            shrink = cv2.cvtColor(shrink, cv2.COLOR_BGR2RGB)
            list_stable_clip.append(shrink)

            feature = []
            f.readline()#400
            for j in range(0, opt.number_feature):
                line = f.readline()
                line_re = line.split()
                feature.append(list(map(float, line_re)))

            feature_clip.append(feature)

            line = f_adjacent.readline()
            line_re = line.split()
            adjacent_clip.append(list(map(float, line_re)))
        f.close()
        f_adjacent.close()

        list_unstable.append(list_unstable_clip)
        list_stable.append(list_stable_clip)
        list_feature.append(feature_clip)
        list_adjacent.append(adjacent_clip)

    return list_stable, list_unstable,list_feature,list_adjacent


class customData(Dataset):
    def __init__(self, files, list_random, list_stable, list_unstable,list_feature,list_adjacent,with_gan):
        self.feature_all_all = list_feature
        self.feature_all_adjacent =list_adjacent
        self.files = files
        self.list_random = list_random
        self.list_stable = list_stable
        self.list_unstable = list_unstable
        self.with_gan=with_gan

    def __len__(self):
        return len(self.list_random)

    def __getitem__(self, index):
        item = self.list_random[index]
        video_id = item[0]
        numframe_id = item[1]
        ##First part

        list_unstable = []
        list_stable = []
        list_stable_unstable = []
        list_stable_D=[]
        for j in range(0, len(index_sample)):
            image_stable = self.list_unstable[video_id][numframe_id - index_sample[len(index_sample) - 1 - j]]
            image_stable = cv2.cvtColor(image_stable, cv2.COLOR_RGB2GRAY)
            list_stable.append(image_stable)

        image_unstable = self.list_unstable[video_id][numframe_id]  # cv2.imread(image_path_unstable + str(int(numframe_id)) + '.png')
        list_unstable.append(image_unstable)

        image_stable = self.list_stable[video_id][numframe_id]  # cv2.imread(image_path_stable + str(int(numframe_id)) + '.png')
        list_stable_unstable.append(image_stable)



        if opt.use_gan and self.with_gan:
            for j in range(0, len(index_sample_discriminator)):
                image_stable = self.list_unstable[video_id][numframe_id - index_sample_discriminator[len(index_sample_discriminator) - 1 - j]]
                image_stable = cv2.cvtColor(image_stable, cv2.COLOR_RGB2GRAY)
                list_stable_D.append(image_stable)
                np_D = np.array(list_stable_D)

        np_A = np.array(list_stable)
        np_B = np.array(list_unstable).transpose(0, 3, 1, 2).squeeze()
        np_C = np.array(list_stable_unstable).transpose(0, 3, 1, 2).squeeze()
        if opt.use_gan and self.with_gan:
            img_data1 = np.concatenate((np_A, np_B, np_C,np_D), axis=0)
        else:
            img_data1 = np.concatenate((np_A, np_B, np_C), axis=0)

        feature_one = np.array(self.feature_all_all[video_id][int(numframe_id)])
        arr_add = np.ones([feature_one.shape[0], 1])
        feature_data1 = np.concatenate((feature_one[:, 2:4], arr_add, feature_one[:, 0:2], arr_add),axis=1)  # stable unstable



        ## second_part

        numframe_id = item[1] + 1
        list_unstable = []
        list_stable = []
        list_stable_unstable = []
        list_stable_D = []
        for j in range(0, len(index_sample)):
            image_stable = self.list_unstable[video_id][numframe_id - index_sample[len(index_sample) - 1 - j]]
            image_stable = cv2.cvtColor(image_stable, cv2.COLOR_RGB2GRAY)

            list_stable.append(image_stable)

        image_unstable = self.list_unstable[video_id][numframe_id]  # cv2.imread(image_path_unstable + str(int(numframe_id)) + '.png')
        list_unstable.append(image_unstable)

        image_stable = self.list_stable[video_id][numframe_id]  # cv2.imread(image_path_stable + str(int(numframe_id)) + '.png')

        list_stable_unstable.append(image_stable)
        if opt.use_gan and self.with_gan:
            for j in range(0, len(index_sample_discriminator)):
                image_stable = self.list_unstable[video_id][numframe_id - index_sample_discriminator[len(index_sample_discriminator) - 1 - j]]
                image_stable = cv2.cvtColor(image_stable, cv2.COLOR_RGB2GRAY)
                list_stable_D.append(image_stable)
                np_D = np.array(list_stable_D)


        np_A = np.array(list_stable)
        np_B = np.array(list_unstable).transpose(0, 3, 1, 2).squeeze()
        np_C = np.array(list_stable_unstable).transpose(0, 3, 1, 2).squeeze()
        if opt.use_gan and self.with_gan:
            img_data2 = np.concatenate((np_A, np_B, np_C,np_D), axis=0)
        else:
            img_data2 = np.concatenate((np_A, np_B, np_C), axis=0)


        feature_one = np.array(self.feature_all_all[video_id][int(numframe_id)])
        arr_add = np.ones([feature_one.shape[0], 1])
        feature_data2 = np.concatenate((feature_one[:, 2:4], arr_add, feature_one[:, 0:2], arr_add),axis=1)  # stable unstable


        feature_one_adjacent = np.array(self.feature_all_adjacent[video_id][int(numframe_id) - 1])

        return img_data1, feature_data1, img_data2, feature_data2, feature_one_adjacent  # stable->unstable


def pre_propossing(images, features):
    images = images.float() * (1. / 255) * 2 - 1
    images_unstable = images[:, 0:period + 1 + 3, :, :]
    images_stable = images[:, period + 1 + 3:, :, :]
    feature_stable = features[:, :, 0:3]
    feature_unstable = features[:, :, 3:6]
    feature_stable = feature_stable.permute(0, 2, 1)
    feature_unstable = feature_unstable.permute(0, 2, 1)
    return images_stable, images_unstable, feature_stable, feature_unstable


def loss_pixel(grid):
    target_height = opt.input_size
    target_width = opt.input_size
    HW = target_height * target_width
    target_coordinate = list(itertools.product(range(target_height), range(target_width)))
    target_coordinate = torch.Tensor(target_coordinate)  # HW x 2
    Y, X = target_coordinate.split(1, dim=1)
    Y = Y * 2 / (target_height - 1) - 1
    X = X * 2 / (target_width - 1) - 1
    target_coordinate = torch.cat([X, Y], dim=1)  # convert from (y, x) to (x, y)
    #One = torch.ones(X.size())
    stable = target_coordinate#torch.cat([target_coordinate, One], dim=1)
    stable = stable.expand(opt.batchSize, target_height * target_width, 2).permute(0, 2, 1)
    stable=stable.reshape([opt.batchSize,2, opt.input_size,opt.input_size])
    stable = stable.cuda()
    #grid_reshape = grid.view(-1, target_height * target_width, 2).permute(0, 2, 1)
    grid=grid.permute(0,3,1,2)
    # affine_loss = torch.mean(torch.abs(torch.matmul(affine, stable.float()) - grid_reshape.float()))
    variation=grid-stable

    delta_x = torch.abs(variation[:, :, 0: -1, :] - variation[:, :, 1:,:])
    delta_y = torch.abs(variation[:, :, :, 0: -1] - variation[:, :,:, 1:])

    delta = (torch.mean(delta_x) + torch.mean(delta_y)) / 2
    '''
    loss = 0
    for i in range(opt.batchSize):
        pts1 = []  # XX
        pts2 = []  # YY

        for sample_height in range(200):
            x = random.randint(0, 255)
            y = random.randint(0, 255)
            X = grid[i, x, y, 0].cpu().detach().numpy()
            Y = grid[i, x, y, 1].cpu().detach().numpy()
            Xp = float(x) / 128 - 1
            Yp = float(y) / 128 - 1

            pts1.append([X, Y, 1, 0, 0, 0, -Xp * X, -Xp * Y])
            pts1.append([0, 0, 0, X, Y, 1, -Yp * X, -Yp * Y])

            pts2.append(Xp)
            pts2.append(Yp)

        pts1 = np.array(pts1, dtype=np.float32)
        pts2 = np.array(pts2, dtype=np.float32)

        loss += np.linalg.norm(np.dot(np.dot(pts1, np.linalg.inv(np.dot(pts1.T, pts1))), np.dot(pts1.T, pts2)) - pts2)

    loss = loss / opt.batchSize / opt.number_feature
    '''
    return delta


def loss_calulate(grid, feature_stable, feature_unstable, fake, real,batchSize):
    feature_loss = 0
    for i in range(batchSize):
        grid_pos = grid[i, ((feature_stable[i, 1, :] + 1) * opt.input_size / 2).int().cpu().numpy(),
                   ((feature_stable[i, 0, :] + 1) * opt.input_size / 2).int().cpu().numpy(), :]
        feature_loss = feature_loss + torch.mean(torch.abs(feature_unstable[i, 0:2, :] - torch.t(grid_pos)))

    feature_loss = feature_loss / opt.batchSize
    # loss = mse_loss + args.balance_para * feature_loss
    # loss_f = torch.nn.MSELoss()
    mse_loss = torch.mean(torch.abs(real[:,0:3,:,:] - fake))

    delta_x = torch.abs(grid[:, :, 0:opt.input_size - 1, :] - grid[:, :, 1:opt.input_size, :])
    delta_y = torch.abs(grid[:, 0:opt.input_size - 1, :, :] - grid[:, 1:opt.input_size, :, :])

    delta_xx = torch.abs(delta_x[:, :, 0:-1, :] - delta_x[:, :, 1:, :])
    delta_yy = torch.abs(delta_y[:, 0:-1, :, :] - delta_y[:, 1:, :, :])

    delta = (torch.mean(delta_xx) + torch.mean(delta_yy)) / 2

    # delta_x=torch.abs(grid[])

    return mse_loss, delta, feature_loss  # ,loss_sim


def load_model(model):
    # pretrained_dict = torchvision.models.resnet50(pretrained=False).state_dict()
    pretrained_dict = torch.load('./checkpoint/day2night.pth')
    model_dict = model.state_dict()

    unnecessary_list = ['model.model.0.weight', 'model.model.3.weight', 'model.model.3.bias']
    # 1. filter out unnecessary keys
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k not in unnecessary_list}

    pretrained_dict = {'module.' + k: v for k, v in pretrained_dict.items()}

    pretrained_dict_new = {k: v for k, v in pretrained_dict.items() if k in model_dict}

    # 2. overwrite entries in the existing state dict
    model_dict.update(pretrained_dict_new)
    # 3. load the new state dict
    model.load_state_dict(model_dict)
    start_epoch = 0

    return model


def train(epoch, lr, list_stable, list_unstable, list_feature, list_adjacent, list_epoch):
    if epoch>20:
        with_gan=True
    else:
        with_gan=False
    netG.train()
    if opt.continue_train and epoch==opt.continue_train:
         net_g_model_out_path = "checkpoint/{}/netG_model_epoch_{}.pth".format(opt.dataset, opt.continue_train)
         checkpoint=torch.load(net_g_model_out_path)
         netG.load_state_dict(checkpoint['net'])

    # netG=load_model(netG)
    optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(opt.beta1, 0.999))
    #optimizerD = optim.Adam([{'params': netD1.parameters()}, {'params': netD2.parameters()}], lr=0.0002,betas=(opt.beta1, 0.999))
    optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(opt.beta1, 0.999))
    writer = SummaryWriter(opt.dir_logs)

    random.shuffle(list_epoch)
    dataset = customData(train_files, list_epoch, list_stable, list_unstable,list_feature,list_adjacent,with_gan)
    sample = torch.utils.data.DataLoader(dataset, batch_size=opt.batchSize, shuffle=True,num_workers=opt.threads,pin_memory=True)
    iter_sample = iter(sample)

    batch_idxs = (int(dataset.__len__())) // opt.batchSize

    for i in range(batch_idxs):
        time_begin = time.time()
        images1, features1, images2, features2, feature_adjacent = next(iter_sample)

        images1 = images1.cuda()
        images2 = images2.cuda()
        features1 = features1.cuda().float()
        features2 = features2.cuda().float()

        feature_adjacent = feature_adjacent.cuda().float()

        image_stable1, image_unstable1, feature_stable1, feature_unstable1 = pre_propossing(images1, features1,)

        image_stable2, image_unstable2, feature_stable2, feature_unstable2 = pre_propossing(images2, features2,)

        # real_a.data.resize_(image_unstable1.size()).copy_(image_unstable1)
        # real_b.data.resize_(image_stable1.size()).copy_(image_stable1)

        # fake_b, grid = netG(real_a)


        fake1=[]
        fake2=[]

        grid1= netG(image_unstable1[:, 0:period + 1, :, :])

        for nl in range(opt.num_layer):
            grid1[nl]=grid1[nl].permute(0,2,3,1)
            fake1.append(functional.grid_sample(image_unstable1[:, period + 1:period + 1 + 3, :, :], grid1[nl]))
        fake1_gray = functional.grid_sample(image_unstable1[:, period//2 : period//2 + 1 :, :], grid1[2])

        grid2 = netG(image_unstable2[:, 0:period + 1, :, :])

        for nl in range(opt.num_layer):
            grid2[nl] = grid2[nl].permute(0, 2, 3, 1)
            fake2.append(functional.grid_sample(image_unstable2[:, period + 1:period + 1 + 3, :, :], grid2[nl]))
        fake2_gray = functional.grid_sample(image_unstable2[:, period // 2: period // 2 + 1:, :], grid2[2])


        ############################
        # (1) Update D network: maximize log(D(x,y)) + log(1 - D(x,G(x)))
        ###########################
        if opt.use_gan and with_gan:

            optimizerD.zero_grad()

            # train with fake
            fake_ab1 = torch.cat((image_stable1[:,3:3+opt.period_D,:,:], fake1_gray, image_stable1[:,3+opt.period_D+1:3+opt.period_D+1+opt.period_D,:,:]), 1)
            pred_fake1 = netD.forward(fake_ab1.detach())
            loss_d_fake1 = criterionGAN(pred_fake1, False)

            fake_ab2 = torch.cat((image_stable2[:,3:3+opt.period_D,:,:], fake2_gray, image_stable2[:,3+opt.period_D+1:3+opt.period_D+1+opt.period_D,:,:]), 1)
            pred_fake2 = netD.forward(fake_ab2.detach())
            loss_d_fake2 = criterionGAN(pred_fake2, False)


            # train with real
            real_ab1 = image_stable1[:,3:3+opt.period_D*2+1,:,:]
            pred_real1 = netD.forward(real_ab1)
            loss_d_real1 = criterionGAN(pred_real1, True)

            real_ab2 = image_stable2[:,3:3+opt.period_D*2+1,:,:]
            pred_real2 = netD.forward(real_ab2)
            loss_d_real2 = criterionGAN(pred_real2, True)

            # Combined loss
            loss_d = (loss_d_fake1 + loss_d_fake2 + loss_d_real1 + loss_d_real2) * 0.5

            loss_d.backward()

            optimizerD.step()


        ############################
        # (2) Update G network: maximize log(D(x,G(x))) + L1(y,G(x))
        ##########################
        optimizerG.zero_grad()
        if opt.use_gan and with_gan:
            fake_ab1 = torch.cat((image_stable1[:,3:3+opt.period_D,:,:], fake1_gray, image_stable1[:,3+opt.period_D+1:3+opt.period_D+1+opt.period_D,:,:]), 1)
            pred_fake1 = netD.forward(fake_ab1)
            loss_d_fake1_g = criterionGAN(pred_fake1, True)*opt.balance_gd

            fake_ab2 = torch.cat((image_stable2[:,3:3+opt.period_D,:,:], fake2_gray, image_stable2[:,3+opt.period_D+1:3+opt.period_D+1+opt.period_D,:,:]), 1)
            pred_fake2 = netD.forward(fake_ab2)
            loss_d_fake2_g = criterionGAN(pred_fake2, True)*opt.balance_gd



        loss_mse=0
        loss_feature=0
        loss_delta=0
        loss_vgg=0
        loss_g2=0
        loss_affine=0
        for nl in range(opt.num_layer):
            loss_mse1, loss_delta1, loss_feature1 = loss_calulate(grid1[nl], feature_stable1, feature_unstable1, fake1[nl], image_stable1,opt.batchSize)

            loss_mse2, loss_delta2, loss_feature2 = loss_calulate(grid2[nl], feature_stable2, feature_unstable2, fake2[nl], image_stable2,opt.batchSize)
            loss_mse += loss_mse1 + loss_mse2
            loss_feature += loss_feature1 + loss_feature2
            loss_delta += loss_delta1 + loss_delta2
            loss_vgg += generator_criterion(fake1[nl], image_stable1[:,0:3,:,:])
            loss_vgg += generator_criterion(fake2[nl], image_stable2[:,0:3,:,:])
            loss_g2+=torch.mean(torch.abs(fake2[nl] - fake1[nl]))

            loss_affine+=loss_pixel(grid1[nl])+loss_pixel(grid2[nl])*100


        # loss_g1 = loss_feature1 + loss_vgg1 + loss_feature2 + loss_vgg2  # +delta# loss_feature#+loss_mse#+10*loss_g_g#loss_g_g  # + loss_g_l1

        loss_g1 = loss_feature + loss_vgg + loss_mse#+ loss_affine * 20  # (delta1+delta2)*10+loss_affine*1000
        if opt.use_gan and with_gan:

            loss_g = loss_g1 + loss_g2 * opt.lamd +(loss_d_fake1_g+loss_d_fake2_g)/2
        else:
            loss_g = loss_g1 + loss_g2 * opt.lamd
        loss_g.backward()

        optimizerG.step()


        if i % 10 == 0:

            writer.add_scalar('scalar/loss_g_adjacent', loss_g2, i + epoch * batch_idxs)
            writer.add_scalar('scalar/loss_g_feature', loss_feature, i + epoch * batch_idxs)
            writer.add_scalar('scalar/loss_vgg', loss_vgg, i + epoch * batch_idxs)
            writer.add_scalar('scalar/delta', loss_delta, i + epoch * batch_idxs)
            writer.add_scalar('scalar/loss_affine', loss_affine, i + epoch * batch_idxs)
            writer.add_scalar('scalar/mse', loss_mse, i + epoch * batch_idxs)
            if opt.use_gan and with_gan:
                writer.add_scalar('scalar/loss_d_fake1_g', loss_d_fake1_g, i + epoch * batch_idxs)
        if i%100==0:
            writer.add_image('image/unstable',
                             vutils.make_grid(image_unstable1[0:3, period + 1:period + 1 + 3, :, :], normalize=True,
                                              scale_each=True),
                             i + epoch * batch_idxs)
            for nl in range(opt.num_layer):

                writer.add_image('image/stable_fake'+str(nl+1),
                                 vutils.make_grid(fake1[nl][0:3, 0:3, :, :], normalize=True, scale_each=True),
                                 i + epoch * batch_idxs)

            writer.add_image('image/stable_real',
                             vutils.make_grid(image_stable1[0:3, 0:3, :, :], normalize=True, scale_each=True),
                             i + epoch * batch_idxs)
        time_end = time.time()
        time_each=time_end-time_begin
        time_left=((opt.nEpochs-epoch)*batch_idxs+batch_idxs-i)*time_each/3600
        print("=>train--Epoch[{}]({}/{}): time(s): {:.4f} Time_left(h): {:.4f}  Learning rate: {:.6f}".format(
            epoch, i, batch_idxs, time_each, time_left, lr))

def test(epoch,list_stable, list_unstable, list_feature, list_adjacent, list_epoch):
    netG.eval()

    writer = SummaryWriter(opt.dir_logs)

    random.shuffle(list_epoch)
    dataset = customData(val_files, list_epoch, list_stable, list_unstable, list_feature, list_adjacent)
    sample = torch.utils.data.DataLoader(dataset, batch_size=opt.testBatchSize, shuffle=False, num_workers=opt.threads,
                                         pin_memory=True)
    iter_sample = iter(sample)

    batch_idxs = (int(dataset.__len__())) // opt.testBatchSize

    for i in range(batch_idxs):
        images1, features1, images2, features2, feature_adjacent = next(iter_sample)

        images1 = images1.cuda()
        images2 = images2.cuda()
        features1 = features1.cuda().float()
        features2 = features2.cuda().float()

        feature_adjacent = feature_adjacent.cuda().float()

        image_stable1, image_unstable1, feature_stable1, feature_unstable1 = pre_propossing(images1, features1, )

        image_stable2, image_unstable2, feature_stable2, feature_unstable2 = pre_propossing(images2, features2, )

        # real_a.data.resize_(image_unstable1.size()).copy_(image_unstable1)
        # real_b.data.resize_(image_stable1.size()).copy_(image_stable1)

        # fake_b, grid = netG(real_a)

        fake1 = []
        fake2 = []
        grid1 = netG(image_unstable1[:, 0:period + 1, :, :])

        for nl in range(opt.num_layer):
            grid1[nl] = grid1[nl].permute(0, 2, 3, 1)
            fake1.append(functional.grid_sample(image_unstable1[:, period + 1:period + 1 + 3, :, :], grid1[nl]))

        grid2 = netG(image_unstable2[:, 0:period + 1, :, :])

        for nl in range(opt.num_layer):
            grid2[nl] = grid2[nl].permute(0, 2, 3, 1)
            fake2.append(functional.grid_sample(image_unstable2[:, period + 1:period + 1 + 3, :, :], grid2[nl]))

        loss_mse = 0
        loss_feature = 0
        loss_delta = 0
        loss_vgg = 0
        loss_g2 = 0
        for nl in range(opt.num_layer):
            loss_mse1, loss_delta1, loss_feature1 = loss_calulate(grid1[nl], feature_stable1, feature_unstable1,
                                                                  fake1[nl], image_stable1,opt.testBatchSize)

            loss_mse2, loss_delta2, loss_feature2 = loss_calulate(grid2[nl], feature_stable2, feature_unstable2,
                                                                  fake2[nl], image_stable2,opt.testBatchSize)
            loss_mse += loss_mse1 + loss_mse2
            loss_feature += loss_feature1 + loss_feature2
            loss_delta += loss_delta1 + loss_delta2
            loss_vgg += generator_criterion(fake1[nl].detach(), image_stable1[:,0:3,:,:].detach())
            loss_vgg += generator_criterion(fake2[nl].detach(), image_stable2[:,0:3,:,:].detach())
            loss_g2 += torch.mean(torch.abs(fake2[nl] - fake1[nl]))

        loss_g1 = loss_feature + loss_vgg + loss_mse  # + loss_affine * 20  # (delta1+delta2)*10+loss_affine*1000

        loss_g = loss_g1 + loss_g2 * opt.lamd
        print("===> val----Epoch[{}]({}/{})".format(epoch, i, batch_idxs))

        if i % 10 == 0:

            writer.add_scalar('scalar/val_loss_g_adjacent', loss_g2, i + epoch * batch_idxs)
            writer.add_scalar('scalar/val_loss_g_feature', loss_feature, i + epoch * batch_idxs)
            writer.add_scalar('scalar/val_loss_vgg', loss_vgg, i + epoch * batch_idxs)
            writer.add_scalar('scalar/val_delta', loss_delta, i + epoch * batch_idxs)
            #writer.add_scalar('scalar/loss_affine', loss_affine, i + epoch * batch_idxs)
            writer.add_scalar('scalar/val_mse', loss_mse, i + epoch * batch_idxs)
            # writer.add_scalar('scalar/loss', lossG, i + epoch * batch_idxs)

            writer.add_image('image/val_unstable',
                             vutils.make_grid(image_unstable1[0:3, period + 1:period + 1 + 3, :, :], normalize=True,
                                              scale_each=True),
                             i + epoch * batch_idxs)
            for nl in range(opt.num_layer):

                writer.add_image('image/val_stable_fake'+str(nl+1),
                                 vutils.make_grid(fake1[nl][0:3, :, :, :], normalize=True, scale_each=True),
                                 i + epoch * batch_idxs)

            writer.add_image('image/val_stable_real',
                             vutils.make_grid(image_stable1[0:3, 0:3, :, :], normalize=True, scale_each=True),
                             i + epoch * batch_idxs)


def checkpoint(epoch):
    if not os.path.exists("checkpoint"):
        os.mkdir("checkpoint")
    if not os.path.exists(os.path.join("checkpoint", opt.dataset)):
        os.mkdir(os.path.join("checkpoint", opt.dataset))
    state = {'net':netG.state_dict(), 'epoch':epoch}

    net_g_model_out_path = "checkpoint/{}/netG_model_epoch_{}.pth".format(opt.dataset, epoch)

    torch.save(state, net_g_model_out_path)
    print("Checkpoint saved to {}".format("checkpoint" + opt.dataset))


def process():
    """Test pix2pix"""
    # record all videos

    class_name = ['Regular', 'QuickRotation', 'Running', 'Parallax', 'Crowd', 'Zooming']
    crop_ratio = 0.2
    num_sample = 20
    stride_sample = 5
    threshold = 10
    rate = 0.3

    for ii in range(len(class_name) - 5):

        path = '/data/zmd/demo/datas/' + class_name[ii] + '/'
        # path = '/data/zmd/DeepStab/DeepStab/unstable_test/'

        list_videos = os.listdir(path)

        # index_sample_test = np.arange(30, 0, -1)
        index_sample_test = np.append(np.arange(-period // 2, 0, 1), np.arange(0, period // 2 + 1, 1))

        # index_sample_test = [32, 16, 8, 4, 2, 1]#[5, 10, 15, 20, 25, 30]#[1, 2, 4, 8, 16, 32]
        net_g_model_out_path = "checkpoint/{}/netG_model_epoch_{}.pth".format(opt.dataset, opt.continue_train)
        checkpoint = torch.load(net_g_model_out_path)
        netG.load_state_dict(checkpoint['net'])
        #netG = torch.load('./checkpoint/unet_512_kalman_10_period_30-1515/netG_model_epoch_35.pth')
        netG.eval()

        for video_id in range(0, len(list_videos)):
            '''
            ###calculate cropping

            cap = cv2.VideoCapture(path + list_videos[video_id])
            num_frame = cap.get(7)
            fps = cap.get(5)
            size_origin = (int(cap.get(3)), int(cap.get(4)))
            size = (256, 256)

            history_frame = []
            ret, img_init = cap.read()
            # img_init_resize = cv2.resize(img_init, size, interpolation=cv2.INTER_AREA)
            # img_init_resize_gray = cv2.cvtColor(img_init_resize, cv2.COLOR_BGR2GRAY)

            for i in range(period // 2 + 1):
                history_frame.append(img_init)
            # for i in range(index_sample_test[0]):
            for i in range(period // 2):
                ret, img_unstable = cap.read()
                # img_init_resize = cv2.resize(img_unstable, size, interpolation=cv2.INTER_AREA)
                history_frame.append(img_unstable)
            # last=img_init_resize
            for i in range(int(num_frame)):

                print('calculating cropping:::   video_index: ' + list_videos[video_id] + '  frame_index:  ' + str(i))
                list_unstable = []
                # list_stable=[]
                for j in range(len(index_sample_test)):
                    list_unstable.append(
                        cv2.resize(cv2.cvtColor(history_frame[period // 2 + index_sample_test[j]], cv2.COLOR_BGR2GRAY),
                                   size, interpolation=cv2.INTER_AREA))

                if (i >= int(num_frame) - period // 2):
                    img_unstable = history_frame[-1]


                else:
                    ret, img_unstable = cap.read()
                    if ret == False:
                        continue
                    # img_unstable = cv2.resize(img_unstable, size, interpolation=cv2.INTER_AREA)
                # img_unstable_resize=cv2.resize(img_unstable,size,interpolation=cv2.INTER_AREA)
                # img_unstable_resize = cv2.cvtColor(img_unstable_resize, cv2.COLOR_BGR2RGB)
                # list_unstable.append(img_unstable_resize)
                now = cv2.cvtColor(history_frame[period // 2], cv2.COLOR_BGR2RGB)
                now = now[np.newaxis, :, :, :]
                now = torch.from_numpy(now)
                now = now.cuda().float()
                now = now.float() * (1. / 255) * 2 - 1
                now = now.permute(0, 3, 1, 2)
                # np1 = np.array(list_stable)
                # np2 = np.array(list_unstable).transpose(0, 3, 1, 2).squeeze()
                # npp = np.concatenate((np1, np2), axis=0)

                npp = np.array(list_unstable)
                images = npp[np.newaxis, :, :, :]

                images = torch.from_numpy(images)
                images = images.cuda().float()
                images = images.float() * (1. / 255) * 2 - 1

                grid = netG(images)

                m = torch.nn.Upsample(scale_factor=(size_origin[1] / 256, size_origin[0] / 256), mode='bilinear')
                grid_resize = m(grid)
                grid_resize = grid_resize.permute(0, 2, 3, 1)

                pts1 = []
                pts2 = []
                for sample_height in range(num_sample):
                    for sample_width in range(num_sample):
                        pts1.append([grid_resize[
                                         0, size_origin[1] // 2 + (sample_height - num_sample // 2) * stride_sample,
                                         size_origin[0] // 2 + (
                                                 sample_width - num_sample // 2) * stride_sample * 2, 0].cpu().detach().numpy(),
                                     grid_resize[
                                         0, size_origin[1] // 2 + (sample_height - num_sample // 2) * stride_sample,
                                         size_origin[0] // 2 + (
                                                 sample_width - num_sample // 2) * stride_sample * 2, 1].cpu().detach().numpy()])
                        pts2.append([(size_origin[0] / 2 + (sample_width - num_sample / 2) * stride_sample * 2) /
                                     size_origin[0] * 2 - 1,
                                     (size_origin[1] // 2 + (sample_height - num_sample // 2) * stride_sample) /
                                     size_origin[1] * 2 - 1])

                pts1 = np.array(pts1, dtype=np.float32)
                pts2 = np.array(pts2, dtype=np.float32)

                M, mask = cv2.findHomography(pts1, pts2, cv2.RANSAC, 3.0)

                pts = np.float32([[-1, -1], [-1, 1], [1, 1], [1, -1]]).reshape(-1, 1, 2)

                dst = cv2.perspectiveTransform(pts, M)
                if (i == 0):
                    y_start = 0
                    x_start = 0
                    y_end = size_origin[1]
                    x_end = size_origin[0]
                else:
                    if (i > period // 2 and i < num_frame - period // 2):
                        y_start = (y_start + (max((dst[0, 0, 1] + 1) * size_origin[1] / 2,
                                                  (dst[3, 0, 1] + 1) * size_origin[
                                                      1] / 2) - y_start) * rate if y_start < max(
                            (dst[0, 0, 1] + 1) * size_origin[1] / 2,
                            (dst[3, 0, 1] + 1) * size_origin[1] / 2) else y_start)
                        y_end = (y_end - (y_end - min(y_end, min((dst[1, 0, 1] + 1) * size_origin[1] / 2,
                                                                 (dst[2, 0, 1] + 1) * size_origin[
                                                                     1] / 2))) * rate if y_end > min(
                            (dst[1, 0, 1] + 1) * size_origin[1] / 2,
                            (dst[2, 0, 1] + 1) * size_origin[1] / 2) else y_end)
                        x_start = (x_start + (max((dst[0, 0, 0] + 1) * size_origin[0] / 2,
                                                  (dst[1, 0, 0] + 1) * size_origin[
                                                      0] / 2) - x_start) * rate if x_start < max(
                            (dst[0, 0, 0] + 1) * size_origin[0] / 2,
                            (dst[1, 0, 0] + 1) * size_origin[0] / 2) else x_start)
                        # x_start = max(x_start,max((dst[0, 0, 0] + 1) * size_origin[0] / 2, (dst[1, 0, 0] + 1) * size_origin[0] / 2))

                        x_end = (x_end - (x_end - min(x_end, min((dst[2, 0, 0] + 1) * size_origin[0] / 2,
                                                                 (dst[3, 0, 0] + 1) * size_origin[
                                                                     0] / 2))) * rate if x_end > min(
                            (dst[2, 0, 0] + 1) * size_origin[0] / 2,
                            (dst[3, 0, 0] + 1) * size_origin[0] / 2) else x_end)

                    # x_end = min(x_end,min((dst[2, 0, 0] + 1) * size_origin[0] / 2, (dst[3, 0, 0] + 1) * size_origin[0] / 2))

                history_frame.pop(0)

                history_frame.append(img_unstable)
            '''
            #################calculate cropping##########
            x_start = 0
            x_end = 640
            y_start = 0
            y_end = 360
            cap = cv2.VideoCapture(path + list_videos[video_id])
            num_frame = cap.get(7)
            fps = cap.get(5)
            size_origin = (int(cap.get(3)), int(cap.get(4)))
            # size_crop=(int(cap.get(3)*(1-crop_ratio)),int(cap.get(4)*(1-crop_ratio*1.5)))
            size = (256, 256)
            size_crop = (int(x_end) - int(x_start) - threshold * 2, int(y_end) - int(y_start) - threshold * 2)
            fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
            videoWriter = cv2.VideoWriter(
                './test/stable_diff_frequency_0.1/' + class_name[ii] + '/' + str(list_videos[video_id]) + '.avi',
                fourcc, int(fps), size_crop)
            history_frame = []
            ret, img_init = cap.read()
            # img_init_resize = cv2.resize(img_init, size, interpolation=cv2.INTER_AREA)
            # img_init_resize_gray = cv2.cvtColor(img_init_resize, cv2.COLOR_BGR2GRAY)

            for i in range(period // 2 + 1):
                history_frame.append(img_init)
            # for i in range(index_sample_test[0]):
            for i in range(period // 2):
                ret, img_unstable = cap.read()
                # img_init_resize = cv2.resize(img_unstable, size, interpolation=cv2.INTER_AREA)
                history_frame.append(img_unstable)
            # last=img_init_resize
            for i in range(int(num_frame)):

                print('video_index: ' + list_videos[video_id] + '  frame_index:  ' + str(i) + '     cropping :' + str(
                    x_start) + '    ' + str(x_end) + '   ' + str(y_start) + '  ' + str(y_end))
                list_unstable = []
                # list_stable=[]
                for j in range(len(index_sample_test)):
                    list_unstable.append(
                        cv2.resize(cv2.cvtColor(history_frame[period // 2 + index_sample_test[j]], cv2.COLOR_BGR2GRAY),
                                   size, interpolation=cv2.INTER_AREA))

                if (i >= int(num_frame) - period // 2):
                    img_unstable = history_frame[-1]


                else:
                    ret, img_unstable = cap.read()
                    if ret == False:
                        continue
                    # img_unstable = cv2.resize(img_unstable, size, interpolation=cv2.INTER_AREA)
                # img_unstable_resize=cv2.resize(img_unstable,size,interpolation=cv2.INTER_AREA)
                # img_unstable_resize = cv2.cvtColor(img_unstable_resize, cv2.COLOR_BGR2RGB)
                # list_unstable.append(img_unstable_resize)
                now = cv2.cvtColor(history_frame[period // 2], cv2.COLOR_BGR2RGB)
                now = now[np.newaxis, :, :, :]
                now = torch.from_numpy(now)
                now = now.cuda().float()
                now = now.float() * (1. / 255) * 2 - 1
                now = now.permute(0, 3, 1, 2)
                # np1 = np.array(list_stable)
                # np2 = np.array(list_unstable).transpose(0, 3, 1, 2).squeeze()
                # npp = np.concatenate((np1, np2), axis=0)

                npp = np.array(list_unstable)
                images = npp[np.newaxis, :, :, :]

                images = torch.from_numpy(images)
                images = images.cuda().float()
                images = images.float() * (1. / 255) * 2 - 1

                grid_whole = netG(images)
                grid=grid_whole[2]
                m = torch.nn.Upsample(scale_factor=(size_origin[1] / 256, size_origin[0] / 256), mode='bilinear')
                grid_resize = m(grid)
                grid_resize = grid_resize.permute(0, 2, 3, 1)

                fake = functional.grid_sample(now, grid_resize)

                # update history_frame
                # fake_gray=cv2.cvtColor(fake[0,:,:,:].data.cpu().numpy(),cv2.COLOR_BGR2GRAY)
                # history_frame.pop(0)
                # history_frame.append(fake_gray)

                samples = fake[0, :, :, :].data.cpu().numpy()
                samples = (samples + 1) * 127.5
                samples = samples.transpose((1, 2, 0))
                # samples = samples.reshape((144, 256,-1))
                samples = np.array(samples.astype(np.uint8))

                # update history_frame
                # fake_gray=cv2.cvtColor(samples,cv2.COLOR_BGR2GRAY)
                history_frame.pop(0)

                history_frame.append(img_unstable)

                # samples_resize=cv2.resize(samples,size_origin,interpolation=cv2.INTER_AREA)

                samples = cv2.cvtColor(samples, cv2.COLOR_BGR2RGB)
                # samples_resize = samples[int(cap.get(4) * crop_ratio):int(cap.get(4) * crop_ratio ) + int(size_crop[1]),int(cap.get(3) * crop_ratio / 2):int(cap.get(3) * crop_ratio / 2) + int(size_crop[0]), :]
                samples_resize = samples[int(y_start) + threshold:int(y_end) - threshold,
                                 int(x_start) + threshold:int(x_end) - threshold, :]
                samples_resize = cv2.GaussianBlur(samples_resize, (3, 3), 0.2, 0)

                cv2.imshow('show_unstable.jpg', history_frame[period // 2])
                cv2.imshow('show_stable.jpg', samples_resize)
                # cv2.imshow('show_crop.jpg',samples_resize)
                cv2.waitKey(1)
                videoWriter.write(samples_resize)  # 写视频帧

            videoWriter.release()



def main():
    if opt.train:
        list_stable,list_unstable,list_feature,list_adjacent=image_store(train_files)
        list_stable_val,list_unstable_val,list_feature_val,list_adjacent_val=image_store(val_files)

        list_epoch = list_random(train_files)
        list_epoch_val = list_random(val_files)

        for epoch in range(opt.continue_train, opt.nEpochs):

                if epoch % 5 == 1:
                    #test(epoch, list_stable_val, list_unstable_val, list_feature_val, list_adjacent_val, list_epoch_val)
                    checkpoint(epoch)

                lr = opt.lr * 0.1 ** int(epoch / 20)
                train(epoch, lr,list_stable,list_unstable,list_feature,list_adjacent,list_epoch)
    else:
        process()


if __name__ == '__main__':
    main()


'''
process()
'''
