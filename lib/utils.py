
from __future__ import print_function
import torch
from torch.utils.data import Dataset, DataLoader
import cv2
import itertools
from lib.cfg import *
from torch import nn
from torchvision.models.vgg import vgg16

class GeneratorLoss(nn.Module):
    def __init__(self):
        super(GeneratorLoss, self).__init__()
        vgg = vgg16(pretrained=True)
        loss_network = nn.Sequential(*list(vgg.features)[:31]).eval()
        loss_network = torch.nn.DataParallel(loss_network)
        for param in loss_network.parameters():
            param.requires_grad = False
        self.loss_network = loss_network
        self.mse_loss = nn.MSELoss()
        #self.tv_loss = TVLoss()

    def forward(self, out_images, target_images):
        # Adversarial Loss
        # adversarial_loss = torch.mean(1 - out_labels)
        # Perception Loss
        perception_loss = self.mse_loss(self.loss_network(out_images), self.loss_network(target_images))
        # Image Loss
        # image_loss = self.mse_loss(out_images, target_images)
        # TV Loss
        #tv_loss = self.tv_loss(out_images)
        return perception_loss# + 2e-8 * tv_loss
def list_random_batchsize(files):
    list_random = []

    for video_id in range(0, len(files)):
        feature_path = opt.path_feature + str(files[video_id]) + '.avi.txt'
        f = open(feature_path, "r")

        numframe = (list(map(int, f.readline().split()))[0]-period-1)//opt.batchSize*opt.batchSize
        for i in range(0 + period // 2, numframe + period//2, 1):
            video_frame = [video_id, i]
            list_random.append(video_frame)
        f.close()
    return list_random

def list_shuffle(list_random):
    length=int(len(list_random)/opt.batchSize)
    list=np.arange(length)
    np.random.shuffle(list)
    list_random_shuffle=[]
    for i in range(length):
        for j in range(opt.batchSize):
            list_random_shuffle.append(list_random[list[i]*opt.batchSize+j])
    return list_random_shuffle


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
    list_affine=[]

    for video_id in range(0, len(files)):
        print('image_store---' + '[' +str(video_id)+ '/'+ str(len(files))+']')
        feature_path = opt.path_feature + str(files[video_id]) + '.avi.txt'
        adjacent_path = opt.path_adjacent + str(files[video_id]) + '.avi.txt'
        affine_path = opt.path_affine + str(files[video_id]) + '.avi.txt'
        f = open(feature_path, "r")
        f_adjacent = open(adjacent_path, "r")
        f_affine = open(affine_path, "r")
        list_unstable_clip = []
        list_stable_clip = []
        feature_clip = []
        adjacent_clip = []
        affine_clip = []
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

            line = f_affine.readline()
            line_re = line.split()
            affine_clip.append(list(map(float, line_re)))
        f.close()
        f_adjacent.close()
        f_affine.close()
        list_unstable.append(list_unstable_clip)
        list_stable.append(list_stable_clip)
        list_feature.append(feature_clip)
        list_adjacent.append(adjacent_clip)
        list_affine.append(affine_clip)
    return list_stable, list_unstable,list_feature,list_adjacent,list_affine


class customData(Dataset):
    def __init__(self, files, list_random, list_stable, list_unstable,list_feature,list_adjacent,list_affine,with_gan):
        self.feature_all_all = list_feature
        self.affine_all_all=list_affine
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
                image_stable = self.list_stable[video_id][numframe_id - index_sample_discriminator[len(index_sample_discriminator) - 1 - j]]
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

        affine_data1 = np.array(self.affine_all_all[video_id][int(numframe_id)])


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
                image_stable = self.list_stable[video_id][numframe_id - index_sample_discriminator[len(index_sample_discriminator) - 1 - j]]
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

        affine_data2 = np.array(self.affine_all_all[video_id][int(numframe_id)])

        feature_one_adjacent = np.array(self.feature_all_adjacent[video_id][int(numframe_id) - 1])

        return img_data1, feature_data1, affine_data1, img_data2, feature_data2, affine_data2, feature_one_adjacent  # stable->unstable


def pre_propossing(images, features):
    images = images.float() * (1. / 255) * 2 - 1
    images_unstable = images[:, 0:period + 1 + 3, :, :]
    images_stable = images[:, period + 1 + 3:, :, :]
    feature_stable = features[:, :, 0:3]
    feature_unstable = features[:, :, 3:6]
    feature_stable = feature_stable.permute(0, 2, 1)
    feature_unstable = feature_unstable.permute(0, 2, 1)
    return images_stable, images_unstable, feature_stable, feature_unstable


def loss_pixel(grid,affine):
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
    grid_reshape = grid.view(-1, target_height * target_width, 2).permute(0, 2, 1)
    # grid=grid.permute(0,3,1,2)
    affine_loss = torch.mean(torch.abs(torch.matmul(affine, stable.float()) - grid_reshape.float()))



    return affine_loss
def frame_clip(sequence, affine):
    sequence_tensor=torch.Tensor(sequence.size())
    boundary = [[-1, -1, 1], [1, -1, 1], [-1, 1, 1], [1, 1, 1]]
    boundary = torch.Tensor(boundary)
    boundary = boundary.expand(opt.batchSize, 4,3).permute(0, 2, 1)
    boundary=boundary.cuda().float()
    affine = affine.view(-1, 2, 3)
    bound_batch=torch.matmul(affine, boundary.float())
    x_start=bound_batch[:,0,[0,2]].cpu().numpy()
    x_end = bound_batch[:, 0, [1, 3]].cpu().numpy()
    y_start = bound_batch[:, 1, [0, 1]].cpu().numpy()
    y_end = bound_batch[:, 1, [2, 3]].cpu().numpy()


    for i in range(opt.batchSize):
        x_s = max(max(x_start[i]),-1)
        x_e = min(min(x_end[i]),1)
        y_s = max(max(y_start[i]),-1)
        y_e = min(min(y_end[i]), 1)

        m = torch.nn.Upsample(size=opt.input_size, mode='bilinear')
        temp=torch.unsqueeze(sequence[i, :, int((y_s + 1) * opt.input_size / 2):int((y_e + 1) * opt.input_size / 2), int((x_s + 1) * opt.input_size / 2):int((x_e + 1) * opt.input_size / 2)],0)
        sequence_tensor[i] = m(temp)
    return sequence_tensor

def frame_clip_batchsize(sequence, affine):
    sequence_tensor=torch.Tensor(sequence.size())
    boundary = [[-1, -1, 1], [1, -1, 1], [-1, 1, 1], [1, 1, 1]]
    boundary = torch.Tensor(boundary)
    boundary = boundary.expand(opt.batchSize, 4,3).permute(0, 2, 1)
    boundary=boundary.cuda().float()
    affine = affine.view(-1, 2, 3)
    bound_batch=torch.matmul(affine, boundary.float())
    x_start=bound_batch[:,0,[0,2]].cpu().numpy()
    x_end = bound_batch[:, 0, [1, 3]].cpu().numpy()
    y_start = bound_batch[:, 1, [0, 1]].cpu().numpy()
    y_end = bound_batch[:, 1, [2, 3]].cpu().numpy()


    x_s_sum=-1
    x_e_sum=1
    y_s_sum=-1
    y_e_sum=1
    for i in range(opt.batchSize):
        x_s = max(max(x_start[i]),-1)
        x_e = min(min(x_end[i]),1)
        y_s = max(max(y_start[i]),-1)
        y_e = min(min(y_end[i]), 1)
        x_s_sum=max(x_s_sum,x_s)
        x_e_sum=min(x_e_sum,x_e)
        y_s_sum=max(y_s_sum,y_s)
        y_e_sum=min(y_e_sum,y_e)

    for i in range(opt.batchSize):
        m = torch.nn.Upsample(size=opt.input_size, mode='bilinear')
        temp=torch.unsqueeze(sequence[i, :, int((y_s_sum + 1) * opt.input_size / 2):int((y_e_sum + 1) * opt.input_size / 2), int((x_s_sum + 1) * opt.input_size / 2):int((x_e_sum + 1) * opt.input_size / 2)],0)
        sequence_tensor[i] = m(temp)
    return sequence_tensor


def loss_calulate(grid, feature_stable, feature_unstable, fake, real,batchSize):
    feature_loss = 0
    for i in range(batchSize):
        grid_pos = grid[i, ((feature_stable[i, 1, :] + 1) * opt.input_size / 2).int().cpu().numpy(),
                   ((feature_stable[i, 0, :] + 1) * opt.input_size / 2).int().cpu().numpy(), :]
        feature_loss = feature_loss + torch.pow(torch.dist(feature_unstable[i, 0:2, :], torch.t(grid_pos)),2) / opt.number_feature
        #feature_loss = feature_loss + torch.mean(torch.abs(feature_unstable[i, 0:2, :] - torch.t(grid_pos)))

    feature_loss = feature_loss / opt.batchSize
    # loss = mse_loss + args.balance_para * feature_loss
    # loss_f = torch.nn.MSELoss()
    mse_loss = torch.mean(torch.abs(real[:,0:3,:,:] - fake))

    delta_x = torch.abs(grid[:, :, 0:opt.input_size - 1, :] - grid[:, :, 1:opt.input_size, :])
    delta_y = torch.abs(grid[:, 0:opt.input_size - 1, :, :] - grid[:, 1:opt.input_size, :, :])

    # delta_xx = torch.abs(delta_x[:, :, 0:-1, :] - delta_x[:, :, 1:, :])
    # delta_yy = torch.abs(delta_y[:, 0:-1, :, :] - delta_y[:, 1:, :, :])

    delta = (torch.mean(delta_x) + torch.mean(delta_y)) / 2

    # delta_x=torch.abs(grid[])

    return mse_loss,delta, feature_loss  # ,loss_sim


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

def generate_maps(grid,batchsize=opt.batchSize):


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
    stable = stable.expand(batchsize, target_height * target_width, 2)
    stable=stable.reshape([batchsize, opt.input_size,opt.input_size,2])
    grid=grid+stable.permute(0,3,1,2).cuda()
    return grid

def loss_pixel1(B_grid,grid_eye,A_tensor_whole,W,H,size):

    # B_grid = grid-grid_eye
    # loss_affine=0
    B_tensor=torch.Tensor().cuda().to(torch.float64)
    A_tensor=torch.Tensor().cuda().to(torch.float64)
    B_grid=B_grid.to(torch.float64)

    for i in range(H):
         for j in range(W):
             B_tensor_temp = torch.reshape(B_grid[:, size // H * i:size // H * (i + 1), size // W * j:size // W * (j + 1), :],[opt.batchSize, size // H * size // W, 2])
             A_tensor_temp = torch.reshape(A_tensor_whole[:, size // H * i:size // H * (i + 1), size // W * j:size // W * (j + 1), :, :],[opt.batchSize, size // H * size // W, -1])
             B_tensor = torch.cat((B_tensor,B_tensor_temp),dim=0)
             A_tensor = torch.cat((A_tensor, A_tensor_temp), dim=0)


    AB = torch.bmm(A_tensor, torch.bmm(torch.bmm(torch.inverse(torch.bmm(A_tensor.permute(0, 2, 1), A_tensor)), A_tensor.permute(0, 2, 1)), B_tensor))
    loss_affine = torch.dist(AB, B_tensor, 1).to(torch.float32)
    B_grid = B_grid.to(torch.float32)

    return loss_affine

def generate_affine_matrix(width,height):
    x2 = width - 1
    y2 = height - 1
    x1 = 0
    y1 = 0
    y = np.arange(height).reshape([height, -1])
    y = np.tile(y, (1, width))
    x = np.arange(width)
    x = np.tile(x, (height, 1))

    xy = np.dstack((x, y)).reshape(height * width, -1)
    Q11 = ((x2 - xy[:, 0]) * (y2 - xy[:, 1]) / (x2 * y2)).reshape(height, width, 1)
    Q21 = ((xy[:, 0] - x1) * (y2 - xy[:, 1]) / (x2 * y2)).reshape(height, width, 1)
    Q12 = ((x2 - xy[:, 0]) * (xy[:, 1] - y1) / (x2 * y2)).reshape(height, width, 1)
    Q22 = ((xy[:, 0] - x1) * (xy[:, 1] - y1) / (x2 * y2)).reshape(height, width, 1)
    Q = np.concatenate((Q11, Q21, Q12, Q22), axis=2)
    # QQ=np.repeat(np.expand_dims(Q,axis=2),2,axis=2)
    QQ = np.expand_dims(Q, axis=2)

    QQQ = np.tile(QQ, (width, height, 1, 1))
    return QQQ