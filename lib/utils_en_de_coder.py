
from __future__ import print_function
import torch
from torch.utils.data import Dataset, DataLoader
import cv2
import itertools
from lib.cfg_en_de_coder import *
import numpy as np


def list_random_batchsize(files):
    list_random = []

    for video_id in range(0, len(files)):
        feature_path = opt.path_feature + str(files[video_id]) + '.avi.txt'
        f = open(feature_path, "r")

        numframe = (list(map(int, f.readline().split()))[0]-period-1)//opt.batchSize*opt.batchSize
        for i in range(period, numframe):
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
        for i in range(period, numframe):
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
            image_stable = self.list_stable[video_id][numframe_id - index_sample[j]]
            image_stable = cv2.cvtColor(image_stable, cv2.COLOR_RGB2GRAY)
            list_stable.append(image_stable)

        image_unstable = self.list_unstable[video_id][numframe_id]  # cv2.imread(image_path_unstable + str(int(numframe_id)) + '.png')
        list_unstable.append(image_unstable)

        image_stable = self.list_stable[video_id][numframe_id]  # cv2.imread(image_path_stable + str(int(numframe_id)) + '.png')
        list_stable_unstable.append(image_stable)



        if opt.use_gan and self.with_gan:
            for j in range(0, len(index_sample_discriminator)):
                image_stable = self.list_stable[video_id][numframe_id - index_sample_discriminator[j]]
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

        return img_data1, feature_data1, affine_data1,  # stable->unstable


def pre_propossing(images, features):
    images = images.float() * (1. / 255) * 2 - 1
    images_unstable = images[:, 0:len(index_sample) + 3, :, :]
    images_stable = images[:, len(index_sample)+3:len(index_sample)+3+3, :, :]
    images_gan=images[:, len(index_sample)+3+3:, :, :]
    feature_stable = features[:, :, 0:3]
    feature_unstable = features[:, :, 3:6]
    feature_stable = feature_stable.permute(0, 2, 1)
    feature_unstable = feature_unstable.permute(0, 2, 1)
    return images_stable, images_unstable, images_gan,feature_stable, feature_unstable


def loss_pixel(grid,affine):
    # target_height = opt.input_size
    # target_width = opt.input_size
    # HW = target_height * target_width
    # target_coordinate = list(itertools.product(range(target_height), range(target_width)))
    # target_coordinate = torch.Tensor(target_coordinate)  # HW x 2
    # Y, X = target_coordinate.split(1, dim=1)
    # Y = Y * 2 / (target_height - 1) - 1
    # X = X * 2 / (target_width - 1) - 1
    # target_coordinate = torch.cat([X, Y], dim=1)  # convert from (y, x) to (x, y)
    # #One = torch.ones(X.size())
    # stable = target_coordinate#torch.cat([target_coordinate, One], dim=1)
    # stable = stable.expand(opt.batchSize, target_height * target_width, 2).permute(0, 2, 1)
    # stable=stable.reshape([opt.batchSize,2, opt.input_size,opt.input_size])
    # stable = stable.cuda()
    #grid_reshape = grid.view(-1, target_height * target_width, 2).permute(0, 2, 1)
    #grid=grid.permute(0,3,1,2)
    # affine_loss = torch.mean(torch.abs(torch.matmul(affine, stable.float()) - grid_reshape.float()))

    boundary = [[-1, -1, 1], [1, -1, 1], [-1, 1, 1], [1, 1, 1]]
    boundary = torch.Tensor(boundary)
    boundary = boundary.expand(opt.batchSize, 4, 3).permute(0, 2, 1)
    boundary = boundary.cuda().float()
    affine = affine.view(-1, 2, 3)
    bound_batch = torch.matmul(affine, boundary.float())
    x_start = bound_batch[:, 0, [0, 2]].cpu().numpy()
    x_end = bound_batch[:, 0, [1, 3]].cpu().numpy()
    y_start = bound_batch[:, 1, [0, 1]].cpu().numpy()
    y_end = bound_batch[:, 1, [2, 3]].cpu().numpy()

    homo_loss=0
    homo_loss1=0
    for i in range(opt.batchSize):
        x_s = max(max(x_start[i]), -1)
        x_e = min(min(x_end[i]), 1)
        y_s = max(max(y_start[i]), -1)
        y_e = min(min(y_end[i]), 1)




        # pts1 = []
        # pts2 = []

        sample_num = 3
        random_loc = np.random.rand(1, 4)  # x,y,w,h
        crop_thres=0
        # pts1 = torch.FloatTensor(pow(sample_num, 2) * 2, 8).cuda()
        # pts2 = torch.FloatTensor(pow(sample_num, 2) * 2, 1).cuda()

        pts11=torch.tensor([]).cuda()
        pts21=torch.tensor([]).cuda()
        x_e=int((x_e+1)*opt.input_size/2)-sample_num-crop_thres
        x_s = int(max(min((x_s + 1) * opt.input_size / 2+crop_thres, x_e), 0))
        y_e =int((y_e + 1) * opt.input_size / 2)-sample_num-crop_thres
        y_s = int(max(min((y_s + 1) * opt.input_size / 2+crop_thres, y_e), 0))

        x_random = int(x_s + (x_e - x_s) * random_loc[0][0])
        w_sample = int(max(((x_e - x_random) * random_loc[0][2]) // sample_num, 1))
        y_random = int(y_s + (y_e - y_s) * random_loc[0][1])
        h_sample = int(max(((y_e - y_random) * random_loc[0][3]) // sample_num, 1))
        index=-1
        list_xy=[]
        for x in np.arange(x_random,x_random+sample_num*w_sample,w_sample):
            for y in np.arange(y_random,y_random+sample_num*h_sample,h_sample):

                list_xy.append([x,y])



                #
                index=index+1
                # X = grid[i, x, y, 0].cpu().detach().numpy()
                # Y = grid[i, x, y, 1].cpu().detach().numpy()
                # Xp = float(x) / opt.input_size*2 - 1
                # Yp = float(y) / opt.input_size*2 - 1
                # pts1[index]=torch.tensor([grid[i, y, x, 0],grid[i, y, x, 1],1,0,0,0,-Xp *grid[i, y, x, 0] , -Xp * grid[i, y, x, 1]],dtype=torch.float)
                # pts2[index] = Xp
                # index=index+1
                # pts1[index] = torch.tensor([0,0,0,grid[i, y, x, 0], grid[i, y, x, 1], 1, -Yp * grid[i, y, x, 0], -Yp * grid[i, y, x, 1]],dtype=torch.float)
                # pts2[index] = Yp
                # pts1.append([X, Y, 1, 0, 0, 0, -Xp * X, -Xp * Y])
                # pts1.append([0, 0, 0, X, Y, 1, -Yp * X, -Yp * Y])
                # pts2.append(Xp)
                # pts2.append(Yp)


        list_xy_float=np.array(list_xy, dtype=np.float32)
        list_xy = np.array(list_xy, dtype=np.int32)
        #
        Xp=list_xy_float[:,0]/opt.input_size*2-1
        Yp=list_xy_float[:,1]/opt.input_size*2-1
        Xp_tensor=torch.from_numpy(Xp).view(pow(sample_num,2),1).cuda()
        Yp_tensor=torch.from_numpy(Yp).view(pow(sample_num,2),1).cuda()
        tensor_one=torch.ones(pow(sample_num,2),1).cuda()
        tensor_zero=torch.zeros(pow(sample_num,2),3).cuda()
        grid0=(grid[i,list_xy[:,1],list_xy[:,0],0].view(pow(sample_num,2),1))
        grid1=(grid[i,list_xy[:,1],list_xy[:,0],1].view(pow(sample_num, 2),1))
        # # print(grid[i,list_xy[0,1],list_xy[0,0],0])
        # # print(grid[i, list_xy[0, 1], list_xy[0, 0], 1])
        # # print(list_xy[0])
        #
        #


        pts11=torch.cat((pts11,torch.cat((grid0,grid1,tensor_one,tensor_zero,torch.mul(-Xp_tensor,grid0),torch.mul(-Xp_tensor,grid1)),1)),0)
        pts21=torch.cat((pts21,Xp_tensor),0)
        pts11 = torch.cat((pts11,torch.cat((tensor_zero,grid0, grid1, tensor_one, torch.mul(-Yp_tensor, grid0), torch.mul(-Yp_tensor, grid1)),1)),0)
        pts21 = torch.cat((pts21, Yp_tensor),0)
        pts11=pts11.double()
        pts21=pts21.double()
        grid1 = grid1.double()
        grid0 = grid0.double()
        # pts1.requires_grad = True
        # pts2.requires_grad = True
        # if torch.det(torch.matmul(torch.t(pts1),pts1))>0:
        #     A=torch.matmul(pts1,torch.matmul(torch.matmul(torch.inverse(torch.matmul(torch.t(pts1),pts1)),torch.t(pts1)),pts2))
        #     homo_loss+=torch.dist(A,pts2,2)*np.exp(-max(w_sample,h_sample)/10)


        if torch.det(torch.matmul(torch.t(pts11), pts11)) > 0:


            H=torch.matmul(torch.matmul(torch.inverse(torch.matmul(torch.t(pts11), pts11)), torch.t(pts11)),pts21)

            # H1=torch.matmul(torch.inverse(pts11),pts21)

            # pts111=pts11.clone()
            # pts211=pts21.clone()
            # for p in range(4):
            #     pts111[2*p]=pts11[p]
            #     pts111[2*p+1]=pts11[p+4]
            #     pts211[2 * p] = pts21[p]
            #     pts211[2 * p + 1] = pts21[p + 4]
            #
            # H11 = torch.matmul(torch.matmul(torch.inverse(torch.matmul(torch.t(pts111), pts111)), torch.t(pts111)), pts211)
            # H111 = torch.matmul(torch.inverse(pts111), pts211)

            #fenzi = torch.matmul(pts11,torch.matmul(torch.matmul(torch.inverse(torch.matmul(torch.t(pts11), pts11)), torch.t(pts11)),pts21))
            fenmu = torch.tensor([]).cuda()
            fenmu=fenmu.double()
            fenzi = torch.tensor([]).cuda()
            fenzi = fenzi.double()
            af = torch.tensor([]).cuda()
            af = af.double()
            affine=affine.double()
            for p in range(pow(sample_num,2)):
                    fenzi=torch.cat((fenzi,H[0]*grid0[p]+H[1]*grid1[p]+H[2]))
                    fenmu=torch.cat((fenmu,H[6]*grid0[p]+H[7]*grid1[p]+1))
                    af=torch.cat((af,affine[i,0,0]*grid0[p]+affine[i,0,1]*grid1[p]+affine[i,0,2]))

            for p in range(pow(sample_num, 2)):
                    fenzi = torch.cat((fenzi, H[3] * grid0[p] + H[4] * grid1[p] + H[5]))
                    fenmu = torch.cat((fenmu, H[6] * grid0[p] + H[7] * grid1[p] + 1))
                    af = torch.cat((af, affine[i,1, 0] * grid0[p] + affine[i,1, 1] * grid1[p] + affine[i,1, 2]))
            fenmu=fenmu.view(pow(sample_num,2)*2,1)
            fenzi = fenzi.view(pow(sample_num, 2) * 2, 1)
            # fenzi=fenzi.double()
            # fenmu=fenmu.double()
            result=torch.mul(fenzi,1/fenmu)
            # if torch.dist(result, pts21, 2)<0.1:
            af=af.view(pow(sample_num,2)*2,1)
            if torch.dist(result, af, 2)<0.05:
                homo_loss += torch.dist(result, pts21, 2)* (np.exp(min(w_sample, h_sample) /opt.input_size*2)-1)
                print("yse")
            #else:
            #    print("wrong matching\n")
        # if np.linalg.det(np.dot(pts1.T, pts1)) != 0:
        #     #print (np.linalg.norm(np.dot(np.dot(pts1, np.linalg.inv(np.dot(pts1.T, pts1))), np.dot(pts1.T, pts2)) - pts2))
        #     homo_loss += np.linalg.norm(np.dot(np.dot(pts1, np.linalg.inv(np.dot(pts1.T, pts1))), np.dot(pts1.T, pts2)) - pts2)*np.exp(-max(w_sample,h_sample)/10)




    # pooling_size=[1,8,32]
    # stride_size=[1,4,16]
    # delta=0
    # for i in range(3):
    #     pooling_operation = nn.AvgPool2d(pooling_size[i], stride=stride_size[i],padding=stride_size[i]//2)
    #     grid_temp=pooling_operation(grid)
    #     stable_temp=pooling_operation(stable)
    #
    #
    #
    #     variation=grid_temp-stable_temp
    #
    #     delta_x = torch.abs(variation[:, :, 0: -1, :] - variation[:, :, 1:,:])
    #     delta_y = torch.abs(variation[:, :, :, 0: -1] - variation[:, :,:, 1:])
    #
    #     delta_temp = (torch.mean(delta_x) + torch.mean(delta_y)) / 2
    #
    #     delta+=delta_temp

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

    return homo_loss
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
