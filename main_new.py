from __future__ import print_function
import os
import torch.optim as optim
from torch.utils.data import Dataset
from lib.networks_cascading import define_G, define_D, GANLoss
import random
import torchvision.utils as vutils
from lib.utils import GeneratorLoss
import torch.nn.functional as functional
import time
from lib.utils import *

from visdom import Visdom

# os.environ["CUDA_VISIBLE_DEVICES"] = "6"
print(opt)
if opt.mode=='train':
    viz = Visdom(server='http://127.0.0.1', port=opt.visdom_port)

if opt.use_gan:
    criterionGAN = GANLoss()

feature_all_all = []

if opt.cuda and not torch.cuda.is_available():
    raise Exception("No GPU found, please run without --cuda")

print('===> Building model')

netG = define_G(opt.input_nc, opt.output_nc, opt.ngf, 'normal', 0.02)
if opt.use_gan:

    netD = define_D(opt.batchSize, opt.ndf, 'n_layers', n_layers_D=4, norm='batch', use_sigmoid=False)


# setup optimizer
generator_criterion=GeneratorLoss()
torch.manual_seed(opt.seed)
if opt.cuda:
    netG = netG.cuda()
    if opt.use_gan:
        netD = netD.cuda()
    generator_criterion = generator_criterion.cuda()
    torch.cuda.manual_seed_all(opt.seed)
    if opt.use_gan:
        criterionGAN.cuda()



def train(epoch, lr, list_stable, list_unstable, list_feature, list_adjacent, list_affine, list_epoch):
    if epoch>opt.start_gan:
        with_gan=True
    else:
        with_gan=False
    netG.train()
    if opt.continue_train>0 and epoch==opt.continue_train:
         net_g_model_out_path = "checkpoint/{}/netG_model_epoch_{}.pth".format(opt.checkpoint_dir, opt.continue_train)
         checkpoint=torch.load(net_g_model_out_path)
         netG.load_state_dict(checkpoint['net'])


    optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(opt.beta1, 0.999))

    if opt.use_gan and with_gan:
        optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(opt.beta1, 0.999))
        list_epoch=list_shuffle(list_epoch)
    else:
        random.shuffle(list_epoch)
    dataset = customData(train_files, list_epoch, list_stable, list_unstable,list_feature,list_adjacent,list_affine, with_gan)
    sample = torch.utils.data.DataLoader(dataset, batch_size=opt.batchSize, shuffle=False,num_workers=opt.threads,pin_memory=True)
    iter_sample = iter(sample)

    batch_idxs = (int(dataset.__len__())) // opt.batchSize

    if opt.shapeloss:
        A_affine = torch.from_numpy(generate_affine_matrix(opt.block, opt.block)).cuda().float().unsqueeze(0).repeat(opt.batchSize, 1,1, 1, 1).to(torch.float64)
        eye = torch.tensor([[1, 0, 0], [0, 1, 0]], dtype=torch.float).unsqueeze(0).expand(opt.batchSize, 2, 3)
        grid_eye = functional.affine_grid(eye, torch.Size((opt.batchSize, 3, opt.input_size, opt.input_size))).cuda()

    for i in range(batch_idxs):
        time_begin = time.time()
        images1, features1, affine1, images2, features2, affine2, feature_adjacent = next(iter_sample)

        images1 = images1.cuda()
        images2 = images2.cuda()
        features1 = features1.cuda().float()
        features2 = features2.cuda().float()
        affine1=affine1.cuda().float()
        affine2=affine2.cuda().float()

        feature_adjacent = feature_adjacent.cuda().float()

        image_stable1, image_unstable1, feature_stable1, feature_unstable1 = pre_propossing(images1, features1)

        image_stable2, image_unstable2, feature_stable2, feature_unstable2 = pre_propossing(images2, features2)

        fake1=[]
        fake2=[]

        grid1= netG(image_unstable1[:, 0:period + 1, :, :])

        for nl in range(opt.num_layer):
            grid1[nl]=generate_maps(grid1[nl]).permute(0,2,3,1)
            fake1_temp = functional.grid_sample((image_unstable1[:, period + 1:period + 1 + 3, :, :] + 1) * 127.5,grid1[nl])
            fake1.append(fake1_temp / 127.5 - 1)
            # fake1.append(functional.grid_sample(image_unstable1[:, period + 1:period + 1 + 3, :, :], grid1[nl]))
        fake1_gray = functional.grid_sample((image_unstable1[:, period // 2: period // 2 + 1:, :] + 1) * 127.5,grid1[opt.num_layer - 1])
        fake1_gray = fake1_gray / 127.5 - 1

        grid2 = netG(image_unstable2[:, 0:period + 1, :, :])

        for nl in range(opt.num_layer):
            grid2[nl] = generate_maps(grid2[nl]).permute(0, 2, 3, 1)
            fake2_temp = functional.grid_sample((image_unstable2[:, period + 1:period + 1 + 3, :, :] + 1) * 127.5,grid2[nl])
            fake2.append(fake2_temp / 127.5 - 1)
        fake2_gray = functional.grid_sample((image_unstable2[:, period // 2: period // 2 + 1:, :] + 1) * 127.5,grid2[opt.num_layer - 1])
        fake2_gray = fake2_gray / 127.5 - 1


        ############################
        # (1) Update D network: maximize log(D(x,y)) + log(1 - D(x,G(x)))
        ###########################
        if opt.use_gan and with_gan:

            optimizerD.zero_grad()

            # train with fake
            fake_ab1=fake1_gray

            fake_ab1 = frame_clip_batchsize(fake_ab1, affine1).permute(1,0,2,3)
            pred_fake1 = netD.forward(fake_ab1.detach())
            loss_d_fake1 = criterionGAN(pred_fake1, False)

            fake_ab2 = fake2_gray#torch.cat((image_stable2[:,3:3+opt.period_D,:,:], fake2_gray, image_stable2[:,3+opt.period_D+1:3+opt.period_D+1+opt.period_D,:,:]), 1)
            fake_ab2 = frame_clip_batchsize(fake_ab2, affine2).permute(1,0,2,3)
            pred_fake2 = netD.forward(fake_ab2.detach())
            loss_d_fake2 = criterionGAN(pred_fake2, False)


            # train with real
            real_ab1 = image_stable1[:,3+opt.period_D:3+opt.period_D+1,:,:]#image_stable1[:,3:3+opt.period_D*2+1,:,:]
            real_ab1 = frame_clip_batchsize(real_ab1, affine1).permute(1,0,2,3)
            pred_real1 = netD.forward(real_ab1)
            loss_d_real1 = criterionGAN(pred_real1, True)

            real_ab2 = image_stable2[:,3+opt.period_D:3+opt.period_D+1,:,:]#image_stable2[:,3:3+opt.period_D*2+1,:,:]
            real_ab2= frame_clip_batchsize(real_ab2, affine2).permute(1,0,2,3)
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
            fake_ab1 = fake1_gray#torch.cat((image_stable1[:,3:3+opt.period_D,:,:], fake1_gray, image_stable1[:,3+opt.period_D+1:3+opt.period_D+1+opt.period_D,:,:]), 1)
            fake_ab1 = frame_clip_batchsize(fake_ab1, affine1).permute(1,0,2,3)
            pred_fake1 = netD(fake_ab1)
            loss_d_fake1_g = criterionGAN(pred_fake1, True)*opt.balance_gd

            fake_ab2 = fake2_gray#torch.cat((image_stable2[:,3:3+opt.period_D,:,:], fake2_gray, image_stable2[:,3+opt.period_D+1:3+opt.period_D+1+opt.period_D,:,:]), 1)
            fake_ab2 = frame_clip_batchsize(fake_ab2, affine2).permute(1,0,2,3)
            pred_fake2 = netD(fake_ab2)
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

            # feature_adjacent = feature_adjacent.view(-1, 2, 3)
            # grid = functional.affine_grid(feature_adjacent, fake1[nl].size())
            #
            # output2_to_output1 = functional.grid_sample(fake2[nl], grid)
            # loss_g2 += torch.mean(torch.abs(output2_to_output1 - fake1[nl]))

            loss_g2+=torch.mean(torch.abs(fake2[nl] - fake1[nl]))

            if opt.shapeloss:
               loss_pixel = loss_pixel1(grid1[nl], grid_eye, A_affine, opt.block, opt.block, opt.input_size) * opt.shapeloss_weight + loss_pixel1(grid2[nl], grid_eye, A_affine, opt.block, opt.block, opt.input_size) * opt.shapeloss_weight

        if opt.shapeloss:
            loss_g1 = loss_feature+loss_vgg+ loss_mse + loss_pixel #+loss_pixel  # + loss_affine * 20  # (delta1+delta2)*10+loss_affine*1000
        else:
            loss_g1 = loss_feature + loss_vgg + loss_mse
        if opt.use_gan and with_gan:
            loss_g = loss_g1 + loss_g2 * opt.lamd +(loss_d_fake1_g+loss_d_fake2_g)/2
        else:
            loss_g = loss_g1 +loss_g2 * opt.lamd

        loss_g.backward()

        optimizerG.step()
        if (i % 10 == 0):
            viz.line(Y=np.array([loss_feature.data.cpu().numpy()]), X=np.array([i + epoch * batch_idxs]), win='train_loss',update='append', name='loss_feature')
            viz.line(Y=np.array([loss_g2.data.cpu().numpy()]), X=np.array([i + epoch * batch_idxs]), win='train_loss',update='append', name='loss_g2')
            viz.line(Y=np.array([loss_vgg.data.cpu().numpy()]), X=np.array([i + epoch * batch_idxs]), win='train_loss',update='append', name='loss_vgg')
            viz.line(Y=np.array([loss_mse.data.cpu().numpy()]), X=np.array([i + epoch * batch_idxs]), win='train_loss',update='append', name='loss_mse')
            if opt.shapeloss:
                viz.line(Y=np.array([loss_pixel.data.cpu().numpy()]), X=np.array([i + epoch * batch_idxs]), win='train_loss',update='append', name='loss_pixel')
        if (i % 10 == 0):
            viz.images(vutils.make_grid(image_stable2[0:3, 0:3, :, :], normalize=True, scale_each=True), win='stable', opts=dict(title='stable', caption=str(i + epoch * batch_idxs)))
            for nl in range(opt.num_layer):
                viz.images(vutils.make_grid(fake2[nl][0:3, 0:3, :, :], normalize=True, scale_each=True), win='fake' + str(nl), opts=dict(title='fake' + str(nl), caption=str(i + epoch * batch_idxs)))
            viz.images(vutils.make_grid(image_unstable2[0:3, period + 1:period + 1 + 3, :, :], normalize=True, scale_each=True), win='unstable', opts=dict(title='unstable', caption=str(i + epoch * batch_idxs)))

        time_end = time.time()
        time_each=time_end-time_begin
        time_left=((opt.nEpochs-epoch-1)*batch_idxs+batch_idxs-i)*time_each/3600
        print("=>train--Epoch[{}]({}/{}): time(s): {:.4f} Time_left(h): {:.4f}  Learning rate: {:.6f}".format(
            epoch, i, batch_idxs, time_each, time_left, lr))

def test(epoch, lr, list_stable, list_unstable, list_feature, list_adjacent, list_affine, list_epoch):
    if epoch > opt.start_gan:
        with_gan = True
    else:
        with_gan = False
    netG.eval()
    if opt.continue_train > 0 and epoch == opt.continue_train:
        net_g_model_out_path = "checkpoint/{}/netG_model_epoch_{}.pth".format(opt.checkpoint_dir, opt.continue_train)
        checkpoint = torch.load(net_g_model_out_path)
        netG.load_state_dict(checkpoint['net'])

    # optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(opt.beta1, 0.999))

    if opt.use_gan and with_gan:
        # optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(opt.beta1, 0.999))
        list_epoch = list_shuffle(list_epoch)
    else:
        random.shuffle(list_epoch)
    dataset = customData(val_files, list_epoch, list_stable, list_unstable, list_feature, list_adjacent, list_affine,
                         with_gan)
    sample = torch.utils.data.DataLoader(dataset, batch_size=opt.batchSize, shuffle=False, num_workers=opt.threads,
                                         pin_memory=True)
    iter_sample = iter(sample)

    batch_idxs = (int(dataset.__len__())) // opt.batchSize

    if opt.shapeloss:
        A_affine = torch.from_numpy(generate_affine_matrix(opt.block, opt.block)).cuda().float().unsqueeze(0).repeat(opt.batchSize, 1, 1, 1, 1).to(torch.float64)
        eye = torch.tensor([[1, 0, 0], [0, 1, 0]], dtype=torch.float).unsqueeze(0).expand(opt.batchSize, 2, 3)
        grid_eye = functional.affine_grid(eye, torch.Size((opt.batchSize, 3, opt.input_size, opt.input_size))).cuda()

    for i in range(batch_idxs):
        time_begin = time.time()
        images1, features1, affine1, images2, features2, affine2, feature_adjacent = next(iter_sample)

        images1 = images1.cuda()
        images2 = images2.cuda()
        features1 = features1.cuda().float()
        features2 = features2.cuda().float()
        affine1 = affine1.cuda().float()
        affine2 = affine2.cuda().float()

        feature_adjacent = feature_adjacent.cuda().float()

        image_stable1, image_unstable1, feature_stable1, feature_unstable1 = pre_propossing(images1, features1)

        image_stable2, image_unstable2, feature_stable2, feature_unstable2 = pre_propossing(images2, features2)

        fake1 = []
        fake2 = []

        grid1 = netG(image_unstable1[:, 0:period + 1, :, :])

        for nl in range(opt.num_layer):
            grid1[nl] = generate_maps(grid1[nl]).permute(0, 2, 3, 1)
            fake1_temp = functional.grid_sample((image_unstable1[:, period + 1:period + 1 + 3, :, :] + 1) * 127.5,
                                                grid1[nl])
            fake1.append(fake1_temp / 127.5 - 1)
            # fake1.append(functional.grid_sample(image_unstable1[:, period + 1:period + 1 + 3, :, :], grid1[nl]))
        fake1_gray = functional.grid_sample((image_unstable1[:, period // 2: period // 2 + 1:, :] + 1) * 127.5,
                                            grid1[opt.num_layer - 1])
        fake1_gray = fake1_gray / 127.5 - 1

        grid2 = netG(image_unstable2[:, 0:period + 1, :, :])

        for nl in range(opt.num_layer):
            grid2[nl] = generate_maps(grid2[nl]).permute(0, 2, 3, 1)
            fake2_temp = functional.grid_sample((image_unstable2[:, period + 1:period + 1 + 3, :, :] + 1) * 127.5,
                                                grid2[nl])
            fake2.append(fake2_temp / 127.5 - 1)
        fake2_gray = functional.grid_sample((image_unstable2[:, period // 2: period // 2 + 1:, :] + 1) * 127.5,
                                            grid2[opt.num_layer - 1])
        fake2_gray = fake2_gray / 127.5 - 1

        ############################
        # (1) Update D network: maximize log(D(x,y)) + log(1 - D(x,G(x)))
        ###########################
        if opt.use_gan and with_gan:
            # optimizerD.zero_grad()

            # train with fake
            fake_ab1 = fake1_gray

            fake_ab1 = frame_clip_batchsize(fake_ab1, affine1).permute(1, 0, 2, 3)
            pred_fake1 = netD.forward(fake_ab1.detach())
            loss_d_fake1 = criterionGAN(pred_fake1, False)

            fake_ab2 = fake2_gray  # torch.cat((image_stable2[:,3:3+opt.period_D,:,:], fake2_gray, image_stable2[:,3+opt.period_D+1:3+opt.period_D+1+opt.period_D,:,:]), 1)
            fake_ab2 = frame_clip_batchsize(fake_ab2, affine2).permute(1, 0, 2, 3)
            pred_fake2 = netD.forward(fake_ab2.detach())
            loss_d_fake2 = criterionGAN(pred_fake2, False)

            # train with real
            real_ab1 = image_stable1[:, 3 + opt.period_D:3 + opt.period_D + 1, :,
                       :]  # image_stable1[:,3:3+opt.period_D*2+1,:,:]
            real_ab1 = frame_clip_batchsize(real_ab1, affine1).permute(1, 0, 2, 3)
            pred_real1 = netD.forward(real_ab1)
            loss_d_real1 = criterionGAN(pred_real1, True)

            real_ab2 = image_stable2[:, 3 + opt.period_D:3 + opt.period_D + 1, :,
                       :]  # image_stable2[:,3:3+opt.period_D*2+1,:,:]
            real_ab2 = frame_clip_batchsize(real_ab2, affine2).permute(1, 0, 2, 3)
            pred_real2 = netD.forward(real_ab2)
            loss_d_real2 = criterionGAN(pred_real2, True)

            # Combined loss
            loss_d = (loss_d_fake1 + loss_d_fake2 + loss_d_real1 + loss_d_real2) * 0.5

            # loss_d.backward()
            #
            # optimizerD.step()

        ############################
        # (2) Update G network: maximize log(D(x,G(x))) + L1(y,G(x))
        ##########################
        # optimizerG.zero_grad()
        if opt.use_gan and with_gan:
            fake_ab1 = fake1_gray  # torch.cat((image_stable1[:,3:3+opt.period_D,:,:], fake1_gray, image_stable1[:,3+opt.period_D+1:3+opt.period_D+1+opt.period_D,:,:]), 1)
            fake_ab1 = frame_clip_batchsize(fake_ab1, affine1).permute(1, 0, 2, 3)
            pred_fake1 = netD(fake_ab1)
            loss_d_fake1_g = criterionGAN(pred_fake1, True) * opt.balance_gd

            fake_ab2 = fake2_gray  # torch.cat((image_stable2[:,3:3+opt.period_D,:,:], fake2_gray, image_stable2[:,3+opt.period_D+1:3+opt.period_D+1+opt.period_D,:,:]), 1)
            fake_ab2 = frame_clip_batchsize(fake_ab2, affine2).permute(1, 0, 2, 3)
            pred_fake2 = netD(fake_ab2)
            loss_d_fake2_g = criterionGAN(pred_fake2, True) * opt.balance_gd

        loss_mse = 0
        loss_feature = 0
        loss_delta = 0
        loss_vgg = 0
        loss_g2 = 0
        loss_affine = 0
        for nl in range(opt.num_layer):
            loss_mse1, loss_delta1, loss_feature1 = loss_calulate(grid1[nl], feature_stable1, feature_unstable1,
                                                                  fake1[nl], image_stable1, opt.batchSize)

            loss_mse2, loss_delta2, loss_feature2 = loss_calulate(grid2[nl], feature_stable2, feature_unstable2,
                                                                  fake2[nl], image_stable2, opt.batchSize)
            loss_mse += loss_mse1.item() + loss_mse2.item()
            loss_feature += loss_feature1.item() + loss_feature2.item()
            loss_delta += loss_delta1.item() + loss_delta2.item()
            loss_vgg += generator_criterion(fake1[nl], image_stable1[:, 0:3, :, :]).item()
            loss_vgg += generator_criterion(fake2[nl], image_stable2[:, 0:3, :, :]).item()

            # feature_adjacent = feature_adjacent.view(-1, 2, 3)
            # grid = functional.affine_grid(feature_adjacent, fake1[nl].size())
            #
            # output2_to_output1 = functional.grid_sample(fake2[nl], grid)
            # loss_g2 += torch.mean(torch.abs(output2_to_output1 - fake1[nl]))

            loss_g2 += torch.mean(torch.abs(fake2[nl] - fake1[nl])).item()

            if opt.shapeloss:
                loss_pixel = loss_pixel1(grid1[nl], grid_eye, A_affine, opt.block, opt.block,opt.input_size).item() * opt.shapeloss_weight + loss_pixel1(grid2[nl], grid_eye,A_affine, opt.block,opt.block,opt.input_size).item() * opt.shapeloss_weight

        if opt.shapeloss:
            loss_g1 = loss_feature + loss_vgg+ loss_mse+ loss_pixel  # +loss_pixel  # + loss_affine * 20  # (delta1+delta2)*10+loss_affine*1000
        else:
            loss_g1 = loss_feature+ loss_vgg + loss_mse
        if opt.use_gan and with_gan:
            loss_g = loss_g1 + loss_g2* opt.lamd + (loss_d_fake1_g + loss_d_fake2_g) / 2
        else:
            loss_g = loss_g1 + loss_g2 * opt.lamd

        # loss_g.backward()
        #
        # optimizerG.step()
        if (i % 10 == 0):
            viz.line(Y=np.array([loss_feature]), X=np.array([i + epoch * batch_idxs]), win='test_loss',
                     update='append', name='loss_feature')
            viz.line(Y=np.array([loss_g2]), X=np.array([i + epoch * batch_idxs]), win='test_loss',
                     update='append', name='loss_g2')
            viz.line(Y=np.array([loss_vgg]), X=np.array([i + epoch * batch_idxs]), win='test_loss',
                     update='append', name='loss_vgg')
            viz.line(Y=np.array([loss_mse]), X=np.array([i + epoch * batch_idxs]), win='test_loss',
                     update='append', name='loss_mse')
            if opt.shapeloss:
                viz.line(Y=np.array([loss_pixel]), X=np.array([i + epoch * batch_idxs]), win='test_loss',
                         update='append', name='loss_pixel')
        if (i % 10 == 0):
            viz.images(vutils.make_grid(image_stable2[0:3, 0:3, :, :], normalize=True, scale_each=True), win='stable',
                       opts=dict(title='stable', caption=str(i + epoch * batch_idxs)))
            for nl in range(opt.num_layer):
                viz.images(vutils.make_grid(fake2[nl][0:3, 0:3, :, :], normalize=True, scale_each=True),
                           win='fake' + str(nl), opts=dict(title='fake' + str(nl), caption=str(i + epoch * batch_idxs)))
            viz.images(vutils.make_grid(image_unstable2[0:3, period + 1:period + 1 + 3, :, :], normalize=True,
                                        scale_each=True), win='unstable',
                       opts=dict(title='unstable', caption=str(i + epoch * batch_idxs)))

        time_end = time.time()
        time_each = time_end - time_begin
        time_left = ((opt.nEpochs - epoch - 1) * batch_idxs + batch_idxs - i) * time_each / 3600
        print("=>val--Epoch[{}]({}/{}): time(s): {:.4f} Time_left(h): {:.4f}  Learning rate: {:.6f}".format(
            epoch, i, batch_idxs, time_each, time_left, lr))


def checkpoint(epoch):
    if not os.path.exists("checkpoint"):
        os.mkdir("checkpoint")
    if not os.path.exists(os.path.join("checkpoint", opt.checkpoint_dir)):
        os.mkdir(os.path.join("checkpoint", opt.checkpoint_dir))
    state = {'net':netG.state_dict(), 'epoch':epoch}

    net_g_model_out_path = "checkpoint/{}/netG_model_epoch_{}.pth".format(opt.checkpoint_dir, epoch)

    torch.save(state, net_g_model_out_path)
    print("Checkpoint saved to {}".format("checkpoint" + opt.checkpoint_dir))


def process():
    """Test pix2pix"""
    # record all videos

    class_name = ['lowquality','Regular', 'QuickRotation', 'Running', 'Parallax', 'Crowd', 'Zooming']
    crop_ratio = 0.2
    num_sample = 20
    stride_sample = 15
    threshold = 0
    rate = 0.5

    for ii in range(1,len(class_name)):

        path = '/media/ssd1/zmd/dataset/normal/' + class_name[ii] + '/'
        # path = '/data/zmd/DeepStab/DeepStab/unstable_test/'

        list_videos = os.listdir(path)

        # index_sample_test = np.arange(30, 0, -1)
        index_sample_test = np.append(np.arange(-period // 2, 0, 1), np.arange(0, period // 2 + 1, 1))

        # index_sample_test = [32, 16, 8, 4, 2, 1]#[5, 10, 15, 20, 25, 30]#[1, 2, 4, 8, 16, 32]
        net_g_model_out_path = "checkpoint/{}/netG_model_epoch_{}.pth".format(opt.checkpoint_dir, opt.continue_train)
        checkpoint = torch.load(net_g_model_out_path)
        netG.load_state_dict(checkpoint['net'])

        if not os.path.exists("test_result"):
            os.mkdir("test_result")
        if not os.path.exists(os.path.join("test_result", class_name[ii])):
            os.mkdir(os.path.join("test_result", class_name[ii]))

        #netG = torch.load('./checkpoint/unet_512_kalman_10_period_30-1515/netG_model_epoch_35.pth')
        netG.eval()

        for video_id in range(0,len(list_videos)):

            ###calculate cropping
            '''
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

                grid_whole = netG(images)
                grid = grid_whole[2]

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

                pts = np.float32([[-1, -1], [-1, 0.992], [0.992, 0.992], [0.992, -1]]).reshape(-1, 1, 2)

                dst = cv2.perspectiveTransform(pts, M)
                if (i == 0):
                    y_start = 0
                    x_start = 0
                    y_end = size_origin[1]
                    x_end = size_origin[0]
                else:
                    if (i > period // 2 and i < num_frame - period // 2):
                        y_start = (y_start + (max((dst[0, 0, 1] + 1) * size_origin[1] / 2,(dst[3, 0, 1] + 1) * size_origin[1] / 2) - y_start) * rate if y_start < max((dst[0, 0, 1] + 1) * size_origin[1] / 2,(dst[3, 0, 1] + 1) * size_origin[1] / 2) else y_start)
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
            videoWriter = cv2.VideoWriter('./test_result/' + class_name[ii] + '/' + str(list_videos[video_id])[0:-4] + '.avi',
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


                list_unstable = []
                # list_stable=[]

                time_sum1 = time.time()
                if i == 0:
                    for j in range(len(index_sample_test)):
                        list_unstable.append(cv2.resize(cv2.cvtColor(history_frame[period // 2 + index_sample_test[j]], cv2.COLOR_BGR2GRAY),size,interpolation=cv2.INTER_AREA))
                    npp = np.array(list_unstable)
                    images = npp[np.newaxis, :, :, :]

                    images = torch.from_numpy(images)
                    images = images.cuda().float()
                    images = images.float() / 255 * 2 - 1

                else:
                    if (i >= int(num_frame) - period // 2):
                        img_unstable = history_frame[-1]
                        img_unstable_resize = cv2.resize(cv2.cvtColor(img_unstable, cv2.COLOR_BGR2GRAY), size,interpolation=cv2.INTER_AREA)
                        img_unstable_tensor = torch.from_numpy(img_unstable_resize[np.newaxis, np.newaxis, :, :]).cuda().float() / 255 * 2 - 1
                        images = torch.cat((images[:, 1:, :, :], img_unstable_tensor), 1)
                        history_frame.pop(0)

                        history_frame.append(img_unstable)


                    else:
                        ret, img_unstable = cap.read()
                        if ret == False:
                            continue
                        img_unstable_resize = cv2.resize(cv2.cvtColor(img_unstable, cv2.COLOR_BGR2GRAY), size,
                                                         interpolation=cv2.INTER_AREA)
                        img_unstable_tensor = torch.from_numpy(img_unstable_resize[np.newaxis, np.newaxis, :, :]).cuda().float() / 255 * 2 - 1
                        images = torch.cat((images[:, 1:, :, :], img_unstable_tensor), 1)
                        history_frame.pop(0)

                        history_frame.append(img_unstable)

                    # img_unstable = cv2.resize(img_unstable, size, interpolation=cv2.INTER_AREA)
                # img_unstable_resize=cv2.resize(img_unstable,size,interpolation=cv2.INTER_AREA)
                # img_unstable_resize = cv2.cvtColor(img_unstable_resize, cv2.COLOR_BGR2RGB)
                # list_unstable.append(img_unstable_resize)
                now = cv2.cvtColor(history_frame[period // 2], cv2.COLOR_BGR2RGB)
                now = now[np.newaxis, :, :, :]
                now = torch.from_numpy(now)
                now = now.cuda().float()
                now = now.float()
                now = now.permute(0, 3, 1, 2)
                # np1 = np.array(list_stable)
                # np2 = np.array(list_unstable).transpose(0, 3, 1, 2).squeeze()
                # npp = np.concatenate((np1, np2), axis=0)

                # npp = np.array(list_unstable)
                # images = npp[np.newaxis, :, :, :]

                # images = torch.from_numpy(images)
                # images = images.cuda().float()
                # images = images.float() / 255 * 2 - 1
                torch.cuda.synchronize()
                time1 = time.time()
                grid = netG(images,False)
                torch.cuda.synchronize()
                time2 = time.time()

                # grid = generate_maps(grid, 1)
                # grid_affine = F.affine_grid(grid[1], images.size())
                # print(grid[1])
                # grid_drift=grid[0]
                # grid_sum=grid_affine.permute(0,3,1,2)+grid_drift
                grid = generate_maps(grid,1)

                m = torch.nn.UpsamplingBilinear2d(size=(size_origin[1], size_origin[0]))
                grid_resize = m(grid)
                grid_resize = grid_resize.permute(0, 2, 3, 1)

                # update history_frame
                # fake_gray=cv2.cvtColor(fake[0,:,:,:].data.cpu().numpy(),cv2.COLOR_BGR2GRAY)
                # history_frame.pop(0)
                # history_frame.append(fake_gray)
                fake = functional.grid_sample(now, grid_resize)
                samples = fake[0, :, :, :].data.cpu().numpy()
                # samples = (samples + 1) * 127.5
                samples = samples.transpose((1, 2, 0))
                # samples = samples.reshape((144, 256,-1))
                samples = np.array(samples.astype(np.uint8))
                # update history_frame
                # fake_gray=cv2.cvtColor(samples,cv2.COLOR_BGR2GRAY)

                samples = cv2.resize(samples, (640, 360), interpolation=cv2.INTER_AREA)

                samples = cv2.cvtColor(samples, cv2.COLOR_BGR2RGB)
                # samples_resize = samples[int(cap.get(4) * crop_ratio):int(cap.get(4) * crop_ratio ) + int(size_crop[1]),int(cap.get(3) * crop_ratio / 2):int(cap.get(3) * crop_ratio / 2) + int(size_crop[0]), :]
                samples_resize = samples[int(y_start) + threshold:int(y_end) - threshold,
                                 int(x_start) + threshold:int(x_end) - threshold, :]
                samples_resize = cv2.GaussianBlur(samples_resize, (3, 3), 0.2, 0)
                unstable_resize = cv2.resize(history_frame[period // 2], (640, 360), interpolation=cv2.INTER_AREA)
                # cv2.imshow('show_unstable.jpg', unstable_resize)
                # cv2.imshow('show_stable.jpg', samples_resize)
                # # cv2.imshow('show_crop.jpg',samples_resize)
                # cv2.waitKey(1)
                videoWriter.write(samples_resize)  # 写视频帧
                time_sum2 = time.time()

                print('class_id:'+ str(ii)+' class_name:'+class_name[ii]+ ' video_id:'+ str(video_id)+' video_name:'+ list_videos[video_id] + ' frame_index:' +
                str(i) + ' cropping :' + str(x_start) + ' ' + str(x_end) + ' ' + str(y_start) + ' ' + str(y_end)+ ' FPS: '+str(int(1/(time2 - time1))))

            videoWriter.release()


def main():
    if opt.mode=='train':
        list_stable,list_unstable,list_feature,list_adjacent,list_affine=image_store(train_files)
        list_stable_val,list_unstable_val,list_feature_val,list_adjacent_val,list_affine_val=image_store(val_files)


        for epoch in range(opt.continue_train, opt.nEpochs):
                if epoch> opt.start_gan:
                    list_epoch = list_random_batchsize(train_files)
                    list_epoch_val = list_random_batchsize(val_files)
                else:
                    list_epoch = list_random(train_files)
                    list_epoch_val = list_random(val_files)


                lr = opt.lr * 0.1 ** int((epoch) / 20)

                if epoch % 5 == 0:
                    test(epoch, lr,list_stable_val, list_unstable_val, list_feature_val, list_adjacent_val, list_affine_val,list_epoch_val)

                train(epoch, lr, list_stable, list_unstable, list_feature, list_adjacent, list_affine, list_epoch)
                if epoch%5==0:
                    checkpoint(epoch)


    else:
        process()


if __name__ == '__main__':
    main()


'''
process()
'''
