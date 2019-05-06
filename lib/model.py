import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
import functools
from lib.cfg_en_de_coder import opt

class UnetGenerator(nn.Module):
    def __init__(self):
        super(UnetGenerator, self).__init__()

        self.localization0 = nn.Sequential(
            nn.Conv2d(8, 16, kernel_size=7,stride=3,padding=3),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(16, 16, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(16, 16, kernel_size=5,stride=3,padding=2),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(16, 16, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(16, 16, kernel_size=3,stride=2,padding=1),
            nn.LeakyReLU(0.2, True),
        )
        self.fc_loc0 = nn.Sequential(
            nn.Linear(16 * 4 * 4, 3 * 2,bias=False)
        )

        self.fc_loc1 = nn.Sequential(
            nn.Linear(16 * 4 * 4, 3 * 2, bias=False)
        )

        self.fc_loc2 = nn.Sequential(
            nn.Linear(16 * 4 * 4, 3 * 2, bias=False)
        )

        self.fc_loc3 = nn.Sequential(
            nn.Linear(16 * 4 * 4, 3 * 2, bias=False)
        )

        self.fc_loc4 = nn.Sequential(
            nn.Linear(16 * 4 * 4, 3 * 2, bias=False)
        )

        self.bias=torch.Tensor([[1,0,0],[0,1,0]]).expand(opt.batchSize,2,3).cuda()
        self.localization1 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=5, stride=3, padding=2),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(64, 32, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(32, 16, kernel_size=5, stride=3, padding=2),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(16, 16, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.2, True),
        )

        self.localization2 = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=7, stride=4, padding=2),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(128, 64, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.2,True),
            nn.Conv2d(64, 16, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.2, True),
        )

        self.localization3 = nn.Sequential(
            nn.Conv2d(512,128, kernel_size=5, stride=3, padding=2),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(128, 16, kernel_size=5, stride=3, padding=2),
            nn.LeakyReLU(0.2, True),
        )
        self.localization4 = nn.Sequential(
            nn.Conv2d(1024, 128, kernel_size=5, stride=3, padding=2),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(128, 16, kernel_size=3, stride=2, padding=2),
            nn.LeakyReLU(0.2, True),
        )
        self.down=self.downconv(512,512)

        self.up1=self.upconv(512,512)
        self.up2=self.upconv(512,512)
        self.up3 = self.upconv(512, 512)
        self.up4 = self.upconv(512, 256)
        self.up5 = self.upconv(256, 128)
        self.up6 = self.upconv(128, 64)
        self.up7 = self.upconv(64, 64)
        self.final=self.downconv(64,3,down=False)


        self.conv11=nn.Conv2d(3,64,kernel_size=7,stride=2,padding=3)
        self.conv12 = nn.Conv2d(5, 64, kernel_size=7, stride=2, padding=3)
        self.conv2 = nn.Conv2d(64,128, kernel_size=5, stride=2, padding=2)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=5, stride=2, padding=2)
        self.conv4 = nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1)
        self.sameconv1=nn.Conv2d(1024, 512, kernel_size=3, stride=1, padding=1)
        self.sameconv2 = nn.Conv2d(1024, 512, kernel_size=3, stride=1, padding=1)
        self.sameconv3 = nn.Conv2d(1024, 512, kernel_size=3, stride=1, padding=1)
        self.sameconv4 = nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1)
        self.sameconv5 = nn.Conv2d(256,128 , kernel_size=3, stride=1, padding=1)
        self.sameconv6 = nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1)




        self.relu = nn.LeakyReLU(0.2, True)
    def stn0(self,input):
        x=self.localization0(input)
        x = x.view(-1, 16 * 4 * 4)
        theta = self.fc_loc0(x)

        theta = theta.view(-1, 2, 3)
        theta = theta + self.bias
        grid = F.affine_grid(theta, input.size())
        output = F.grid_sample(input[:,5:8,:,:], grid)
        return output
    def stn1(self,input1,input2,ngf=64):

        x1=(self.conv11(input1))
        x2=(self.conv12(input2))

        x1 = self.relu(x1)
        x2 = self.relu(x2)
        x=self.localization1(torch.cat([x1,x2],dim=1))
        x = x.view(-1, 16 * 4 * 4)
        theta = self.fc_loc1(x)

        theta = theta.view(-1, 2, 3)
        theta = theta + self.bias

        grid = F.affine_grid(theta, x1.size())
        output = F.grid_sample(x1, grid)
        return output,x2,theta

    def stn2(self,input1,input2,ngf=128):

        x1=(self.conv2(input1))
        x2=(self.conv2(input2))

        x1 = self.relu(x1)
        x2 = self.relu(x2)
        x=self.localization2(torch.cat([x1,x2],dim=1))
        x = x.view(-1, 16 * 4 * 4)
        theta = self.fc_loc2(x)

        theta = theta.view(-1, 2, 3)
        theta = theta + self.bias

        grid = F.affine_grid(theta, x1.size())
        output = F.grid_sample(x1, grid)
        return output,x2,theta

    def stn3 (self,input1,input2,ngf=256):

        x1=(self.conv3(input1))
        x2=(self.conv3(input2))

        x1=self.relu(x1)
        x2 = self.relu(x2)
        x=self.localization3(torch.cat([x1,x2],dim=1))
        x = x.view(-1, 16 * 4 * 4)
        theta = self.fc_loc3(x)
        theta = theta.view(-1, 2, 3)
        theta = theta + self.bias
        grid = F.affine_grid(theta, x1.size())
        output = F.grid_sample(x1, grid)
        return output,x2,theta

    def stn4 (self,input1,input2,ngf=512):

        x1=(self.conv4(input1))
        x2=(self.conv4(input2))

        x1 = self.relu(x1)
        x2 = self.relu(x2)
        x=self.localization4(torch.cat([x1,x2],dim=1))
        x = x.view(-1, 16 * 4 * 4)
        theta = self.fc_loc4(x)
        theta = theta.view(-1, 2, 3)
        theta = theta + self.bias
        grid = F.affine_grid(theta, x1.size())
        output = F.grid_sample(x1, grid)
        return output,x2,theta


    def downconv(self,inc,outc,down=True):
        if down:
            conv = nn.Conv2d(inc,outc, kernel_size=5, stride=2, padding=2)
        else:
            conv = nn.Conv2d(inc, outc, kernel_size=3, stride=1, padding=1)
        return conv
    def finalconv(self,inc,outc):
        conv = nn.Conv2d(inc,outc, kernel_size=3, stride=1, padding=1)
        return conv

    def upconv(self,inc,outc):
        conv = nn.ConvTranspose2d(inc, outc,kernel_size=4, stride=2,padding=1)
        return conv
    # def sameconv(self,input,reduce=True):
    #     if reduce:
    #         conv = nn.Conv2d(input.shape[1], input.shape[1]//2, kernel_size=3, stride=1, padding=1).cuda()
    #     else:
    #         conv = nn.Conv2d(input.shape[1], 3 , kernel_size=3, stride=1, padding=1).cuda()
    #     return conv(input)

    def transform(self,input,theta,layer):

        for i in range(layer):
            grid = F.affine_grid(theta[layer-i-1], input.size())
            input = F.grid_sample(input, grid)
        return input

    def forward(self,input):
        x0=self.stn0(input)
        x1,y1,t1=self.stn1(x0,input[:,0:5,:,:])#64x128x128
        x2,y2,t2=self.stn2(x1,y1)#128x64x64
        x3, y3,t3 = self.stn3(x2, y2)#256x32x32
        x4, y4,t4 = self.stn4(x3, y3)#512x16x16
        x5=nn.LeakyReLU(0.2, True)(self.down(x4))#512x8x8
        x6=nn.LeakyReLU(0.2, True)(self.down(x5)) #512x4x4
        x7=nn.LeakyReLU(0.2, True)(self.down(x6))#2x2
        T=[t4,t3,t2,t1]

        x8=nn.ReLU(True)(self.sameconv1(torch.cat([nn.ReLU(True)(self.up2(x7)), x6], dim=1)))#512x4x4
        x9 = nn.ReLU(True)(self.sameconv2(torch.cat([nn.ReLU(True)(self.up2(x8)), x5], dim=1)))  # 512x8x8
        x10 = nn.ReLU(True)(self.sameconv3(torch.cat([nn.ReLU(True)(self.up3(x9)), self.transform(x4,T,1)], dim=1)))  # 512x16x16
        x11 = nn.ReLU(True)(self.sameconv4(torch.cat([nn.ReLU(True)(self.up4(x10)), self.transform(x3,T,2)], dim=1))) # 256x32x32
        x12 = nn.ReLU(True)(self.sameconv5(torch.cat([nn.ReLU(True)(self.up5(x11)), self.transform(x2,T,3)], dim=1)))  # 128x64x64
        x13 = nn.ReLU(True)(self.sameconv6(torch.cat([nn.ReLU(True)(self.up6(x12)), self.transform(x1,T,4)], dim=1)))  # 64x128x128
        x14 = self.up7(x13)  # 64x256x256
        x_final=nn.Tanh()(self.final(x14))#3x256x256
        return x_final


def get_norm_layer(norm_type='instance'):
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=False)
    elif norm_type == 'none':
        norm_layer = None
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer


def init_weights(net, init_type='normal', gain=0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:
            init.normal_(m.weight.data, 1.0, gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)


def init_net(net, init_type='normal', init_gain=0.02, gpu_ids=[],parallel=False):

    if parallel:
        net = torch.nn.DataParallel(net)
    else:
        if len(gpu_ids) > 0:
            assert (torch.cuda.is_available())
            net.to(gpu_ids[0])
            net = torch.nn.DataParallel(net, gpu_ids)
    init_weights(net, init_type, gain=init_gain)
    return net


def define_G(input_nc, output_nc, ngf, init_type='normal', init_gain=0.02, gpu_ids=[]):

    net = UnetGenerator()
    return init_net(net, init_type, init_gain, gpu_ids,parallel=True)


def define_D(input_nc, ndf, netD,
             n_layers_D=3, norm='batch', use_sigmoid=False, init_type='normal', init_gain=0.02, gpu_ids=[]):
    net = None
    norm_layer = get_norm_layer(norm_type=norm)

    if netD == 'basic':
        net = NLayerDiscriminator(input_nc, ndf, n_layers=3, norm_layer=norm_layer, use_sigmoid=use_sigmoid)
    elif netD == 'n_layers':
        net = NLayerDiscriminator(input_nc, ndf, n_layers_D, norm_layer=norm_layer, use_sigmoid=use_sigmoid)
    elif netD == 'pixel':
        net = PixelDiscriminator(input_nc, ndf, norm_layer=norm_layer, use_sigmoid=use_sigmoid)
    else:
        raise NotImplementedError('Discriminator model name [%s] is not recognized' % net)
    return init_net(net, init_type, init_gain, gpu_ids)

# Defines the GAN loss which uses either LSGAN or the regular GAN.
# When LSGAN is used, it is basically same as MSELoss,
# but it abstracts away the need to create the target label tensor
# that has the same size as the input
class GANLoss(nn.Module):
    def __init__(self, use_lsgan=True, target_real_label=1.0, target_fake_label=-1.0):
        super(GANLoss, self).__init__()
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))
        if use_lsgan:
            self.loss = nn.MSELoss()
        else:
            self.loss = nn.BCELoss()

    def get_target_tensor(self, input, target_is_real):
        if target_is_real:
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label
        return target_tensor.expand_as(input)

    def __call__(self, input, target_is_real):
        target_tensor = self.get_target_tensor(input, target_is_real)
        return self.loss(input, target_tensor)


# Defines the Unet generator.
# |num_downs|: number of downsamplings in UNet. For example,
# if |num_downs| == 7, image of size 128x128 will become of size 1x1
# at the bottleneck

# Defines the submodule with skip connection.
# X -------------------identity---------------------- X
#   |-- downsampling -- |submodule| -- upsampling --|


class down(nn.Module):
    def __init__(self, input_nc, output_nc,kernal_size=3, stride=2, padding=1, use_bias=True,norm_layer=nn.BatchNorm2d,out=False):
        super(down, self).__init__()
        downconv = nn.Conv2d(input_nc, output_nc, kernel_size=kernal_size,
                             stride=stride, padding=padding, bias=use_bias)
        downrelu = nn.LeakyReLU(0.2, True)
        downnorm = norm_layer(output_nc)
        uprelu = nn.ReLU(True)
        if out:
            self.mpconv = nn.Sequential(downconv,downnorm,nn.Tanh())
        else:
            self.mpconv = nn.Sequential(downconv,downnorm,downrelu)
    def forward(self, x):
        x = self.mpconv(x)
        return x

class down_bottom(nn.Module):
    def __init__(self, input_nc, output_nc,kernal_size=3, stride=2, padding=1, use_bias=True,norm_layer=nn.BatchNorm2d,left_input=True):
        super(down_bottom, self).__init__()
        downconv = nn.Conv2d(input_nc*2, output_nc, kernel_size=kernal_size,
                             stride=stride, padding=padding, bias=use_bias)
        downrelu = nn.LeakyReLU(0.2, True)
        downnorm = norm_layer(output_nc)


        self.conv_same = nn.Sequential(nn.Conv2d(input_nc, input_nc, kernel_size=kernal_size, stride=1, padding=padding, bias=use_bias),norm_layer(input_nc), downrelu)
        if left_input:

            self.mpconv = nn.Sequential(downconv,downnorm,downrelu)
        else:
            self.mpconv = nn.Sequential(nn.Conv2d(input_nc, output_nc, kernel_size=kernal_size,stride=stride, padding=padding, bias=use_bias), downnorm, downrelu)

        self.left_input=left_input

    def forward(self, x,x_up):

        if self.left_input:

            #self.mpconv(torch.cat([x,self.conv_same(x_up)], dim=1))
            return self.mpconv(torch.cat([x,self.conv_same(x_up)], dim=1))
        else:
            return self.mpconv(self.conv_same(x_up))


class up(nn.Module):
    def __init__(self, input_nc, output_nc, use_bias=True, norm_layer=nn.BatchNorm2d, out=False):
        super(up, self).__init__()

        uprelu = nn.ReLU(True)
        upnorm = norm_layer(output_nc)
        upconv = nn.ConvTranspose2d(input_nc, output_nc,
                                    kernel_size=4, stride=2,
                                    padding=1, bias=use_bias)

        upp = [upconv, upnorm, uprelu]

        self.mpconv = nn.Sequential(*upp)

        self.out = out

    def forward(self, x1, x2):
        if self.out:
            return self.mpconv(x1)
        else:

            return torch.cat([self.mpconv(x1), x2], dim=1)


class up_bottom(nn.Module):
    def __init__(self, input_nc, output_nc, inner_nc, use_bias=True, norm_layer=nn.BatchNorm2d, out=False):
        super(up_bottom, self).__init__()

        uprelu = nn.ReLU(True)
        upnorm = norm_layer(output_nc)
        upconv = nn.ConvTranspose2d(inner_nc, output_nc,
                                    kernel_size=4, stride=2,
                                    padding=1, bias=use_bias)
        upp = [upconv, upnorm, uprelu]

        self.mpconv = nn.Sequential(*upp)
        self.out = out
        self.conv_same = nn.Sequential(
            nn.ConvTranspose2d(input_nc, input_nc, kernel_size=3, stride=1, padding=1, bias=use_bias),
            norm_layer(input_nc), uprelu)
        # self.conv_inner = nn.Sequential(nn.ConvTranspose2d(inner_nc, output_nc, kernel_size=4, stride=2, padding=1, bias=use_bias),norm_layer(input_nc), uprelu)

    def forward(self, x_up, x_left, x_before):
        if self.out:
            return self.mpconv(torch.cat([self.conv_same(x_up), x_left], dim=1))
        else:
            nima = torch.cat([self.conv_same(x_up), x_left], dim=1)

            return torch.cat([self.mpconv(nima), x_before], dim=1)


# Defines the PatchGAN discriminator with the specified arguments.
class NLayerDiscriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d, use_sigmoid=False):
        super(NLayerDiscriminator, self).__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        kw = 3
        padw = 1
        sequence = [
            nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=1, padding=padw),
            nn.LeakyReLU(0.2, True)
        ]

        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2**n, 8)
            sequence =sequence+ [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                          kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2**n_layers, 8)
        sequence =sequence+ [
            nn.Conv2d(ndf * nf_mult_prev, ndf * 2,
                      kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            norm_layer(ndf * 2),
            nn.LeakyReLU(0.2, True)
        ]

        sequence =sequence + [nn.Conv2d(ndf * 2, 1, kernel_size=kw, stride=1, padding=padw)]

        if use_sigmoid:
            sequence =sequence + [nn.Sigmoid()]

        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        return self.model(input)


class PixelDiscriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, norm_layer=nn.BatchNorm2d, use_sigmoid=False):
        super(PixelDiscriminator, self).__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        self.net = [
            nn.Conv2d(input_nc, ndf, kernel_size=1, stride=1, padding=0),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf, ndf * 2, kernel_size=1, stride=1, padding=0, bias=use_bias),
            norm_layer(ndf * 2),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf * 2, 1, kernel_size=1, stride=1, padding=0, bias=use_bias)]

        if use_sigmoid:
            self.net.append(nn.Sigmoid())

        self.net = nn.Sequential(*self.net)

    def forward(self, input):
        return self.net(input)