import torch
import torch.nn as nn
from torch.nn import init
import functools
from torch.optim import lr_scheduler

###############################################################################
# Helper Functions
###############################################################################

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

    net = UnetGenerator(input_nc, output_nc, ngf)
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
class UnetGenerator(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=64):
        super(UnetGenerator, self).__init__()

        self.transfer = down(input_nc=input_nc, output_nc=ngf, kernal_size=5, stride=1, padding=2)
        self.down1 = down(ngf, ngf * 2)
        self.down2 = down(ngf * 2, ngf * 4)
        self.down3 = down(ngf * 4, ngf * 8)
        self.down4 = down(ngf * 8, ngf * 8)
        self.down5 = down(ngf * 8, ngf * 8)
        self.down6 = down(ngf * 8, ngf * 8)
        self.down7 = down(ngf * 8, ngf * 8)

        self.up7 = up(ngf * 8, ngf * 8)
        self.up6 = up(ngf * 16, ngf * 8)
        self.up5 = up(ngf * 16, ngf * 8)
        self.up4 = up(ngf * 16, ngf * 8)
        self.up3 = up(ngf * 16, ngf * 4)
        self.up2 = up(ngf * 8, ngf * 2)
        self.up1 = up(ngf * 4, ngf * 1, out=True)
        self.out = down(input_nc=ngf, output_nc=output_nc, kernal_size=3, stride=1, padding=1, out=True)

        self.down_bottom1 = down_bottom(ngf, ngf * 2, left_input=False)
        self.down_bottom2 = down_bottom(ngf * 2, ngf * 4)
        self.down_bottom3 = down_bottom(ngf * 4, ngf * 8)
        self.down_bottom4 = down_bottom(ngf * 8, ngf * 8)
        self.down_bottom5 = down_bottom(ngf * 8, ngf * 8)
        self.down_bottom6 = down_bottom(ngf * 8, ngf * 8)
        self.down_bottom7 = down_bottom(ngf * 8, ngf * 8)

        # x_up juanji,,,  shangcaiyang shuchu ,,,,,  he up jiazai yiqi shangcaiyangshuru

        self.up_bottom7 = up_bottom(ngf * 8, ngf * 8, ngf * 16)
        self.up_bottom6 = up_bottom(ngf * 16, ngf * 8, ngf * 32)
        self.up_bottom5 = up_bottom(ngf * 16, ngf * 8, ngf * 32)
        self.up_bottom4 = up_bottom(ngf * 16, ngf * 8, ngf * 32)
        self.up_bottom3 = up_bottom(ngf * 16, ngf * 4, ngf * 32)
        self.up_bottom2 = up_bottom(ngf * 8, ngf * 2, ngf * 16)
        self.up_bottom1 = up_bottom(ngf * 4, ngf * 1, ngf * 8, out=True)


    def forward(self, input1):
        x11 = self.transfer(input1)  # 256->256
        x12 = self.down1(x11)  # 256->128
        x13 = self.down2(x12)  # 128->64
        x14 = self.down3(x13)  # 64->32
        x15 = self.down4(x14)  # 32->16
        x16 = self.down5(x15)  # 16->8
        x17 = self.down6(x16)  # 8->4
        x18 = self.down7(x17)  # 4->2

        x177 = self.up7(x18, x17)
        x166 = self.up6(x177, x16)
        x155 = self.up5(x166, x15)
        x144 = self.up4(x155, x14)
        x133 = self.up3(x144, x13)
        x122 = self.up2(x133, x12)
        x111 = self.up1(x122, x11)
        x_first = self.out(x111)

        ##########################################

        x22 = self.down_bottom1(None, x11)
        x23 = self.down_bottom2(x22, x12)
        x24 = self.down_bottom3(x23, x13)
        x25 = self.down_bottom4(x24, x14)
        x26 = self.down_bottom5(x25, x15)
        x27 = self.down_bottom6(x26, x16)
        x28 = self.down_bottom7(x27, x17)

        x277 = self.up_bottom7(x18, x28, x27)
        x266 = self.up_bottom6(x177, x277, x26)
        x255 = self.up_bottom5(x166, x266, x25)
        x244 = self.up_bottom4(x155, x255, x24)
        x233 = self.up_bottom3(x144, x244, x23)
        x222 = self.up_bottom2(x133, x233, x22)
        x211 = self.up_bottom1(x122, x222, x222)
        x_second = self.out(x211)
        #####################
        x32 = self.down_bottom1(None, x11)
        x33 = self.down_bottom2(x32, x22)
        x34 = self.down_bottom3(x33, x23)
        x35 = self.down_bottom4(x34, x24)
        x36 = self.down_bottom5(x35, x25)
        x37 = self.down_bottom6(x36, x26)
        x38 = self.down_bottom7(x37, x27)

        x377 = self.up_bottom7(x28, x38, x37)
        x366 = self.up_bottom6(x277, x377, x36)
        x355 = self.up_bottom5(x266, x366, x35)
        x344 = self.up_bottom4(x255, x355, x34)
        x333 = self.up_bottom3(x244, x344, x33)
        x322 = self.up_bottom2(x233, x333, x32)
        x311 = self.up_bottom1(x222, x322, x222)
        x_third = self.out(x311)












        #op_unstable1 = torch.cat([self.op1(input1[:, 0:6, :, :]), self.op2(input1[:, 6:9, :, :])], dim=1)
        #op_unstable2 = torch.cat([self.op1(input2[:, 0:5, :, :]), self.op2(input2[:, 5:8, :, :])], dim=1)
        return [x_first, x_second, x_third]


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
