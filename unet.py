import torch
import torch.fft
import torch.nn as nn
from signal_utils import torch_ifft, torch_fft

from kornia.geometry.transform import translate, rotate

GPU = torch.cuda.is_available()
Tensor = torch.cuda.FloatTensor if GPU else torch.Tensor

class ConvBlock(torch.nn.Module):
    def __init__(self, input_channel, output_channel,batch_normalization=True):
        super(ConvBlock, self).__init__()

        self.conv1 = torch.nn.Conv2d(input_channel, output_channel, 3, padding=1)
        self.bn1 = torch.nn.BatchNorm2d(output_channel)
        self.conv2 = torch.nn.Conv2d(output_channel,output_channel,3,padding=1)
        self.bn2 = torch.nn.BatchNorm2d(output_channel)
        self.relu = torch.nn.ReLU()
        self.batch_normalization = batch_normalization

    def forward(self, x):
        x = self.conv1(x)
        if self.batch_normalization:
            x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        if self.batch_normalization:
            x = self.bn2(x)

        x=self.relu(x)

        return x

class DownSample(torch.nn.Module):
    def __init__(self, factor=2):
        super(DownSample, self).__init__()
        self.down_sample = torch.nn.MaxPool2d(factor, factor)

    def forward(self,x):
        return self.down_sample(x)


class UpSample(torch.nn.Module):
    def __init__(self, factor=2):
        super(UpSample, self).__init__()
        self.up_sample = torch.nn.Upsample(scale_factor=factor, mode='bilinear')
    def forward(self,x):
        return self.up_sample(x)


class CropConcat(torch.nn.Module):
    def __init__(self,crop = True):
        super(CropConcat, self).__init__()
        self.crop = crop

    def do_crop(self,x, tw, th):
        b,c,w, h = x.size()
        x1 = int(round((w - tw) / 2.))
        y1 = int(round((h - th) / 2.))
        return x[:,:,x1:x1 + tw, y1:y1 + th]

    def forward(self,x,y):
        b, c, h, w = y.size()
        if self.crop:
            x = self.do_crop(x,h,w)
        return torch.cat((x,y),dim=1)


class UpBlock(torch.nn.Module):
    def __init__(self,input_channel, output_channel,batch_normalization=True,downsample = False):
        super(UpBlock, self).__init__()
        self.downsample = downsample
        self.conv = ConvBlock(input_channel,output_channel,batch_normalization=batch_normalization)
        self.downsampling = DownSample()

    def forward(self,x):
        x1 = self.conv(x)
        if self.downsample:
            x = self.downsampling(x1)
        else:
            x = x1
        return x,x1

class DownBlock(torch.nn.Module):
    def __init__(self,input_channel, output_channel,batch_normalization=True,Upsample = False):
        super(DownBlock, self).__init__()
        self.Upsample = Upsample
        self.conv = ConvBlock(input_channel,output_channel,batch_normalization=batch_normalization)
        self.upsampling = UpSample()
        self.crop = CropConcat()

    def forward(self,x,y):
        if self.Upsample:
            x = self.upsampling(x)
        x = self.crop(y,x)
        x = self.conv(x)
        return x


class ResUNet(torch.nn.Module):
    def __init__(self, input_channel, output_channel, domain):
        super(ResUNet, self).__init__()
        self.input_channel = input_channel
        self.domain = domain

        #Down Blocks
        self.conv_block1 = ConvBlock(input_channel,64)
        self.conv_block2 = ConvBlock(64,128)
        self.conv_block3 = ConvBlock(128,256)
        self.conv_block4 = ConvBlock(256,512)
        self.conv_block5 = ConvBlock(512,1024)

        #Up Blocks
        self.conv_block6 = ConvBlock(1024+512, 512)
        self.conv_block7 = ConvBlock(512+256, 256)
        self.conv_block8 = ConvBlock(256+128, 128)
        self.conv_block9 = ConvBlock(128+64, 64)

        #Last convolution
        self.last_conv = torch.nn.Conv2d(64, output_channel,1)

        self.crop = CropConcat()

        self.downsample = DownSample()
        self.upsample =   UpSample()

    def forward(self, x, mask):
        # trans: 2 t1 (img)
        if self.input_channel == 2:
            t2u_input = None
        elif self.input_channel == 1:
            raise NotImplementedError
            t2u_input = x.clone()
        elif self.input_channel == 6:
            raise NotImplementedError
            t2u_input = x[:,3:].clone()
        # reg: 2 t2s, 2 t2u, rec: 2 t2sw, t2u
        elif self.input_channel == 4:
            t2u_input = x[:,2:].clone()
        if torch.is_complex(x):
            x = x.real.type(torch.float)
        x1 = self.conv_block1(x) #  64      192 192

        x = self.downsample(x1)  #  64      96  96
        x2 = self.conv_block2(x) # 128      96  96

        x= self.downsample(x2)   # 128      48  48
        x3 = self.conv_block3(x) # 256      48  48

        x= self.downsample(x3)   # 256      24  24
        x4 = self.conv_block4(x) # 512      24  24

        x = self.downsample(x4)  # 512      12  12
        x5 = self.conv_block5(x) #1024      12  12

        x = self.upsample(x5)    #1024      24  24
        x6 = self.crop(x4, x)     #1024+512  24  24
        x6 = self.conv_block6(x6)  # 512      24  24

        x = self.upsample(x6)     # 512      48  48
        x7 = self.crop(x3,x)      # 512+256  48  48
        x7 = self.conv_block7(x7)  # 256      48  48

        x = self.upsample(x7)     # 256      96  96
        x8 = self.crop(x2,x)       # 256+128  96  96
        x8 = self.conv_block8(x8)   # 128      96  96

        x = self.upsample(x8)      # 128     192 192
        x = self.crop(x1,x)       # 128+64  192 192
        x = self.conv_block9(x)   #  64     192 192

        # x = self.last_conv(x) + t2u_input    #   1     192 192
        x = self.last_conv(x)
        if mask is not None and t2u_input is not None:
            DCed = self.DC(x, t2u_input, mask)
            return DCed
        else:
            return x


    def DC(self, pred, x_input, mask):
        ffted_pred = torch.complex(pred[:, :1], pred[:, 1:])
        ffted_input = torch.complex(x_input[:, :1], x_input[:, 1:])
        if self.domain == 'i':
            ffted_pred = torch_fft(ffted_pred, (-2, -1))
            ffted_input = torch_fft(ffted_input, (-2, -1))
        mask = Tensor(mask)
        combined_k = ffted_pred * (1 - mask) + ffted_input * mask
        combined = torch_ifft(combined_k, (-2, -1))

        if self.domain == 'k':
            return torch.concat((combined_k.real, combined_k.imag), 1)
        elif self.domain == 'i':
            return torch.concat((combined.real, combined.imag), 1)


class RegNetEncoder(torch.nn.Module):
    def __init__(self):
        super(RegNetEncoder, self).__init__()
        self.conv_block1 = ConvBlock(1, 32)
        self.conv_block2 = ConvBlock(32, 64)
        self.conv_block3 = ConvBlock(64, 128)
        self.downsample = DownSample()

    def forward(self, x):
        x1 = self.conv_block1(x)  # 64      192 192

        x = self.downsample(x1)  # 64      96  96
        x2 = self.conv_block2(x)  # 128      96  96

        x = self.downsample(x2)  # 128      48  48
        x3 = self.conv_block3(x)  # 256      48  48

        x = self.downsample(x3)  # 256      24  24

        return x


class RegNetDecoder(torch.nn.Module):
    def __init__(self):
        super(RegNetDecoder, self).__init__()
        self.conv_block6 = ConvBlock(256, 128)
        self.conv_block7 = ConvBlock(128, 64)
        self.conv_block8 = ConvBlock(64, 32)
        self.last_conv = torch.nn.Conv2d(32, 2, 1)
        self.upsample = UpSample()

    def forward(self, x):
        x = self.upsample(x)  # 1024      24  24
        x6 = self.conv_block6(x)  # 512      24  24

        x = self.upsample(x6)  # 512      48  48
        x7 = self.conv_block7(x)  # 256      48  48

        x = self.upsample(x7)  # 256      96  96
        x8 = self.conv_block8(x)  # 128      96  96

        x = self.last_conv(x8)

        return x


class RegNet(torch.nn.Module):
    def __init__(self, input_channel, domain):
        super(RegNet, self).__init__()
        self.input_channel = input_channel
        self.domain = domain


        self.localization = nn.Sequential(
            nn.Conv2d(2, 8, kernel_size=7),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(),
            nn.Conv2d(8, 10, kernel_size=5),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU()
        )

        self.fc_loc = nn.Sequential(
            nn.Linear(8 * 12 * 12, 32),
            nn.ReLU(),
            nn.Linear(32, 3)
        )

        self.fc_loc[2].weight.data.zero_()
        self.fc_loc[2].bias.data.copy_(torch.tensor([0, 0, 0], dtype=torch.float))

        self.conv_block11 = ConvBlock(input_channel, 16)
        self.conv_block22 = ConvBlock(16, 32)
        self.conv_block33 = ConvBlock(32, 16)
        self.conv_block44 = ConvBlock(16, 8)
        self.downsample = DownSample()


    def unet_encoder(self, x):
        x1 = self.conv_block11(x)  # 64      192 192

        x = self.downsample(x1)  # 64      96  96
        x2 = self.conv_block22(x)  # 128      96  96

        x = self.downsample(x2)  # 16      48  48
        x3 = self.conv_block33(x)  # 256      48  48

        x = self.downsample(x3)  # 256      24  24
        x4 = self.conv_block44(x)  # 512      24  24

        x = self.downsample(x4)  # 512      12  12

        return x

    def get_image_params(self, x, moving):
        xs = self.unet_encoder(torch.concat((x, moving), 1))
        xs = xs.contiguous().view(-1, 8 * 12 * 12)
        theta = self.fc_loc(xs)
        theta = theta.view(-1, 3)
        return theta

    def get_k_motion_field(self, x, moving, part):
        x_encoded = self.encoder[part](x)
        m_encoded = self.encoder[part](moving)
        concated = torch.concat((x_encoded, m_encoded), 1)
        return self.decoder[part](concated)

    def stn(self, x, moving):
        if self.domain == 0 or self.domain == 1: # k-space or image
            grid = self.get_image_params(x, moving)
        else:
            raise ValueError()

        if self.domain == 0:
            moving = torch_ifft(moving[:, :1] + 1j * moving[:, 1:2], (-2, -1))
            moving = torch.concat((moving.real, moving.imag), 1)

        rotated_x = rotate(moving, grid[:, 0])
        translated_x = translate(rotated_x, grid[:, 1:])
        if self.domain == 0:
            translated_x = torch_fft(translated_x[:, :1] + 1j * translated_x[:, 1:], (-2, -1))
            translated_x = torch.concat((translated_x.real, translated_x.imag), 1)
        return translated_x, grid

    def forward(self, x, moving):
        return self.stn(x, moving)


