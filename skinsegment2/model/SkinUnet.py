import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from thop import clever_format, profile
from torchsummary import summary
from .SeAttBlock import ECA_Spatial_Attention
from time import time
from fvcore.nn import FlopCountAnalysis
def autopad(k, p=None, d=1):  # 保持feature map大小不变

    if d > 1:
        # actual kernel-size
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
    if p is None:
        # auto-pad
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p


class Conv(nn.Module):

    default_act = nn.GELU()

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):  #
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)  # 没有使用空洞卷积 普通卷积
        self.bn = nn.BatchNorm2d(c2, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)  # eps防止分母是0
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        return self.act(self.conv(x))


class DWConv(Conv):  # 深度卷积
    """Depth-wise convolution with args(ch_in, ch_out, kernel, stride, dilation, activation)."""

    def __init__(self, c1, c2, k=1, s=1, d=1, act=True):
        super().__init__(c1, c2, k, s, g=math.gcd(c1, c2), d=d, act=act)


class ECDF(nn.Module):


    def __init__(self, c1, c2, N=32, shortcut=True, g=1, e=0.5):
        super().__init__()
        self.se_att=ECA_Spatial_Attention(c2)
        self.N = N
        self.c = int(c2 * e / self.N)
        self.add = shortcut and c1 == c2
        #self.wtc = WTConv2d( c2//2 , c2//2)
        self.pwconv1 = Conv(c1, c2 // self.N, 1, 1)
        self.pwconv2 = Conv(c2 // 2, c2, 1, 1)
        self.m = nn.ModuleList(DWConv(self.c, self.c, k=3, act=False) for _ in range((N - 2)//2))

    def forward(self, x):
        """Forward pass through CMRF Module."""
        x_residual = x
        x = self.pwconv1(x)  # 吧输出通道缩小N倍

        x_even,x_odd = [x[:, 0::2, :, :]],[x[:, 1::2, :, :]]
        x_even.extend( m(x_even[-1]) for m in self.m )
        x_odd.extend(  m(x_odd[-1]) for m in self.m)
        x=x_even+x_odd
        y = torch.cat(x, dim=1)  # 加上每次的结果
        #y = self.wtc(y)
        y = self.pwconv2(y)  # 点卷积
        y = self.se_att(y)

        return x_residual + y if self.add else y  # 加上残差



'''
U-shape/U-like Model
'''



class UNetEncoder(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UNetEncoder, self).__init__()
        self.ecdf = ECDF(in_channels, out_channels)
        self.downsample = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x = self.cmrf(x)
        return self.downsample(x), x


# Decoder in TinyU-Net
class UNetDecoder(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UNetDecoder, self).__init__()
        self.ecdf = ECDF(in_channels, out_channels)
        self.upsample = F.interpolate

    def forward(self, x, skip_connection):
        x = self.upsample(x, scale_factor=2, mode='bicubic', align_corners=False)
        x = torch.cat([x, skip_connection], dim=1)
        x = self.cmrf(x)
        return x


# TinyU-Net
class SkinUNet(nn.Module):
    """TinyU-Net with args(in_channels, num_classes)."""
    '''
    in_channels: The number of input channels
    num_classes: The number of segmentation classes
    '''

    def __init__(self, in_channels=3, num_classes=2):
        super(SkinUNet, self).__init__()
        in_filters = [192, 384, 768, 1024]
        out_filters = [64, 128, 256, 512]

        self.encoder1 = UNetEncoder(in_channels, 64)
        self.encoder2 = UNetEncoder(64, 128)
        self.encoder3 = UNetEncoder(128, 256)
        self.encoder4 = UNetEncoder(256, 512)

        self.decoder4 = UNetDecoder(in_filters[3], out_filters[3])
        self.decoder3 = UNetDecoder(in_filters[2], out_filters[2])
        self.decoder2 = UNetDecoder(in_filters[1], out_filters[1])
        self.decoder1 = UNetDecoder(in_filters[0], out_filters[0])
        self.final_conv = nn.Conv2d(out_filters[0], num_classes, kernel_size=1)

    def forward(self, x):
        x, skip1 = self.encoder1(x)
        x, skip2 = self.encoder2(x)
        x, skip3 = self.encoder3(x)
        x, skip4 = self.encoder4(x)

        x = self.decoder4(x, skip4)
        x = self.decoder3(x, skip3)
        x = self.decoder2(x, skip2)
        x = self.decoder1(x, skip1)
        x = self.final_conv(x)
        return x


if __name__ == '__main__':
     from thop import profile
     import  torch
     model = SkinUNet(3, 2)
     input = torch.randn(1, 3, 224, 224)  # 输入样本

     flops, params = profile(model, inputs=(input,))
     print(f"FLOPs: {flops / 1e9:.2f} G, Params: {params / 1e6:.2f} M")