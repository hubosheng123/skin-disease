"""
Haar Wavelet-based Downsampling (HWD)
Original address of the paper: https://www.sciencedirect.com/science/article/abs/pii/S0031320323005174
Code reference: https://github.com/apple1986/HWD/tree/main
"""
import torch
import torch.nn as nn
from pytorch_wavelets import DWTForward
from pytorch_wavelets import DWTInverse

class HWDownsampling(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(HWDownsampling, self).__init__()
        self.wt = DWTForward(J=1, wave='haar', mode='zero')
        self.conv_bn_relu = nn.Sequential(
            nn.Conv2d(in_channel * 4, out_channel, kernel_size=1, stride=1),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        yL, yH = self.wt(x)
        y_HL = yH[0][:, :, 0, ::]
        y_LH = yH[0][:, :, 1, ::]
        y_HH = yH[0][:, :, 2, ::]
        x = torch.cat([yL, y_HL, y_LH, y_HH], dim=1)
        x = self.conv_bn_relu(x)
        return x

class HWUpsampling(nn.Module):
    def __init__(self, in_channel, out_channel):
        super().__init__()
        self.idwt = DWTInverse(wave='haar', mode='zero')
        self.conv = nn.Conv2d(in_channel // 4, out_channel, 1)  # 调整通道数

    def forward(self, x):
        # 拆分通道：假设输入x形状为 [B, C, H, W]
        c = x.shape[1] // 4
        yL = x[:, :c, :, :]
        yH = [x[:, c:, :, :].unsqueeze(2)]  # 重组高频分量（需适配实际结构）
        x_recon = self.idwt((yL, yH))
        return self.conv(x_recon)

if __name__ == '__main__':
    # downsampling_layer = HWDownsampling(3, 24)
    # input_data = torch.rand((1, 3, 64, 64))
    #output_data = downsampling_layer(input_data)
    c = torch.rand(1, 4, 64, 64)
    aa=HWUpsampling(24,12)
    aat=aa(c)
    # print("Input shape:", input_data.shape)
    print("Output shape:", c.shape)
    # print("a shape:", aat.output_shape)