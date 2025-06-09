import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import torch
import torch.nn as nn


class ECABlock(nn.Module):
    def __init__(self, in_channels, gamma=2, b=1):
        super().__init__()
        self.gap = nn.AdaptiveAvgPool2d(1)
        # 自适应卷积核大小计算
        kernel_size = self._get_kernel_size(in_channels, gamma, b)
        self.conv = nn.Conv1d(1, 1, kernel_size=kernel_size, padding=(kernel_size-1)//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def _get_kernel_size(self, c, gamma, b):
        # 计算自适应核大小（公式：k = nearest_odd(log2(c)/gamma + b))
        k = int(math.log2(c) / gamma + b)
        k = k if k % 2 else k + 1  # 确保为奇数
        return max(k, 3)  # 最小核大小为3

    def forward(self, x):
        b, c, h, w = x.size()
        # Squeeze阶段（GAP）
        squeeze = self.gap(x).view(b, c, 1)  # [B,C,1]
        # Excitation阶段（1D卷积）
        weights = self.conv(squeeze.permute(0,2,1))  # [B,1,C]
        weights = self.sigmoid(weights.permute(0,2,1))  # [B,C,1]
        # 扩展维度并返回权重
        return weights.view(b, c, 1, 1)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        # 深度可分离卷积 + 自适应核大小（可选）
        self.conv = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, groups=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        concat = torch.cat([avg_out, max_out], dim=1)
        return self.conv(concat)  # [B,1,H,W]


class ECA_Spatial_Attention(nn.Module):
    def __init__(self, in_channels, gamma=2, b=1, kernel_size=7):
        super().__init__()
        self.eca = ECABlock(in_channels, gamma, b)
        self.spatial = SpatialAttention(kernel_size)

        self.alpha = nn.Parameter(torch.ones(1))
        self.beta = nn.Parameter(torch.ones(1))

    def forward(self, x):
        channel_weights = self.eca(x)  # 通道权重 [B,C,1,1]
        spatial_weights = self.spatial(x)  # 空间权重 [B,1,H,W]
        # 权重融合（相加或相乘）
        combined_weights = self.alpha * channel_weights + self.beta * spatial_weights
        return x * combined_weights.expand_as(x)


# class SEBlock(nn.Module):
#     def __init__(self, in_channels, reduction=8):
#         super(SEBlock, self).__init__()
#         # Squeeze操作（全局特征压缩）
#         self.gap = nn.AdaptiveAvgPool2d(1)  # [B,C,H,W] -> [B,C,1,1]
#
#         # Excitation操作（通道关系建模）
#         self.fc = nn.Sequential(
#             nn.Linear(in_channels, in_channels // reduction),  # 降维全连接层
#             nn.ReLU(inplace=True),
#             nn.Linear(in_channels // reduction, in_channels),  # 升维全连接层
#             nn.Sigmoid()  # 权重归一化[0,1]
#         )
#
#     def forward(self, x):
#         b, c, _, _ = x.size()
#         # Squeeze阶段
#         squeeze = self.gap(x).view(b, c)  # [B,C,1,1] -> [B,C]
#
#         # Excitation阶段
#         weights = self.fc(squeeze).view(b, c, 1, 1)  # [B,C] -> [B,C,1,1]
#
#         # Reweight阶段（通道加权）
#         return x * weights.expand_as(x)  # 广播机制实现逐通道乘法
#
# class SE_Spatial_Attention(nn.Module):
#     def __init__(self, in_channels, reduction=8, kernel_size=7):
#         super().__init__()
#         # 通道注意力（SE模块）
#         self.se = SEBlock(in_channels, reduction)
#         # 空间注意力（CBAM中的空间部分）
#         self.spatial = nn.Sequential(
#             nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2),
#             nn.Sigmoid()
#         )
#
#     def forward(self, x):
#         # 通道重校准
#         x = self.se(x)
#         # 空间注意力计算
#         avg_out = torch.mean(x, dim=1, keepdim=True)  # 通道平均池化
#         max_out, _ = torch.max(x, dim=1, keepdim=True)  # 通道最大池化
#         spatial_weights = self.spatial(torch.cat([avg_out, max_out], dim=1))
#         # 空间加权
#
#         return x * spatial_weights
if __name__ == '__main__':
    for i in [64,128,256,512]:
      x=torch.randn(8,i,224,224)
      se=Parallel_SE_Spatial_Attention(i)
      print(se(x).shape)
