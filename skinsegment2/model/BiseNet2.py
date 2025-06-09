import torch
import torch.nn as nn
import torch.nn.functional as F
from thop import profile
from torchvision.models import resnet18
from torchsummary import summary

class AttentionRefinementModule(nn.Module):
    """ARM (Attention Refinement Module)"""

    def __init__(self, in_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.bn = nn.BatchNorm2d(in_channels)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # 全局平均池化 + 卷积 + BN + Sigmoid
        attention = F.adaptive_avg_pool2d(x, 1)
        attention = self.conv(attention)
        attention = self.bn(attention)
        attention = self.sigmoid(attention)
        return x * attention


class FeatureFusionModule(nn.Module):
    """FFM (Feature Fusion Module)"""

    def __init__(self, num_classes):
        super().__init__()
        # 输入通道：spatial_path (num_classes) + context_path (256)
        self.conv1 = nn.Conv2d(256 + num_classes, num_classes, kernel_size=1)
        self.bn = nn.BatchNorm2d(num_classes)
        self.relu = nn.ReLU(inplace=True)

        # 通道注意力
        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(num_classes, num_classes, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, spatial_feat, context_feat):
        # 调整空间分支尺寸（与Context Path对齐）
        spatial_feat = F.interpolate(
            spatial_feat,
            size=context_feat.shape[2:],
            mode='bilinear',
            align_corners=False
        )

        # 拼接特征
        fused = torch.cat([spatial_feat, context_feat], dim=1)
        fused = self.conv1(fused)
        fused = self.bn(fused)
        fused = self.relu(fused)

        # 注意力加权
        att = self.attention(fused)
        output = fused * att + fused
        return output


class SpatialPath(nn.Module):
    """Spatial Path (保留高分辨率空间信息)"""

    def __init__(self, num_classes):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, num_classes, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(num_classes),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.layers(x)


class BiSeNet(nn.Module):
    """完整的BiSeNet模型"""

    def __init__(self, num_classes, backbone='resnet18'):
        super().__init__()
        self.num_classes = num_classes

        # 1. 构建双分支
        self.spatial_path = SpatialPath(num_classes)
        self._build_context_path(backbone)

        # 2. 注意力与融合模块
        self.arm = AttentionRefinementModule(256)
        self.ffm = FeatureFusionModule(num_classes)

    def _build_context_path(self, backbone):
        """Context Path (使用预训练骨干网络)"""
        if backbone == 'resnet18':
            base_model = resnet18(pretrained=True)
            self.context_path = nn.Sequential(
                base_model.conv1,
                base_model.bn1,
                base_model.relu,
                base_model.maxpool,
                base_model.layer1,
                base_model.layer2,
                base_model.layer3  # 输出1/8尺寸
            )
            self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        else:
            raise ValueError("仅支持resnet18作为骨干网络")

    def forward(self, x):
        # 1. 双分支前向传播
        spatial_out = self.spatial_path(x)  # [N, num_classes, H/8, W/8]

        context_out = self.context_path(x)  # [N, 256, H/8, W/8]
        global_feat = self.global_avg_pool(context_out)
        context_out = self.arm(context_out)  # 注意力细化

        # 2. 特征融合
        output = self.ffm(spatial_out, context_out)

        # 3. 上采样至输入分辨率
        output = F.interpolate(
            output,
            size=x.shape[2:],
            mode='bilinear',
            align_corners=False
        )
        return output
if '__main__' == __name__:

    model = BiSeNet(2)

    input = torch.randn(1,3,224,224)  # 输入样本

    flops, params = profile(model, inputs=(input,))
    print(f"FLOPs: {flops / 1e9:.2f} G, Params: {params / 1e6:.2f} M")