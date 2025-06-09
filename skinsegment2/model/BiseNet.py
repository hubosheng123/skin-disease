import torch
import torch.nn as nn
import torch.nn.functional as F
from fvcore.nn import FlopCountAnalysis

# 双分支的BiSeNet
class BiSeNet(nn.Module):
    def __init__(self, num_classes):
        super(BiSeNet, self).__init__()

        self.num_classes = num_classes

        # 空间分支：输出通道数调整为 num_classes
        self.spatial_branch = self._make_spatial_branch()

        # 通道分支：输出通道数调整为 num_classes
        self.context_branch = self._make_context_branch()

        # 输出融合层，输入通道数为 2 * num_classes
        self.fusion = nn.Conv2d(2 * num_classes, num_classes, kernel_size=1, stride=1, padding=0)

    def _make_spatial_branch(self):
        # 空间分支用于提取空间信息，输出通道数设为 num_classes
        layers = [
            nn.Conv2d(3, self.num_classes, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.num_classes, self.num_classes, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.num_classes, self.num_classes, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
        ]
        return nn.Sequential(*layers)

    def _make_context_branch(self):
        # 通道分支用于提取语义信息，输出通道数设为 num_classes
        layers = [
            nn.Conv2d(3, self.num_classes, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.num_classes, self.num_classes, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.num_classes, self.num_classes, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
        ]
        return nn.Sequential(*layers)

    def forward(self, x):
        # 通过空间分支提取空间信息
        spatial_features = self.spatial_branch(x)

        # 通过通道分支提取语义信息
        context_features = self.context_branch(x)

        # 上采样空间分支的特征图，使其与通道分支的特征图大小一致
        spatial_features = F.interpolate(spatial_features, size=context_features.shape[2:], mode='bilinear',
                                         align_corners=False)

        # 融合空间分支和通道分支的特征
        combined_features = torch.cat([spatial_features, context_features], dim=1)

        # 最后的预测层
        output = self.fusion(combined_features)

        return output
if __name__ == '__main__':
    model = BiSeNet(num_classes=2)
    from time import time

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)


    dummy_input = torch.randn(1, 3, 224, 224).to(device)
    def count_parameters(model):
        return sum(p.numel() for p in model.parameters())
    print(count_parameters(model))
    flops = FlopCountAnalysis(model, dummy_input)
    print("FLOPS:", flops.total())

# 模型实例化
# num_classes = 2  # 假设有21个类别
# model = BiSeNet(num_classes)
#
# # 打印模型结构
# print(model)
#
#
# # 测试网络的前向传播
#
#     # 假设输入图像大小为(batch_size, 3, H, W)，这里H和W可以是任意大小，假设为256x256
# batch_size = 2
# height = 224
# width = 224
#
#     # 创建随机输入数据
# input_data = torch.randn(batch_size, 3, height, width)
#
#     # 前向传播
# output = model(input_data)
#
#     # 打印输出的形状
# print(f"Input shape: {input_data.shape}")
# print(f"Output shape: {output.shape}")


# 测试模型的前向传播




