import torch
import torch.nn as nn
import torchvision.models as models
from thop import profile
from torch.nn import functional as F

class ASPP(nn.Module):
    def __init__(self, in_channels, out_channels=256):
        super(ASPP, self).__init__()
        
        # 1x1卷积分支
        self.conv_1x1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )
        
        # 膨胀率6的3x3卷积
        self.conv_6_1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=6, dilation=6, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )
        
        # 膨胀率12的3x3卷积
        self.conv_12_1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=12, dilation=12, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )
        
        # 膨胀率18的3x3卷积
        self.conv_18_1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=18, dilation=18, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )
        
        # 全局平均池化分支
        self.avg_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),  # 输出尺寸 (batch_size, in_channels, 1, 1)
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )
        
        # 融合卷积
        self.fusion = nn.Sequential(
            nn.Conv2d(out_channels*5, out_channels, 1, bias=False),  # 5个分支拼接
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Dropout(0.5)
        )

    def forward(self, x):
        # 原始特征图尺寸 (batch_size, in_channels, h, w)
        h, w = x.shape[2], x.shape[3]
        
        # 各分支处理
        conv_1x1 = self.conv_1x1(x)
        conv_6x6 = self.conv_6_1(x)
        conv_12x12 = self.conv_12_1(x)
        conv_18x18 = self.conv_18_1(x)
        global_feat = self.avg_pool(x)
        global_feat = F.interpolate(global_feat, (h, w), mode='bilinear', align_corners=True)  # 上采样到原尺寸
        
        # 沿通道维度拼接 (batch_size, out_channels*5, h, w)
        concat = torch.cat([conv_1x1, conv_6x6, conv_12x12, conv_18x18, global_feat], dim=1)
        
        # 融合后输出 (batch_size, out_channels, h, w)
        return self.fusion(concat)

class DeepLabV3(nn.Module):
    def __init__(self, num_classes, output_stride=16):
        super(DeepLabV3, self).__init__()
        
        # 使用ResNet作为主干网络
        resnet = models.resnet50(pretrained=True)
        
        # 修改ResNet的stage4 (根据output_stride调整空洞卷积)
        if output_stride == 16:
            # 最后一个block的膨胀率设为2，步长保持1
            resnet.layer4[0].conv2.stride = (1, 1)
            resnet.layer4[0].downsample[0].stride = (1, 1)
            for block in resnet.layer4[1:]:
                block.conv2.dilation = (2, 2)
                block.conv2.padding = (2, 2)
        
        # 移除全连接层和池化层
        self.backbone = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,
            resnet.layer2,
            resnet.layer3,
            resnet.layer4
        )
        
        # ASPP模块 (ResNet最后一层输出通道数为2048)
        self.aspp = ASPP(in_channels=2048)
        
        # 分类头
        self.classifier = nn.Sequential(
            nn.Conv2d(256, num_classes, 1)  # 保持空间维度不变
        )

    def forward(self, x):
        # 输入尺寸 (batch_size, 3, h, w)
        original_size = x.shape[2:]
        
        # 主干网络输出 (batch_size, 2048, h/16, w/16)
        features = self.backbone(x)
        
        # ASPP处理 (batch_size, 256, h/16, w/16)
        aspp_out = self.aspp(features)
        
        # 上采样回原图尺寸 (batch_size, num_classes, h, w)
        out = self.classifier(aspp_out)
        out = F.interpolate(out, size=original_size, 
                           mode='bilinear', align_corners=True)
        
        return out

# 测试尺寸匹配
if __name__ == '__main__':
        model = DeepLabV3(2)

        input = torch.randn(1, 3, 224, 224)  # 输入样本

        flops, params = profile(model, inputs=(input,))
        print(f"FLOPs: {flops / 1e9:.2f} G, Params: {params / 1e6:.2f} M")



