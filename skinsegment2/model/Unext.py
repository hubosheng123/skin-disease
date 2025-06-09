import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class ConvBlock(nn.Module):
    """卷积块：两层 3x3 卷积 + 批归一化 + ReLU"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)


class ShiftedWindowTransformer(nn.Module):
    """移位窗口 Transformer 模块（简化版）"""

    def __init__(self, dim, num_heads=4, window_size=8):
        super().__init__()
        self.window_size = window_size
        self.norm = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)

    def forward(self, x):
        # 输入尺寸: [B, C, H, W]
        B, C, H, W = x.shape
        x = rearrange(x, 'b c h w -> b (h w) c')  # [B, H*W, C]

        # LayerNorm + 自注意力
        x = self.norm(x)
        attn_out, _ = self.attn(x, x, x)  # [B, H*W, C]

        # 恢复空间维度
        attn_out = rearrange(attn_out, 'b (h w) c -> b c h w', h=H, w=W)
        return attn_out


class UNeXt(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, embed_dims=[32, 64, 128, 256], num_heads=4):
        super().__init__()

        # ------------------- 编码器部分 -------------------
        self.encoder1 = ConvBlock(in_channels, embed_dims[0])
        self.pool1 = nn.MaxPool2d(2)

        self.encoder2 = ConvBlock(embed_dims[0], embed_dims[1])
        self.pool2 = nn.MaxPool2d(2)

        self.encoder3 = ConvBlock(embed_dims[1], embed_dims[2])
        self.pool3 = nn.MaxPool2d(2)

        # 底部 Transformer 层
        self.bottleneck = nn.Sequential(
            ConvBlock(embed_dims[2], embed_dims[3]),
            ShiftedWindowTransformer(embed_dims[3], num_heads=num_heads)
        )

        # ------------------- 解码器部分 -------------------
        self.upconv3 = nn.ConvTranspose2d(embed_dims[3], embed_dims[2], kernel_size=2, stride=2)
        self.decoder3 = ConvBlock(embed_dims[3], embed_dims[2])

        self.upconv2 = nn.ConvTranspose2d(embed_dims[2], embed_dims[1], kernel_size=2, stride=2)
        self.decoder2 = ConvBlock(embed_dims[2], embed_dims[1])

        self.upconv1 = nn.ConvTranspose2d(embed_dims[1], embed_dims[0], kernel_size=2, stride=2)
        self.decoder1 = ConvBlock(embed_dims[1], embed_dims[0])

        # 最终分割头
        self.seg_head = nn.Conv2d(embed_dims[0], out_channels, kernel_size=1)

    def forward(self, x):
        # ------------------- 编码器 -------------------
        e1 = self.encoder1(x)  # [B, 32, H, W]
        p1 = self.pool1(e1)  # [B, 32, H/2, W/2]

        e2 = self.encoder2(p1)  # [B, 64, H/2, W/2]
        p2 = self.pool2(e2)  # [B, 64, H/4, W/4]

        e3 = self.encoder3(p2)  # [B, 128, H/4, W/4]
        p3 = self.pool3(e3)  # [B, 128, H/8, W/8]

        # 底部 Transformer
        bottleneck = self.bottleneck(p3)  # [B, 256, H/8, W/8]

        # ------------------- 解码器 -------------------
        d3 = self.upconv3(bottleneck)  # [B, 128, H/4, W/4]
        d3 = torch.cat([e3, d3], dim=1)  # [B, 256, H/4, W/4]
        d3 = self.decoder3(d3)  # [B, 128, H/4, W/4]

        d2 = self.upconv2(d3)  # [B, 64, H/2, W/2]
        d2 = torch.cat([e2, d2], dim=1)  # [B, 128, H/2, W/2]
        d2 = self.decoder2(d2)  # [B, 64, H/2, W/2]

        d1 = self.upconv1(d2)  # [B, 32, H, W]
        d1 = torch.cat([e1, d1], dim=1)  # [B, 64, H, W]
        d1 = self.decoder1(d1)  # [B, 32, H, W]

        # 输出分割结果
        out = self.seg_head(d1)  # [B, out_channels, H, W]
        return out


# ------------------- 测试代码 -------------------
if __name__ == "__main__":
    from thop import profile
    import torch

    model = UNeXt(3, 2)
    input = torch.randn(1, 3, 224, 224)  # 输入样本

    flops, params = profile(model, inputs=(input,))
    print(f"FLOPs: {flops / 1e9:.2f} G, Params: {params / 1e6:.2f} M")
