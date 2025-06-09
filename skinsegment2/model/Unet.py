import torch
from thop import profile
from torch import nn
from torch.nn import functional as F
class conv_block(nn.Module):
    def __init__(self,in_channel,out_channel):
        super(conv_block,self).__init__()
        self.layer=nn.Sequential(
            nn.Conv2d(in_channels=in_channel,out_channels=out_channel,kernel_size=3,stride=1,padding=1,padding_mode="reflect"),
            nn.BatchNorm2d(out_channel),
            nn.Dropout(0.5),
            nn.LeakyReLU()
        )
    def forward(self,x):
        x=self.layer(x)
        return x

class downsample(nn.Module):
    def __init__(self,input_channel):
        super(downsample,self).__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(input_channel,input_channel, kernel_size=3, stride=2, padding=1, padding_mode="reflect"),
            nn.BatchNorm2d(input_channel),
            nn.LeakyReLU()
        )
    def forward(self,x):
        x=self.layer(x)
        return x

class upsample(nn.Module):
    def __init__(self,input_channel,output_channel):
        super(upsample,self).__init__()
        self.layer=nn.Sequential(
            nn.Conv2d(input_channel, output_channel, kernel_size=3, stride=1, padding=1, padding_mode="reflect"),
            nn.BatchNorm2d(output_channel),
            nn.LeakyReLU()
        )
    def forward(self,x):
        x=F.interpolate(x,scale_factor=2,mode='bilinear')
        x=self.layer(x)
        return x


class Unet(nn.Module):
    def __init__(self,num_class):
        super(Unet,self).__init__()
        self.con1=conv_block(3,64)
        self.con2=conv_block(64,64)
        self.down1=downsample(64)

        self.con3=conv_block(64,128)
        self.con4=conv_block(128,128)
        self.down2=downsample(128)

        self.con5=conv_block(128,256)
        self.con6=conv_block(256,256)
        self.down3=downsample(256)

        self.con7=conv_block(256,512)
        self.con8=conv_block(512,512)
        self.down4=downsample(512)

        self.con9 = conv_block(512, 1024)
        self.con10 = conv_block(1024, 1024)
        #--------------------------上面是encoder部分------------------------------

        self.up1=upsample(1024,512)
        self.con11 = conv_block(1024, 512)
        self.con12= conv_block(512, 512)

        self.up2 = upsample(512, 256)
        self.con13 = conv_block(512, 256)
        self.con14 = conv_block(256, 256)

        self.up3 = upsample(256, 128)
        self.con15 = conv_block(256, 128)
        self.con16 = conv_block(128, 128)

        self.up4 = upsample(128, 64)
        self.con17 = conv_block(128, 64)
        self.con18 = conv_block(64, 64)

        self.out=nn.Conv2d(in_channels=64,out_channels=num_class,kernel_size=3,stride=1,padding=1,padding_mode="reflect")
        self.TH=nn.Sigmoid()
    def forward(self,x):
        e1=self.con1(x)
        e1=self.con2(e1)

        e2=self.down1(e1)
        e2=self.con3(e2)
        e2=self.con4(e2)

        e3 = self.down2(e2)
        e3 = self.con5(e3)
        e3 = self.con6(e3)

        e4 = self.down3(e3)
        e4 = self.con7(e4)
        e4 = self.con8(e4)

        e5 = self.down4(e4)
        e5 = self.con9(e5)
        e5 = self.con10(e5)
    #----------------------------------------encode的部分---------------------------------------
        d1=self.up1(e5)
        d1=torch.cat((e4,d1),dim=1)  #图像处理当中dim=1可以看作channel的维度
        d1=self.con11(d1)
        d1=self.con12(d1)

        d2 = self.up2(d1)
        d2 = torch.cat((e3, d2), dim=1)
        d2 = self.con13(d2)
        d2 = self.con14(d2)

        d3 = self.up3(d2)
        d3 = torch.cat((e2, d3), dim=1)
        d3 = self.con15(d3)
        d3 = self.con16(d3)

        d4 = self.up4(d3)
        d4 = torch.cat((e1, d4), dim=1)
        d4 = self.con17(d4)
        d4 = self.con18(d4)

        output=self.out(d4)
        output=self.TH(output)
        output=torch.softmax(output,dim=1)
        return output

if __name__ == '__main__':
        model = Unet(2)

        input = torch.randn(1, 3, 224, 224)  # 输入样本

        flops, params = profile(model, inputs=(input,))
        print(f"FLOPs: {flops / 1e9:.2f} G, Params: {params / 1e6:.2f} M")

