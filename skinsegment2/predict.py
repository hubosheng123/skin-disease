import argparse
from model.SkinUnet import SkinUNet
import numpy as np
import torch
import os
import torch.nn as nn
import torchvision.transforms as T
from model.Unet import Unet
from model.BiseNet import BiSeNet
from model.Unext import UNeXt
from PIL import Image
def get_args_parser():
    parser = argparse.ArgumentParser('Predict Image', add_help=False)
    parser.add_argument('--image_path', default=r'.\2018.jpg', type=str, metavar='MODEL',help='Name of model to train')
    parser.add_argument('--input_size', default=[224,224],nargs='+',type=int,help='images input size')
    parser.add_argument('--weights', default=r'.\trained_model\Unet.pth', type=str,help='dataset path')
    parser.add_argument('--nb_classes', default=2, type=int,help='number of the classification types')
    parser.add_argument('--device', default='cuda',help='device to use for training / testing')

    return parser
def main(args):
    device = torch.device(args.device)

    image = Image.open(args.image_path).convert('RGB')
    img_size = image.size

    transforms = T.Compose([
        T.Resize(args.input_size),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]),
    ])
    model = Unet(2)
    #model = SkinUNet(in_channels=3,num_classes=args.nb_classes)
   # model = BiSeNet(num_classes=args.nb_classes)
    #model= UNeXt(in_channels=3, out_channels=2)
    checkpoint = torch.load(args.weights, map_location='cpu')
    msg = model.load_state_dict(checkpoint, strict=True)
    print(msg)

    model.to(device)
    model.eval()

    input_tensor = transforms(image).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(input_tensor)
        pred = output.argmax(1).squeeze(0).cpu().numpy().astype(np.uint8)
        pred[pred == 1] = 255
    mask = Image.fromarray(pred)
    out = mask.resize(img_size)
    out.save("resulttttttt.png")
if __name__ == '__main__':
    #获取训练参数
    args = get_args_parser()
    #解析训练参数
    args = args.parse_args()
    #训练参数传入主函数
    main(args)

