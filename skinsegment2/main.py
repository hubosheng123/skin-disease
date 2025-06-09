import os
import torch
import torch.nn as nn
import torch.utils.data
from model.Unet import Unet
from model.Transunet import TransUNet
from model.SkinUnet import SkinUNet
from model.BiseNet import BiSeNet
from model.UnetP import R2U_Net,NestedUNet
from utils.engine import train_and_val
import argparse
from model.SegNet import SegNet
import numpy as np
from model.Unext import UNeXt
from utils.transform import Resize,Compose,ToTensor,Normalize,RandomHorizontalFlip
from utils.datasets import SegData
import torch.optim.lr_scheduler
import json
def get_args_parser():
    parser = argparse.ArgumentParser('Image Segmentation Train', add_help=False)
    parser.add_argument('--batch_size', default=50, type=int,help='Batch size for training')
    parser.add_argument('--epochs', default=300, type=int)
    parser.add_argument('--input_size', default=[224,224],nargs='+',type=int,help='images input size')
    parser.add_argument('--data_path', default='./datasets/2018', type=str,help='dataset path')

    parser.add_argument('--init_lr', default=1e-5, type=float,help='intial lr')
    parser.add_argument('--max_lr', default=1e-3, type=float,help='max lr')
    parser.add_argument('--weight_decay', default=1e-5, type=float,help='weight decay')

    parser.add_argument('--nb_classes', default=2, type=int,help='number of the classification types')
    parser.add_argument('--output_dir', default='./output_dir',help='path where to save, empty for no saving')
    parser.add_argument('--device', default='cuda',help='device to use for training / testing')
    parser.add_argument('--num_workers', default=16, type=int)
    parser.add_argument('--model_name', type=str, default='SkinUnet',
                        choices=['Unet', 'Transunet', 'SkinUnet','BiSeNet','UNeXt','SegNet'],
                        help='选择模型架构')
    return parser
def main(args):

    device = torch.device(args.device)
    print(device)
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    train_transform = Compose([
                                    Resize(args.input_size),
                                    RandomHorizontalFlip(0.5),
                                    ToTensor(),
                                    Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                             ])

    val_transform = Compose([
                                    Resize(args.input_size),
                                    ToTensor(),
                                    Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                            ])

    train_dataset = SegData(image_path=os.path.join(args.data_path, 'images/Training_Input'),
                            mask_path=os.path.join(args.data_path, 'labels/Training_label'),
                            data_transforms=train_transform)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True,
                                               num_workers=args.num_workers,pin_memory=True,persistent_workers=True )

    val_dataset = SegData(image_path=os.path.join(args.data_path, 'images/Val_Input'),
                            mask_path=os.path.join(args.data_path, 'labels/Val_label'),
                            data_transforms=val_transform)
    val_loader = torch.utils.data.DataLoader(dataset=val_dataset, batch_size=args.batch_size, shuffle=False,
                                             num_workers=args.num_workers,pin_memory=True)

    #model = TinyUNet(in_channels=3,num_classes=args.nb_classes)
    # model = TransUNet(img_dim=224,
    #                       in_channels=3,
    #                       out_channels=128,
    #                       head_num=4,
    #                       mlp_dim=512,
    #                       block_num=8,
    #                       patch_dim=16,
    #                       class_num=2)
    #model = R2U_Net(img_ch=3,output_ch=args.nb_classes)
    #model=NestedUNet(out_ch=2)
    if args.model_name == 'SegNet':
        model = SegNet(3,args.nb_classes)
    if args.model_name == 'UNeXt':
        model = UNeXt(in_channels=3, out_channels=2)
    if args.model_name == 'SkinUnet':
        model = SkinUNet(in_channels=3,num_classes=args.nb_classes)
    if args.model_name == 'Unet':
        model = Unet(num_class=args.nb_classes)
    if args.model_name == 'Transunet':
        model = TransUNet(img_dim=224,
                          in_channels=3,
                          out_channels=128,
                          head_num=4,
                          mlp_dim=512,
                          block_num=8,
                          patch_dim=16,
                          class_num=2)
    if args.model_name == 'BiSeNet':
        model = BiSeNet(num_classes=args.nb_classes)
    loss_function = nn.CrossEntropyLoss()

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.init_lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, args.max_lr, total_steps=args.epochs, verbose=True)

    history = train_and_val(args.epochs, model, train_loader,val_loader,loss_function, optimizer,scheduler,args.output_dir,device,args.nb_classes)

    with open('output_dir/mymodel/2018/history-2018.json', 'w', encoding='utf-8') as file:

        json.dump(history, file, ensure_ascii=False, indent=4)
    # plot_loss(np.arange(0,args.epochs),args.output_dir, history)
    # plot_pix_acc(np.arange(0,args.epochs),args.output_dir, history)
    # plot_miou(np.arange(0,args.epochs),args.output_dir, history)
    # plot_lr(np.arange(0,args.epochs),args.output_dir, history)
    # plot_dice(np.arange(0,args.epochs),args.output_dir, history)
    # plot_dice(np.arange(0,args.epochs),args.output_dir, history)

if __name__ == '__main__':
    main(get_args_parser().parse_args())