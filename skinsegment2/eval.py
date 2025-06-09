import argparse
from utils.transform import Resize,Compose,ToTensor,Normalize,RandomHorizontalFlip
from utils.datasets import SegData
import torch
import os
import numpy as np
from tqdm import tqdm
from model.Unet import Unet
from model.TinyUNet import TinyUNet
from model.Deeplab import DeepLabV3
from model.SkinUnet import SkinUNet
from model.Transunet import TransUNet
from utils.metrics import Evaluator
ph2_path=r'E:\PH2 Dataset\PH2Dataset\PH2 Dataset images'
def get_args_parser():
    parser = argparse.ArgumentParser('Eval Model', add_help=False)
    parser.add_argument('--batch_size', default=1, type=int,help='Batch size for training')
    parser.add_argument('--input_size', default=[224,224],nargs='+',type=int,help='images input size')
    parser.add_argument('--data_path', default=ph2_path, type=str,help='dataset path')
    parser.add_argument('--weights', default='./trained_model/last_trans.pth', type=str,help='dataset path')
    parser.add_argument('--nb_classes', default=2, type=int,help='number of the classification types')
    parser.add_argument('--device', default='cuda',help='device to use for training / testing')
    parser.add_argument('--num_workers', default=4, type=int)

    return parser
def main(args):

    device = torch.device(args.device)

    segmetric = Evaluator(args.nb_classes)
    segmetric.reset()

    test_transform = Compose([
                                    Resize(args.input_size),
                                    ToTensor(),
                                    Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                            ])

    test_dataset = SegData(image_path=os.path.join(args.data_path, 'images/Test_Input'),
                            mask_path=os.path.join(args.data_path, 'labels/Test_label'),
                            data_transforms=test_transform)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=args.batch_size, shuffle=False,
                                             num_workers=args.num_workers)

    #model =SkinUNet(in_channels=3,num_classes=args.nb_classes)
    #model = DeepLabV3(num_classes=2)
    model = TransUNet(img_dim=224,
                      in_channels=3,
                      out_channels=128,
                      head_num=4,
                      mlp_dim=512,
                      block_num=8,
                      patch_dim=16,
                      class_num=2)
    checkpoint = torch.load(args.weights, map_location='cpu')
    msg = model.load_state_dict(checkpoint, strict=True)
    print(msg)

    model.to(device)
    model.eval()

    classes = ["background","skin disease area"]

    with torch.no_grad():
        with tqdm(total=len(test_loader)) as pbar:
            for image, label in test_loader:
                label[label==255] = 1
                output = model(image.to(device))
                pred = output.data.cpu().numpy()
                label = label.cpu().numpy()
                pred = np.argmax(pred, axis=1)
                segmetric.add_batch(label, pred)
                pbar.update(1)

    pix_acc = segmetric.Pixel_Accuracy()
    every_iou,miou = segmetric.Mean_Intersection_over_Union()
    dice_per_class, mean_dice  =segmetric.Dice()
    precision_per_class, mean_precision=segmetric.Precision()
    class_recall  =segmetric.Pixel_Accuracy_Class()
    FWIoU  = segmetric.Frequency_Weighted_Intersection_over_Union()#加权交并比
    IoU, MIoU = segmetric.Mean_Intersection_over_Union()
    print("Pixel Accuracy is :", pix_acc)
    print('Dice(F1) is :', dice_per_class,'AVG is :', mean_dice)
    print('Precision is :', precision_per_class,'AVG is :',mean_precision)
    print('avg_recall is :', class_recall)
    print('FWIoU is :', FWIoU)
    print("==========Every IOU==========")
    for name,prob in zip(classes,every_iou):
        print(name+" : "+str(prob))
    print("=============================")
    print("MiOU is :", miou)
if __name__ == '__main__':
    #获取训练参数
    args = get_args_parser()
    #解析训练参数
    args = args.parse_args()
    #训练参数传入主函数
    main(args)
