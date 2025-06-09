import os
import torch
import time


from tqdm import tqdm
import matplotlib.pyplot as plt

from .EarlyStopping import EarlyStopping
from .metrics import Evaluator
import numpy as np
def train_and_val(epochs, model, train_loader, val_loader,criterion, optimizer,scheduler,output_dir,device,nb_classes):

    train_loss = []
    val_loss = []
    train_pix_acc = []
    val_pix_acc = []
    train_iou=[]
    val_iou=[]
    train_miou = []
    val_miou = []
    learning_rate = []
    best_miou = 0
    train_precision=[]
    val_precision=[]
    train_dice=[]
    val_dice=[]
    segmetric_train = Evaluator(nb_classes)
    segmetric_val = Evaluator(nb_classes)
    early_stopping = EarlyStopping(patience=20, delta=0.001)
    model.to(device)
    end_epochs=0
    need_time=0
    train_FWlou=[]
    val_FWlou=[]
    fit_time = time.time()

    for e in range(epochs):

        torch.cuda.empty_cache()
        segmetric_train.reset()
        segmetric_val.reset()

        since = time.time()
        training_loss = 0

        model.train()
        with tqdm(total=len(train_loader)) as pbar:
            for image, label in train_loader:

                image = image.to(device)


                label[label == 255] = 1
                label = label.to(device)
                output = model(image)

                loss = criterion(output, label)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                pred = output.data.cpu().numpy()
                label = label.cpu().numpy()
                pred = np.argmax(pred, axis=1)

                training_loss += loss.item()
                segmetric_train.add_batch(label, pred)
                pbar.update(1)

        model.eval()
        validation_loss = 0

        with torch.no_grad():
            with tqdm(total=len(val_loader)) as pb:
                for image, label in val_loader:

                    image = image.to(device)
                    label[label == 255] = 1
                    label = label.to(device)

                    output = model(image)
                    loss = criterion(output, label)

                    pred = output.data.cpu().numpy()
                    label = label.cpu().numpy()
                    pred = np.argmax(pred, axis=1)

                    validation_loss += loss.item()
                    segmetric_val.add_batch(label, pred)
                    pb.update(1)

        train_loss.append(training_loss / len(train_loader))
        val_loss.append(validation_loss / len(val_loader))

        train_pix_acc.append(segmetric_train.Pixel_Accuracy())
        val_pix_acc.append(segmetric_val.Pixel_Accuracy())
        train_iou.append(segmetric_train.Mean_Intersection_over_Union()[0])
        val_iou.append(segmetric_val.Mean_Intersection_over_Union()[0])
        train_miou.append(segmetric_train.Mean_Intersection_over_Union()[1])
        val_miou.append(segmetric_val.Mean_Intersection_over_Union()[1])

        train_FWlou.append(segmetric_train.Frequency_Weighted_Intersection_over_Union())
        val_FWlou.append(segmetric_val.Frequency_Weighted_Intersection_over_Union())
        learning_rate.append(scheduler.get_last_lr())

        train_precision.append(segmetric_train.Precision()[1])
        val_precision.append(segmetric_val.Precision()[1])

        train_dice.append(segmetric_train.Dice()[1])
        val_dice.append(segmetric_val.Dice()[1])

        torch.save(model.state_dict(), os.path.join(output_dir,'last.pth'))
        if best_miou < segmetric_val.Mean_Intersection_over_Union()[1]:
            best_miou=segmetric_val.Mean_Intersection_over_Union()[1]
            torch.save(model.state_dict(), os.path.join(output_dir,'miou_best.pth'))



        print("Epoch:{}/{}..".format(e + 1, epochs),
              "Train Pix Acc: {:.3f}".format(segmetric_train.Pixel_Accuracy()),
              "Val Pix Acc: {:.3f}".format(segmetric_val.Pixel_Accuracy()),
              "Train MIoU: {:.3f}".format(segmetric_train.Mean_Intersection_over_Union()[1]),
              "Val MIoU: {:.3f}".format(segmetric_val.Mean_Intersection_over_Union()[1]),
              "Train Loss: {:.3f}".format(training_loss / len(train_loader)),
              "Val Loss: {:.3f}".format(validation_loss / len(val_loader)),
              "Time: {:.2f}s".format((time.time() - since)))

        scheduler.step()
        early_stopping(np.mean(val_loss), model)
        end_epochs = epochs
        if early_stopping.early_stop:
            print(f"Early stopping at epoch {epochs}")

            break

    need_time = time.time() - fit_time
    history = {'train_iou':train_iou,'val_iou':val_iou,'train_loss': train_loss, 'val_loss': val_loss ,'train_pix_acc': train_pix_acc,
               'val_pix_acc': val_pix_acc,'train_miou': train_miou, 'val_miou': val_miou,'lr'
               :learning_rate,'train_precision':train_precision,'val_precision':val_precision,'train_dice':train_dice,
               'val_dice':val_dice,'train_FWlou':train_FWlou,'val_FWlou':val_FWlou,'need_time':need_time}
    print('Total time: {:.2f} m'.format((time.time() - fit_time) / 60))

    return history
# def plot_precision(x,output_dir, history):
#     plt.plot(x, history['train_precision'], label='train', marker='o')
#     plt.plot(x, history['val_precision'], label='val', marker='o')
#     plt.title('precision per epoch')
#     plt.ylabel('precision')
#     plt.xlabel('epoch')
#     plt.legend(), plt.grid()  # 添加图例 添加网格线
#     plt.savefig(os.path.join(output_dir, 'precision.png'))
#     plt.clf()
# def plot_dice(x,output_dir, history):
#     plt.plot(x, history['train_dice'], label='train', marker='o')
#     plt.plot(x, history['val_dice'], label='val', marker='o')
#     plt.title('dice per epoch')
#     plt.ylabel('dice')
#     plt.xlabel('epoch')
#     plt.legend(), plt.grid()  # 添加图例 添加网格线
#     plt.savefig(os.path.join(output_dir, 'dice.png'))
#     plt.clf()
# def plot_loss(x,output_dir, history):
#     plt.plot(x, history['val_loss'], label='val', marker='o')
#     plt.plot(x, history['train_loss'], label='train', marker='o')
#     plt.title('Loss per epoch')
#     plt.ylabel('loss')
#     plt.xlabel('epoch')
#     plt.legend(), plt.grid() #添加图例 添加网格线
#     plt.savefig(os.path.join(output_dir,'loss.png'))
#     plt.clf()
# def plot_pix_acc(x,output_dir, history):
#     plt.plot(x, history['train_pix_acc'], label='train_pix_acc', marker='x')
#     plt.plot(x, history['val_pix_acc'], label='val_pix_acc', marker='x')
#     plt.title('Pix Acc per epoch')
#     plt.ylabel('pixal accuracy')
#     plt.xlabel('epoch')
#     plt.legend(), plt.grid()
#     plt.savefig(os.path.join(output_dir,'pix_acc.png'))
#     plt.clf()
# def plot_miou(x,output_dir, history):
#     plt.plot(x, history['train_miou'], label='train_miou', marker='x')
#     plt.plot(x, history['val_miou'], label='val_miou', marker='x')
#     plt.title('MIoU per epoch')
#     plt.ylabel('miou')
#     plt.xlabel('epoch')
#     plt.legend(), plt.grid()
#     plt.savefig(os.path.join(output_dir,'miou.png'))
#     plt.clf()
# def plot_lr(x,output_dir,  history):
#     plt.plot(x, history['lr'], label='learning_rate', marker='x')
#     plt.title('learning rate per epoch')
#     plt.ylabel('Learning_rate')
#     plt.xlabel('epoch')
#     plt.legend(), plt.grid()
#     plt.savefig(os.path.join(output_dir,'learning_rate.png'))
#     plt.clf()
