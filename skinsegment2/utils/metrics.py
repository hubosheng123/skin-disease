import numpy as np

class Evaluator(object):
    def __init__(self, num_class):
        self.num_class = num_class
        self.confusion_matrix = np.zeros((self.num_class,)*2)
    #计算像素准确率
    def Pixel_Accuracy(self):
        Acc = np.diag(self.confusion_matrix).sum() / self.confusion_matrix.sum()# diag是求主对角线元素
        return Acc

    def Pixel_Accuracy_Class(self): #计算平均召回率 所有正例中被预测出正例
        Acc = np.diag(self.confusion_matrix) / self.confusion_matrix.sum(axis=1) #每个类召回率
        Acc = np.nanmean(Acc)
        return Acc

    def Precision(self):#预测为正例的正确性 真正例/真正加 假正
        """计算每个类别的精确率（Precision）及其平均值"""
        precision_per_class = []
        for i in range(self.num_class):
            tp = self.confusion_matrix[i, i]
            fp = np.sum(self.confusion_matrix[:, i]) - tp  # 预测为i但真实非i的样本数
            denominator = tp + fp
            precision = tp / denominator if denominator != 0 else 0.0
            precision_per_class.append(precision)
        mean_precision = np.nanmean(precision_per_class)
        return precision_per_class, mean_precision

    def Dice(self):
        """计算每个类别的 Dice 系数及其平均值（与 F1 分数等价）"""
        dice_per_class = []
        for i in range(self.num_class):
            tp = self.confusion_matrix[i, i]
            fp = np.sum(self.confusion_matrix[:, i]) - tp  # 预测为i但真实非i的样本数
            fn = np.sum(self.confusion_matrix[i, :]) - tp  # 真实为i但预测非i的样本数
            denominator = 2 * tp + fp + fn
            dice = (2 * tp) / denominator if denominator != 0 else 0.0
            dice_per_class.append(dice)
        mean_dice = np.nanmean(dice_per_class)
        return dice_per_class, mean_dice
    #计算每一类IoU和MIoU
    def Mean_Intersection_over_Union(self):
        IoU = np.diag(self.confusion_matrix) / (
                    np.sum(self.confusion_matrix, axis=1) + np.sum(self.confusion_matrix, axis=0) -
                    np.diag(self.confusion_matrix))
        MIoU = np.nanmean(IoU)
        return IoU.tolist(),MIoU

    def Frequency_Weighted_Intersection_over_Union(self):#频率加权交并比（Frequency Weighted Intersection over Union, FWIoU）​，用于在语义分割任务中评估模型性能，尤其适用于类别不平衡数据（图片中背景占得部分多 而人少情况）
        freq = np.sum(self.confusion_matrix, axis=1) / np.sum(self.confusion_matrix)
        iu = np.diag(self.confusion_matrix) / (
                    np.sum(self.confusion_matrix, axis=1) + np.sum(self.confusion_matrix, axis=0) -
                    np.diag(self.confusion_matrix))

        FWIoU = (freq[freq > 0] * iu[freq > 0]).sum()
        return FWIoU

    def _generate_matrix(self, gt_image, pre_image):
        mask = (gt_image >= 0) & (gt_image < self.num_class)
        label = self.num_class * gt_image[mask].astype('int') + pre_image[mask]
        count = np.bincount(label, minlength=self.num_class**2)
        confusion_matrix = count.reshape(self.num_class, self.num_class)
        return confusion_matrix
    #加入数据
    def add_batch(self, gt_image, pre_image):
        assert gt_image.shape == pre_image.shape
        self.confusion_matrix += self._generate_matrix(gt_image, pre_image)
    #重置
    def reset(self):
        self.confusion_matrix = np.zeros((self.num_class,) * 2)
