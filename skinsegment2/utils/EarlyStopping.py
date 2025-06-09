import torch
class EarlyStopping:
    def __init__(self, patience=20, delta=0, path='./output_dir/minloss_Best.pth'):
        self.patience = patience  # 容忍未改善的epoch次数
        self.delta = delta  # 最小改进阈值
        self.counter = 0  # 未改善计数器
        self.best_score = None  # 最佳指标值
        self.early_stop = False  # 停止标志
        self.path = path  # 最佳模型保存路径

    def __call__(self, val_loss, model):
        score = -val_loss  # 转换为最大化问题
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(model)
        elif score < self.best_score + self.delta:  # 未达改进阈值
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True  # 触发停止
        else:
            self.best_score = score
            self.save_checkpoint(model)
            self.counter = 0  # 重置计数器

    def save_checkpoint(self, model):
        torch.save(model.state_dict(), self.path)