import os

import numpy as np
from torch.utils.data import Dataset
from PIL import Image


class SegData(Dataset):
    def __init__(self, image_path, mask_path, data_transforms=None):
        self.image_path = image_path
        self.mask_path = mask_path

        self.images = os.listdir(self.image_path)
        self.masks = os.listdir(self.mask_path)
        self.transform = data_transforms

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image_filename = self.images[idx]

        # 判断文件格式
        if image_filename.endswith('.jpg'):
            # 原有 jpg 处理逻辑
            mask_filename = image_filename.replace('.jpg', '') + '_segmentation.png'
        elif image_filename.endswith('.bmp'):
            # bmp 格式处理逻辑
            mask_filename = image_filename.replace('.bmp', '') + '_lesion.bmp'
        else:
            raise ValueError(f"不支持的图像格式: {image_filename}")

        # 加载图像与 mask
        image = Image.open(os.path.join(self.image_path, image_filename)).convert('RGB')
        mask = Image.open(os.path.join(self.mask_path, mask_filename)).convert('L')

        # 应用 transform（如果有的话）
        if self.transform is not None:
            image, mask = self.transform(image, mask)

        return image, mask

