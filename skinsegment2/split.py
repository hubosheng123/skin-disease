import os
import shutil
from tqdm import tqdm

def process_split(list_file, image_dest, label_dest=None):
    """处理数据集分割文件，支持无标签的测试集"""
    with open(list_file, 'r') as f:
        lines = f.readlines()

    for line in lines:
        parts = line.strip().split(' ')
        img_path = parts[0]
        label_path = parts[1] if len(parts) > 1 else None  # 测试集可能无标签[1](@ref)

        # 复制图像
        img_filename = os.path.basename(img_path)
        dest_img = os.path.join(image_dest, img_filename)
        os.makedirs(os.path.dirname(dest_img), exist_ok=True)
        shutil.copy(img_path, dest_img)

        # 复制标签（仅当存在且目标路径非空）
        if label_path and label_dest:
            label_filename = os.path.basename(label_path)
            dest_label = os.path.join(label_dest, label_filename)
            os.makedirs(os.path.dirname(dest_label), exist_ok=True)
            shutil.copy(label_path, dest_label)


# 定义基础路径
base_dir = os.getcwd()

# 处理训练集和验证集（含标签）
for split in ['train', 'val']:
    process_split(
        list_file=f"{split}_list.txt",
        image_dest=os.path.join(base_dir, 'images', split),
        label_dest=os.path.join(base_dir, 'labels', split)
    )

# 处理测试集（仅图像）
process_split(
    list_file="test_list.txt",
    image_dest=os.path.join(base_dir, 'test', 'images'),
    label_dest=None  # 显式禁用标签复制
)

print("数据集划分完成！")