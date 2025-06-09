import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.signal import savgol_filter
import json

# 加载训练历史数据
with open('output_dir/mymodel/2018/history-2018.json', 'r') as f:
    history = json.load(f)

# 提取全局像素准确率
train_acc = np.array(history['train_pix_acc'])
val_acc = np.array(history['val_pix_acc'])  # 新增验证集准确率
epochs = np.arange(1, len(train_acc) + 1)

# 数据平滑处理
window_size = 5
train_smooth = savgol_filter(train_acc, window_size, 2)
val_smooth = savgol_filter(val_acc, window_size, 2)

# 配置学术图表风格
sns.set_style("whitegrid")
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman'],
    'axes.labelsize': 14,
    'axes.titlesize': 16,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'figure.dpi': 300
})

# 创建图表对象
fig, ax = plt.subplots(figsize=(8, 5))

# 绘制双曲线（训练集 + 验证集）
ax.plot(epochs, train_smooth, color='#E74C3C', linewidth=2, label='Train')  # 红色
ax.plot(epochs, val_smooth, color='#2980B9', linewidth=2, linestyle='--', label='Val')  # 蓝色虚线

# 配置坐标轴
ax.set_xlabel('Training Epoch', fontweight='bold')
ax.set_ylabel('Pixel Accuracy', fontweight='bold')
ax.set_title('Training & Validation Accuracy Progress', fontweight='bold', pad=15)
ax.legend(frameon=True, loc='lower right', borderpad=0.8)

# 设置刻度系统
ax.set_xlim(1, len(epochs))
ax.set_ylim(0.5,1)  # 根据实际数据调整
ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
ax.yaxis.set_major_locator(plt.MultipleLocator(0.05))
ax.yaxis.set_minor_locator(plt.MultipleLocator(0.01))
ax.minorticks_on()

# 增强网格显示
ax.grid(which='major', linestyle='--', alpha=0.7)
ax.grid(which='minor', linestyle=':', alpha=0.4)

# 保存输出
plt.savefig('train_val_pix_acc_curve.pdf', bbox_inches='tight', dpi=300)
plt.show()