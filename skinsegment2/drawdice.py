import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.signal import savgol_filter
import json

# 加载训练历史数据
with open('output_dir/mymodel/2018/history-2018.json', 'r') as f:
    history = json.load(f)

# 提取并处理数据
train_dice = np.array(history['train_dice'])
val_dice = np.array(history['val_dice'])  # 新增验证集数据
epochs = np.arange(1, len(train_dice) + 1)

# 数据平滑处理
window_size = 5
train_smooth = savgol_filter(train_dice, window_size, 2)
val_smooth = savgol_filter(val_dice, window_size, 2)

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

# 绘制训练集曲线（绿色实线）
ax.plot(epochs, train_dice, color='tab:green', alpha=0.15, linewidth=1)
ax.plot(epochs, train_smooth, color='tab:green', linewidth=2, label='Train Dice')

# 绘制验证集曲线（橙色虚线）
ax.plot(epochs, val_dice, color='tab:orange', alpha=0.15, linewidth=1, linestyle='--')
ax.plot(epochs, val_smooth, color='tab:orange', linewidth=2, linestyle='--', label='Val Dice')

# 配置坐标轴
ax.set_xlabel('Training Epoch', fontweight='bold')
ax.set_ylabel('Dice Coefficient', fontweight='bold')
ax.set_title('Training & Validation Dice Progress', fontweight='bold', pad=15)
ax.legend(frameon=True, loc='lower right', ncol=2)  # 双列图例

# 设置刻度系统
ax.set_xlim(1, len(epochs))
ax.set_ylim(0.0, 1.0)
ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
ax.yaxis.set_major_locator(plt.MultipleLocator(0.1))
ax.yaxis.set_minor_locator(plt.MultipleLocator(0.05))
ax.minorticks_on()

# 网格线
ax.grid(True, linestyle='--', alpha=0.6)

# 保存并显示
plt.savefig('train_val_dice_curve.pdf', bbox_inches='tight')
plt.show()