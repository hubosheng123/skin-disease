import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.signal import savgol_filter
import json

# 加载训练历史数据
with open('output_dir/mymodel/2018/history-2018.json', 'r') as f:
    history = json.load(f)

# 提取并解包训练数据
train_iou = np.array(history['train_iou'])
train_background = train_iou[:, 0]  # 第一列为background IoU
train_target = train_iou[:, 1]      # 第二列为target IoU

# 数据平滑处理
window_size = 5
train_target_smooth = savgol_filter(train_target, window_size, 2)
train_background_smooth = savgol_filter(train_background, window_size, 2)

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

# 设置Y轴范围及刻度 (新增部分)
ax.set_ylim(0.0, 1.0)  # 强制Y轴范围为0~1
ax.yaxis.set_major_locator(plt.MultipleLocator(0.1))  # 主刻度间隔0.1
ax.yaxis.set_minor_locator(plt.MultipleLocator(0.05)) # 次刻度间隔0.05
ax.minorticks_on()  # 启用次刻度

# 绘制训练曲线（目标类）
ax.plot(train_target, color='tab:blue', alpha=0.15, linewidth=1)
ax.plot(train_target_smooth, color='tab:blue', linewidth=2, label='Target')

# 绘制训练曲线（背景类）
ax.plot(train_background, color='tab:orange', alpha=0.15, linewidth=1)
ax.plot(train_background_smooth, color='tab:orange', linewidth=2, label='Background')

# 图表装饰元素
ax.set_xlabel('Training Epoch', fontweight='bold')
ax.set_ylabel('IoU', fontweight='bold')  # 修正拼写错误
ax.set_title('Training IoU Progress', fontweight='bold', pad=15)
ax.legend(frameon=True, loc='lower right')
ax.grid(True, linestyle='--', alpha=0.6)
ax.set_xlim(0, len(train_target)-1)

# 设置X轴为整数刻度
ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))

# 保存并显示
plt.savefig('training_iou_curve_full_range.pdf', bbox_inches='tight')
plt.show()