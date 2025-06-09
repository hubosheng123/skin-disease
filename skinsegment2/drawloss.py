import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.signal import savgol_filter
import json

# 加载训练历史数据
with open('output_dir/mymodel/2018/history-2018.json', 'r') as f:
    history = json.load(f)

# 提取训练损失数据
train_loss = np.array(history['train_loss'])
epochs = np.arange(1, len(train_loss) + 1)

# 动态窗口平滑处理
window_size = min(11, len(train_loss)//2 * 2-1)  # 自适应窗口大小
if window_size >= 3:
    loss_smooth = savgol_filter(train_loss, window_size, 2)
else:
    loss_smooth = train_loss

# 配置紫色系学术风格
sns.set_style("whitegrid")
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman'],
    'axes.prop_cycle': plt.cycler('color', ['#9B59B6', '#6C3483']),  # 紫色系配色
    'axes.labelsize': 14,
    'axes.titlesize': 16,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'figure.dpi': 300,
    'axes.grid.axis': 'y'
})

# 创建图表对象
fig, ax = plt.subplots(figsize=(8, 5))

# 绘制双紫色渐变曲线

ax.plot(epochs, loss_smooth, color='#6C3483', linewidth=2.5, label='Train_Loss')

# 配置坐标轴系统
ax.set_xlabel('Training Epoch', fontweight='bold')
ax.set_ylabel('Loss Value', fontweight='bold')
ax.set_title('Training Loss Progression', fontweight='bold', pad=15)

# 智能坐标轴范围
loss_max = np.max(train_loss) * 1.15 if len(train_loss) > 0 else 1.0
ax.set(xlim=(1, len(epochs)), ylim=(0, loss_max))
ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
ax.yaxis.set_major_formatter(plt.FormatStrFormatter('%.2f'))  # 两位小数精度

# 增强图例显示
legend = ax.legend(frameon=True, loc='upper right',
                  handlelength=1.5, edgecolor='#4A235A')
legend.get_frame().set_alpha(0.9)

# 高级网格配置
ax.grid(which='major', linestyle='--', linewidth=0.7, alpha=0.8)
ax.grid(which='minor', linestyle=':', linewidth=0.5, alpha=0.5)
ax.minorticks_on()

# 输出高品质图表
plt.savefig('training_loss_purple.pdf', bbox_inches='tight', dpi=300)
plt.show()