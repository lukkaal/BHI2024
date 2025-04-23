import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# 读取CSV文件
data = pd.read_csv("/home/pci/luka_bhi_visul/visual_data/combined_finaldata.csv")

# 排除 Datetime 列并选择其他特征
columns_to_analyze = data.drop(columns=['Datetime'], errors='ignore').select_dtypes(
    include=['float64', 'int64']).columns

# 分组计算平均值、最大值和最小值
grouped = data.groupby('Activity_Type')[columns_to_analyze].agg(['mean', 'max', 'min'])

# 设置颜色调色板
colors = sns.color_palette("deep", len(columns_to_analyze))

# 创建图表
fig, ax = plt.subplots(figsize=(12, 8))

# 设置每组柱子的位置和宽度
activity_types = grouped.index
x = np.arange(len(activity_types))  # Activity_Type 位置
bar_width = 0.1  # 每个柱子的宽度

# 绘制柱状图
for i, feature in enumerate(columns_to_analyze):
    means = grouped[(feature, 'mean')]
    max_vals = grouped[(feature, 'max')]
    min_vals = grouped[(feature, 'min')]

    # 绘制柱状图
    ax.bar(x + i * bar_width, means, width=bar_width, label=feature, color=colors[i], alpha=0.8)

    # 绘制最大值和最小值标线
    ax.vlines(x + i * bar_width, ymin=min_vals, ymax=max_vals, color='black', linewidth=1)

# 添加标题和轴标签
ax.set_title("Feature Analysis by Activity Type", fontsize=16, fontweight='bold')
ax.set_xlabel("Activity Type", fontsize=14, fontweight='bold')
ax.set_ylabel("Values", fontsize=14, fontweight='bold')
ax.set_xticks(x + bar_width * (len(columns_to_analyze) - 1) / 2)
ax.set_xticklabels(activity_types, rotation=45, ha='right', fontsize=12)

# 添加图例
ax.legend(title="Features", bbox_to_anchor=(1.05, 1), loc='upper left')

# 调整布局
plt.tight_layout()
plt.show()
