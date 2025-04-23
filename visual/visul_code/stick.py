import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 读取CSV文件
#data = pd.read_csv("/home/pci/luka_bhi_visul/visual_data/combined_finaldata.csv")

# 读取CSV文件
data = pd.read_csv("/home/pci/luka_bhi_visul/visual_data/combined_finaldata.csv")

# 排除 Datetime 列，选择其余的数值列
numerical_columns = data.drop(columns=['Datetime'], errors='ignore').select_dtypes(include=['float64', 'int64']).columns

# 设置子图的行列布局
num_columns = len(numerical_columns)
ncols = 2  # 每行显示3个子图
nrows = (num_columns + ncols - 1) // ncols  # 根据列数计算需要多少行

# 定义深色调调色板
colors = sns.color_palette("deep", n_colors=len(numerical_columns))

# 设置 Seaborn 样式
sns.set_theme(style="whitegrid")

# 创建子图
fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(15, nrows * 4))
axes = axes.flatten()  # 展平子图数组，便于迭代

# 遍历每个数值列，绘制直方图
for i, col in enumerate(numerical_columns):
    ax = axes[i]
    sns.histplot(data[col].dropna(), bins=20, kde=False, color=colors[i], ax=ax, edgecolor='black', linewidth=1.2)
    
    # 添加均值和中位数线
    mean = data[col].mean()
    median = data[col].median()
    ax.axvline(mean, color='red', linestyle='--', label=f'Mean: {mean:.2f}')
    ax.axvline(median, color='green', linestyle='--', label=f'Median: {median:.2f}')
    ax.legend(loc='upper right', fontsize=10)

    # 设置标题和标签
    ax.set_title(f'Distribution of {col}', fontsize=14, fontweight='bold')
    ax.set_xlabel('Value', fontsize=12, fontweight='bold')
    ax.set_ylabel('Amounts', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.5)

# 如果子图比数值列多，隐藏多余的子图
for j in range(len(numerical_columns), len(axes)):
    axes[j].axis('off')

# 添加整体标题
fig.suptitle('Feature Distributions in the Dataset', fontsize=16, fontweight='bold')

# 调整布局
plt.tight_layout(pad=2.0)
plt.show()
