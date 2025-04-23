
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from openTSNE import TSNE
from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler

# 读取CSV文件
data = pd.read_csv("/home/pci/luka_bhi_visul/visual_data/combined_finaldata.csv")
# 2. 数据预处理
features = data.drop(columns=['Activity_Type', 'Datetime'], errors='ignore')  # 特征列
labels = data['Activity_Type']  # 标签列

# 3. 数据标准化
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

# 4. 标签编码
label_encoder = LabelEncoder()
labels_encoded = label_encoder.fit_transform(labels)

# 5. 使用 openTSNE 进行降维，并调整 perplexity
tsne = TSNE(n_components=2, perplexity=10, random_state=42, n_jobs=-1)
tsne_results = tsne.fit(features_scaled)

# 6. 缩放 t-SNE 结果
tsne_df = pd.DataFrame(tsne_results, columns=['Dimension 1', 'Dimension 2'])
scaler = MinMaxScaler(feature_range=(-30, 30))  # 将结果限制在 [-30, 30] 范围内
tsne_df[['Dimension 1', 'Dimension 2']] = scaler.fit_transform(tsne_df[['Dimension 1', 'Dimension 2']])
tsne_df['Label'] = labels_encoded
tsne_df['Activity_Type'] = labels

# 7. 自定义颜色与标记
markers = ['o', 's', 'D', '^', 'v', '<', '>', 'p', 'h', 'x', '+', '*']  # 不同的标记形状
palette = sns.color_palette("tab20", len(tsne_df['Activity_Type'].unique()))  # 不同颜色

# 8. 绘制 t-SNE 图
plt.figure(figsize=(12, 8))
categories = tsne_df['Activity_Type'].unique()
for i, category in enumerate(categories):
    subset = tsne_df[tsne_df['Activity_Type'] == category]
    plt.scatter(
        subset['Dimension 1'],
        subset['Dimension 2'],
        label=category,
        alpha=0.8,
        s=50,
        c=[palette[i]],  # 设置颜色
        marker=markers[i % len(markers)]  # 设置标记
    )

# 9. 设置图表样式
plt.title("t-SNE Visualization of Activity Types (Compact)", fontsize=16)
plt.xlabel("Dimension 1", fontsize=12)
plt.ylabel("Dimension 2", fontsize=12)
plt.legend(
    title="Activity Types", 
    loc='center left', 
    bbox_to_anchor=(1, 0.5), 
    fontsize=10, 
    frameon=True
)
plt.grid(alpha=0.3)  # 添加网格线
plt.tight_layout()
plt.show()
