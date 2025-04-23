import pandas as pd
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler, LabelEncoder

# 读取CSV文件
data = pd.read_csv("/home/pci/luka_bhi_visul/visual_data/combined_finaldata.csv")

# 数据预处理
label_encoder = LabelEncoder()
data['Activity_Type'] = label_encoder.fit_transform(data['Activity_Type'])

# 选择用于 t-SNE 的特征
features = data.drop(columns=['Datetime', 'Activity_Type'], errors='ignore')

# 对特征进行标准化
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

# 使用 t-SNE 对数据降维
tsne = TSNE(n_components=2, perplexity=40, random_state=42, n_iter=2000)
tsne_results = tsne.fit_transform(features_scaled)

# 将结果添加到 DataFrame 中
data['TSNE-1'] = tsne_results[:, 0]
data['TSNE-2'] = tsne_results[:, 1]

# 点形状的映射：为每个活动类型设置不同的点形状
markers = [ 's', 'D', '^', 'v', '<', '>', 'p', 'h', 'x', '+', '*']
unique_activity_types = data['Activity_Type'].unique()
# 定义自定义的柔和颜色（手动排除橙色等）
custom_palette = ['#4682B4', '#3CB371', '#FF8C00', '#8A2BE2', '#5F9EA0', '#1E90FF']


plt.figure(figsize=(10, 8))
for i, activity_type in enumerate(data['Activity_Type'].unique()):
    subset = data[data['Activity_Type'] == activity_type]
    plt.scatter(
        subset['TSNE-1'], 
        subset['TSNE-2'], 
        label=label_encoder.inverse_transform([activity_type])[0], 
        alpha=0.7,  # 调整透明度
        s=2,  # 点大小
        c=[custom_palette[i % len(custom_palette)]]  # 自定义颜色
    )


# 设置图例和标题
plt.title('t-SNE Visualization of Activity Types', fontsize=16)
plt.xlabel('t-SNE Component 1')
plt.ylabel('t-SNE Component 2')
plt.legend(title='Activity Type', loc='best', fontsize=10)
plt.grid(alpha=0.3)
plt.show()