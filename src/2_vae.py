import pandas as pd
import torch
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import torch.nn.functional as F
from tqdm import tqdm  # 导入 tqdm 库

# 加载数据
data = pd.read_csv('/home/pci/bhi_p/vae_generated/combined_finaldata.csv')

# 假设数值型特征和分类特征是已知的，手动指定列
numerical_columns = ['Heart rate___beats/minute', 'Calories burned_kcal', 'Exercise duration_s',
        'Sleep duration_minutes', 'Sleep type duration_minutes', 'Floors climbed___floors', 'Datetime']  # 示例数值型列
categorical_columns = ['Activity_Type']  # 示例分类列

# 对数值型特征进行归一化
scaler = MinMaxScaler()
data[numerical_columns] = scaler.fit_transform(data[numerical_columns])

# 对分类特征进行独热编码
data = pd.get_dummies(data, columns=categorical_columns)

# 打印数据的列名，查看独热编码后的列
print(data.columns)

# 分离特征和标签
# 假设你需要移除的是所有与 Activity_Type 相关的列，作为特征 X
X = data.drop(columns=['Activity_Type_Floors Climbed', 'Activity_Type_Light Sleep',
                       'Activity_Type_No Physical Activity', 'Activity_Type_REM Sleep',
                       'Activity_Type_Running', 'Activity_Type_Walking'])
print(X.columns)

# 标签 y 包含所有独热编码的目标列
y = data[['Activity_Type_Floors Climbed', 'Activity_Type_Light Sleep',
          'Activity_Type_No Physical Activity', 'Activity_Type_REM Sleep',
          'Activity_Type_Running', 'Activity_Type_Walking']]

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(X_train.dtypes)

# 强制转换所有特征列为 float32 类型
X_train = X_train.astype(np.float32)
X_test = X_test.astype(np.float32)

y_train = y_train.astype(np.float32)
y_test = y_test.astype(np.float32)

# 转为 PyTorch 张量
X_train = torch.tensor(X_train.values, dtype=torch.float32)
X_test = torch.tensor(X_test.values, dtype=torch.float32)
y_train = torch.tensor(y_train.values, dtype=torch.float32)
y_test = torch.tensor(y_test.values, dtype=torch.float32)

# 打印数据形状以检查
print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)
print("y_train shape:", y_train.shape)
print("y_test shape:", y_test.shape)

import torch
import torch.nn as nn
import torch.nn.functional as F

# VAE编码器
class Encoder(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super(Encoder, self).__init__()
        self.fc1 = nn.Linear(input_dim, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc_mean = nn.Linear(128, latent_dim)  # 均值
        self.fc_log_var = nn.Linear(128, latent_dim)  # 方差对数
        self.bn1 = nn.BatchNorm1d(512)  # 批归一化
        self.bn2 = nn.BatchNorm1d(256)  # 批归一化
        self.bn3 = nn.BatchNorm1d(128)  # 批归一化
        self.dropout = nn.Dropout(0.3)  # Dropout 防止过拟合

    def forward(self, x):
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.fc2(x)))
        x = F.relu(self.bn3(self.fc3(x)))
        x = self.dropout(x)  # Dropout
        mean = self.fc_mean(x)
        log_var = self.fc_log_var(x)
        return mean, log_var

# VAE解码器
class Decoder(nn.Module):
    def __init__(self, latent_dim, output_dim):
        super(Decoder, self).__init__()
        self.fc1 = nn.Linear(latent_dim, 128)
        self.fc2 = nn.Linear(128, 256)
        self.fc3 = nn.Linear(256, 512)
        self.fc4 = nn.Linear(512, output_dim)
        self.bn1 = nn.BatchNorm1d(128)  # 批归一化
        self.bn2 = nn.BatchNorm1d(256)  # 批归一化
        self.bn3 = nn.BatchNorm1d(512)  # 批归一化
        self.dropout = nn.Dropout(0.3)  # Dropout 防止过拟合

    def forward(self, z):
        z = F.relu(self.bn1(self.fc1(z)))
        z = F.relu(self.bn2(self.fc2(z)))
        z = F.relu(self.bn3(self.fc3(z)))
        z = self.dropout(z)  # Dropout
        return torch.sigmoid(self.fc4(z))  # 输出归一化到 [0, 1]

# VAE模型
class VAE(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super(VAE, self).__init__()
        self.encoder = Encoder(input_dim, latent_dim)
        self.decoder = Decoder(latent_dim, input_dim)

    def reparameterize(self, mean, log_var):
        std = torch.exp(0.5 * log_var)  # 计算标准差
        epsilon = torch.randn_like(std)  # 从标准正态分布采样
        return mean + epsilon * std  # 重参数化

    def forward(self, x):
        mean, log_var = self.encoder(x)
        z = self.reparameterize(mean, log_var)
        reconstructed = self.decoder(z)
        return reconstructed, mean, log_var


#def vae_loss(reconstructed, original, mean, log_var):
    # 重构损失（均方误差）
#    reconstruction_loss = F.mse_loss(reconstructed, original, reduction='mean')
    # KL散度
#    kl_divergence = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())
#    return reconstruction_loss + kl_divergence / original.size(0)
def vae_loss(reconstructed, original, mean, log_var, kl_weight=1.0):
    reconstruction_loss = F.mse_loss(reconstructed, original, reduction='mean')
    kl_divergence = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())
    return reconstruction_loss + kl_weight * kl_divergence / original.size(0)

from torch.optim import Adam

# 参数
input_dim = X_train.shape[1]
latent_dim = 64
batch_size = 128
epochs = 1000
learning_rate = 1e-3

# 检查是否使用 GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# 初始化模型和优化器
vae = VAE(input_dim, latent_dim).to(device)  # 将模型转移到 GPU
optimizer = Adam(vae.parameters(), lr=learning_rate)

# 数据加载
train_loader = torch.utils.data.DataLoader(X_train, batch_size=batch_size, shuffle=True)

# 训练循环
for epoch in range(epochs):
    vae.train()
    total_loss = 0
    # 使用 tqdm 进度条
    for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}", unit="batch"):
        batch = batch.to(device)  # 将当前批次转移到 GPU
        optimizer.zero_grad()
        reconstructed, mean, log_var = vae(batch)
        loss = vae_loss(reconstructed, batch, mean, log_var)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch + 1}, Loss: {total_loss / len(train_loader):.4f}")

# 生成新数据
vae.eval()  # 切换到评估模式
num_samples = 100  # 生成样本数量
z_samples = torch.randn(num_samples, latent_dim).to(device)  # GPU 上生成潜在向量
generated_data = vae.decoder(z_samples).detach().cpu().numpy()  # 解码并转回 CPU

# 反归一化
generated_data = scaler.inverse_transform(generated_data)

# 转为DataFrame保存
generated_df = pd.DataFrame(generated_data, columns=X.columns)
generated_df.to_csv('/home/pci/bhi_p/vae_generated/generated_table.csv', index=False)

print("done")
