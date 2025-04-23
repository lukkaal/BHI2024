import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from scipy.stats import wasserstein_distance, ks_2samp, entropy
from sklearn.metrics import pairwise_distances
from tqdm import tqdm

import seaborn as sns
import matplotlib.pyplot as plt


numerical_cols = ['Heart rate___beats/minute', 'Calories burned_kcal', 'Exercise duration_s',
                               'Sleep duration_minutes', 'Sleep type duration_minutes', 'Floors climbed___floors',
                               'Datetime']

# Step 1: Load and Preprocess Data
class PhysicalActivityDataset(Dataset):
    def __init__(self, filepath):
        # Load the dataset
        df = pd.read_csv(filepath)

        # Extract numerical features and categorical features
        self.numerical_cols = ['Heart rate___beats/minute', 'Calories burned_kcal', 'Exercise duration_s',
                               'Sleep duration_minutes', 'Sleep type duration_minutes', 'Floors climbed___floors',
                               'Datetime']
        self.categorical_cols = ['Activity_Type', 'Code']

        # Handle missing values
        df[self.numerical_cols] = df[self.numerical_cols].fillna(0)  # Replace NaN with 0
        # df[numerical_cols] = data[numerical_cols].fillna(0)  # Replace NaN with 0
        df['Datetime'] = pd.to_datetime(df['Datetime'], format='%Y=%m-%d %H:%M')
        df['Datetime'] = df['Datetime'].dt.hour

        # Normalize numerical data
        self.scaler = MinMaxScaler()
        self.numerical_data = self.scaler.fit_transform(df[self.numerical_cols])

        # One-hot encode categorical data
        self.encoder = OneHotEncoder(sparse_output=False)
        self.categorical_data = self.encoder.fit_transform(df[self.categorical_cols])

        # Create combined data
        self.data = np.concatenate([self.numerical_data, self.categorical_data], axis=1)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


# Load the dataset
dataset = PhysicalActivityDataset('/home/pci/bhi_p/combined_data.csv')

data_loader = DataLoader(dataset, batch_size=512, shuffle=True)


# Step 2: Define Generator and Discriminator Networks
class Generator(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, output_dim),
            nn.Sigmoid(),
        )

    def forward(self, noise, condition):
        x = torch.cat((noise, condition), dim=1)
        return self.model(x)


class Discriminator(nn.Module):
    def __init__(self, input_dim):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.4),
            nn.Linear(1024, 512),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.4),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 1),
            # No activation for WGAN-GP
        )

    def forward(self, data):
        return self.model(data)


# Step 3: Training Setup with Gradient Penalty
def gradient_penalty(D, real_data, fake_data, device):
    alpha = torch.rand(real_data.size(0), 1).to(device)
    alpha = alpha.expand_as(real_data)
    interpolates = alpha * real_data + (1 - alpha) * fake_data
    interpolates = interpolates.to(device)
    interpolates.requires_grad_(True)

    d_interpolates = D(interpolates)
    gradients = torch.autograd.grad(outputs=d_interpolates, inputs=interpolates,
                                    grad_outputs=torch.ones(d_interpolates.size()).to(device),
                                    create_graph=True, retain_graph=True, only_inputs=True)[0]
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty


# Initialize networks and optimizers
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
noise_dim = 128
num_conditions = dataset.categorical_data.shape[1]
num_features = dataset.data.shape[1]

G = Generator(input_dim=noise_dim + num_conditions, output_dim=num_features).to(device)
D = Discriminator(input_dim=num_features).to(device)

lambda_gp = 15
g_optimizer = optim.Adam(G.parameters(), lr=0.00005, betas=(0.5, 0.9))
d_optimizer = optim.Adam(D.parameters(), lr=0.0001, betas=(0.5, 0.9))

# Step 4: Train the WGAN-GP
num_epochs = 3000
g_loss_list = []
d_loss_list = []
n_critic = 5

for epoch in range(num_epochs):
    g_epoch_loss = 0.0
    d_epoch_loss = 0.0
    for i, batch in enumerate(tqdm(data_loader, desc=f"Epoch [{epoch + 1}/{num_epochs}]")):
        batch = batch.to(device).float()
        batch_size = batch.size(0)

        # Split numerical and categorical features
        numerical_data = batch[:, :len(dataset.numerical_cols)]
        categorical_data = batch[:, len(dataset.numerical_cols):]

        # Train Discriminator
        for _ in range(n_critic):
            d_optimizer.zero_grad()
            noise = torch.randn(batch_size, noise_dim).to(device)
            condition = categorical_data
            fake_data = G(noise, condition).detach()

            real_outputs = D(batch)
            fake_outputs = D(fake_data)

            d_real_loss = -torch.mean(real_outputs)
            d_fake_loss = torch.mean(fake_outputs)
            gp = gradient_penalty(D, batch.data, fake_data.data, device)
            d_loss = d_real_loss + d_fake_loss + lambda_gp * gp
            d_loss.backward()
            d_optimizer.step()

            d_epoch_loss += d_loss.item()

        # Train Generator
        g_optimizer.zero_grad()
        noise = torch.randn(batch_size, noise_dim).to(device)
        fake_data = G(noise, condition)
        fake_outputs = D(fake_data)
        g_loss = -torch.mean(fake_outputs)
        g_loss.backward()
        g_optimizer.step()

        g_epoch_loss += g_loss.item()

    g_loss_list.append(g_epoch_loss / len(data_loader))
    d_loss_list.append(d_epoch_loss / (len(data_loader) * n_critic))
    print(f"Epoch [{epoch + 1}/{num_epochs}] | D Loss: {d_loss_list[-1]:.4f} | G Loss: {g_loss_list[-1]:.4f}")

# Save the trained model
torch.save(G.state_dict(), 'generator.pth')
torch.save(D.state_dict(), 'discriminator.pth')

# Step 5: Post-Training Evaluation
G.eval()
with torch.no_grad():
    num_synthetic_samples = len(dataset)
    noise = torch.randn(num_synthetic_samples, noise_dim).to(device)
    condition = torch.from_numpy(dataset.categorical_data).to(device).float()
    synthetic_data = G(noise, condition).cpu().numpy()

# Reverse normalization for numerical features
synthetic_numerical_data = synthetic_data[:, :len(dataset.numerical_cols)]
syn = synthetic_numerical_data
synthetic_numerical_data = dataset.scaler.inverse_transform(synthetic_numerical_data)

real_numerical_data = dataset.numerical_data
real = real_numerical_data
real_numerical_data = dataset.scaler.inverse_transform(real_numerical_data)

# Evaluation metrics
#for i in range(real_numerical_data.shape[1]):
#    w_distance = wasserstein_distance(real_numerical_data[:, i], synthetic_numerical_data[:, i])
#    print(f"Wasserstein Distance for feature {dataset.numerical_cols[i]}: {w_distance:.4f}")

for i in range(real_numerical_data.shape[1]):
    w_distance = wasserstein_distance(real[:, i], syn[:, i])
    print(f"Wasserstein Distance for feature {dataset.numerical_cols[i]}: {w_distance:.4f}")

for i in range(real_numerical_data.shape[1]):
    ks_stat, p_value = ks_2samp(real_numerical_data[:, i], synthetic_numerical_data[:, i])
    print(f"KS Test for feature {dataset.numerical_cols[i]}: Statistic={ks_stat:.4f}, p-value={p_value:.4f}")


def jensen_shannon_distance(p, q):
    p_hist, _ = np.histogram(p, bins=100, density=True)
    q_hist, _ = np.histogram(q, bins=100, density=True)
    p_hist += 1e-10
    q_hist += 1e-10
    return entropy((p_hist + q_hist) / 2) - (entropy(p_hist) + entropy(q_hist)) / 2


for i in range(real_numerical_data.shape[1]):
    js_distance = jensen_shannon_distance(real_numerical_data[:, i], synthetic_numerical_data[:, i])
    print(f"Jensen-Shannon Distance for feature {dataset.numerical_cols[i]}: {js_distance:.4f}")

# 4. Distance Pairwise Correlation
real_corr = np.corrcoef(real_numerical_data, rowvar=False)
synthetic_corr = np.corrcoef(synthetic_numerical_data, rowvar=False)
pairwise_corr_distance = np.linalg.norm(real_corr - synthetic_corr)
print(f"Distance Pairwise Correlation: {pairwise_corr_distance:.4f}")

print(real_numerical_data.shape)
print(synthetic_numerical_data.shape)

real_numerical_data = pd.DataFrame(real_numerical_data, columns=numerical_cols)
real_numerical_data.to_csv('/home/pci/bhi_p/bhi_p/generated_dataset/dataset/real_numerical_data.csv', index=False)
synthetic_numerical_data = pd.DataFrame(synthetic_numerical_data, columns=numerical_cols)
synthetic_numerical_data.to_csv('/home/pci/bhi_p/bhi_p/generated_dataset/dataset/synthetic_numerical_data.csv', index=False)


for column in numerical_cols:
    plt.figure(figsize=(10, 6))
    sns.histplot(real_numerical_data[column], kde=True, color="blue", label="Original Data", stat="density")
    sns.histplot(synthetic_numerical_data[column], kde=True, color="orange", label="Synthetic Data", stat="density")
    plt.title(f"Distribution Comparison for {column}")
    plt.legend()
    plt.show()
