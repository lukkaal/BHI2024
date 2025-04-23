import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.model_selection import train_test_split

# Step 1: Load and Preprocess Data
class PhysicalActivityDataset:
    def __init__(self, filepath):
        # Load the dataset
        df = pd.read_csv(filepath)

        # Extract numerical and categorical columns
        self.numerical_cols = ['Heart rate___beats/minute', 'Calories burned_kcal', 'Exercise duration_s',
                               'Sleep duration_minutes', 'Sleep type duration_minutes', 'Floors climbed___floors', 'Datetime']
        self.categorical_cols = ['Activity_Type', 'Code']

        # Handle missing values
        df[self.numerical_cols] = df[self.numerical_cols].fillna(0)  # Replace NaN with 0
        df['Datetime'] = pd.to_datetime(df['Datetime'], format='%Y=%m-%d %H:%M')
        df['Datetime'] = df['Datetime'].dt.hour  # Convert to hour for simplicity

        # Normalize numerical data
        self.scaler = MinMaxScaler()
        self.numerical_data = self.scaler.fit_transform(df[self.numerical_cols])

        # One-hot encode categorical data
        self.encoder = OneHotEncoder(sparse_output=False)
        self.categorical_data = self.encoder.fit_transform(df[self.categorical_cols])

        # Combine numerical and categorical data
        self.data = np.concatenate([self.numerical_data, self.categorical_data], axis=1)

    def get_data(self):
        return self.data

    def get_features(self):
        return self.data.shape[1]

# Load dataset
dataset = PhysicalActivityDataset('/home/pci/bhi_p/bhi_p/combined_data.csv')
data = dataset.get_data()

# Train-test split (optional)
X_train, X_test = train_test_split(data, test_size=0.2, random_state=42)

from tgan.model import TGANModel

# Step 2: Initialize and Train the TGAN Model
# Define TGAN model
tgan = TGANModel(continuous_columns=dataset.numerical_cols, categorical_columns=dataset.categorical_cols)

# Train the model on the training data
tgan.fit(X_train)

# Generate synthetic data
num_synthetic_samples = len(X_test)  # The number of synthetic samples to generate
synthetic_data = tgan.sample(num_synthetic_samples)

# Step 3: Post-Training Evaluation
# Reverse the normalization for numerical features
synthetic_numerical_data = synthetic_data[:, :len(dataset.numerical_cols)]
synthetic_numerical_data = dataset.scaler.inverse_transform(synthetic_numerical_data)

# Reverse the one-hot encoding for categorical features
synthetic_categorical_data = synthetic_data[:, len(dataset.numerical_cols):]
synthetic_categorical_data = dataset.encoder.inverse_transform(synthetic_categorical_data)

# Evaluate the synthetic data using similar metrics
real_numerical_data = dataset.numerical_data
real_numerical_data = dataset.scaler.inverse_transform(real_numerical_data)

# Evaluation metrics (Wasserstein Distance, KS Test, Jensen-Shannon Distance, Pairwise Correlation)
from scipy.stats import wasserstein_distance, ks_2samp, entropy
import numpy as np

# Wasserstein Distance
for i in range(real_numerical_data.shape[1]):
    w_distance = wasserstein_distance(real_numerical_data[:, i], synthetic_numerical_data[:, i])
    print(f"Wasserstein Distance for feature {dataset.numerical_cols[i]}: {w_distance:.4f}")

# KS Test
for i in range(real_numerical_data.shape[1]):
    ks_stat, p_value = ks_2samp(real_numerical_data[:, i], synthetic_numerical_data[:, i])
    print(f"KS Test for feature {dataset.numerical_cols[i]}: Statistic={ks_stat:.4f}, p-value={p_value:.4f}")

# Jensen-Shannon Distance
def jensen_shannon_distance(p, q):
    p_hist, _ = np.histogram(p, bins=100, density=True)
    q_hist, _ = np.histogram(q, bins=100, density=True)
    p_hist += 1e-10
    q_hist += 1e-10
    return entropy((p_hist + q_hist) / 2) - (entropy(p_hist) + entropy(q_hist)) / 2

for i in range(real_numerical_data.shape[1]):
    js_distance = jensen_shannon_distance(real_numerical_data[:, i], synthetic_numerical_data[:, i])
    print(f"Jensen-Shannon Distance for feature {dataset.numerical_cols[i]}: {js_distance:.4f}")

# Pairwise Correlation
real_corr = np.corrcoef(real_numerical_data, rowvar=False)
synthetic_corr = np.corrcoef(synthetic_numerical_data, rowvar=False)
pairwise_corr_distance = np.linalg.norm(real_corr - synthetic_corr)
print(f"Distance Pairwise Correlation: {pairwise_corr_distance:.4f}")
