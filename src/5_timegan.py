from sdv.single_table import TVAESynthesizer
from sdv.metadata import Metadata
import pandas as pd

# Load dataset
data = pd.read_csv('/home/pci/bhi_p/bhi_p/generated_dataset/dataset/real_numerical_data.csv')
metadata = Metadata()

# 获取数据框中的所有列
for column in data.columns:
    metadata.add_column(column)  # 'numerical' 表示连续列

print(metadata)

# Initialize and train the TVAE model
tvae = TVAESynthesizer(
    metadata, # required
    enforce_min_max_values=True,
    enforce_rounding=False,
    epochs=1000,
    batch_size=512,
)
tvae.fit(data)

# Generate synthetic data
synthetic_data = tvae.sample(len(data))

# Save to CSV
synthetic_data.to_csv('generated_data.csv', index=False)