from sdv.metadata import Metadata
import pandas as pd
from sdv.single_table import TVAESynthesizer
# 读取数据
data = pd.read_csv('/home/pci/bhi_p/bhi_p/generated_dataset/dataset/real_numerical_data.csv')

# 创建通用的 Metadata 对象
metadata = Metadata()

# 假设你只有一个表，给它命名为 'table_name'
metadata.add_table('table_name')

# 获取数据框中的所有列，并添加到元数据中
for column in data.columns:
    metadata.add_column(column, table_name='table_name', sdtype='numerical')  # 'sdtype' 明确指定列的类型，移除 'type' 参数

# 查看元数据
print(metadata)
metadata.save_to_json('/home/pci/bhi_p/bhi_p/metadata.json')
tvae = TVAESynthesizer(
    metadata, # required
    enforce_min_max_values=True,
    enforce_rounding=False,
    epochs=2000,
    batch_size=512,
    verbose=True

)
tvae.fit(data)

# Generate synthetic data
synthetic_data = tvae.sample(len(data))

# Save to CSV
synthetic_data.to_csv('/home/pci/bhi_p/bhi_p/generated_dataset/tvae_generated/tvae_generated_data.csv', index=False)