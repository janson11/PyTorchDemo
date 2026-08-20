import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader


# 自定义CSV 数据集
class CSVDataset(Dataset):
    def __init__(self, csv_file):
        # 读取CSV文件
        self.data = pd.read_csv(csv_file)

    def __len__(self):
        # 返回数据大小
        return len(self.data)

    def __getitem__(self, index):
        # 使用 .iloc明确基于位置索引
        row = self.data.iloc[index]
        # 将特征和标签分开
        features = torch.tensor(row.iloc[:-1].to_numpy(), dtype=torch.float32)  # 特征
        label = torch.tensor(row.iloc[-1], dtype=torch.float32)  # 标签
        return features, label


# 实例化数据集和DataLoader
dateset = CSVDataset(csv_file='data/runoob_pytorch_data.csv')
dataloader = DataLoader(dateset, batch_size=4, shuffle=True)

# 遍历DataLoader
for features, label in dataloader:
    print("特征:", features)
    print("标签:", label)
    break
