import torch

from torch.utils.data import Dataset
from torch.utils.data import DataLoader


# 自定义数据集
class MyDataset(Dataset):
    def __init__(self, data, labels):
        # 数据初始化
        self.data = data
        self.labels = labels

    def __len__(self):
        # 返回数据集大小
        return len(self.data)

    def __getitem__(self, idx):
        # 按索引返回数据和标签
        sample = self.data[idx]
        label = self.labels[idx]
        return sample, label


# 生成示例数据
data = torch.rand(100, 5)  # 100个样本，每个样本有5个特征
lables = torch.randint(0, 2, (100,))  # 100个标签，取值为0或者1

# 实例化数据集
dataset = MyDataset(data, lables)
# 实例化DataLoader
dataloader = DataLoader(dataset, batch_size=10, shuffle=True, num_workers=0)

# 遍历DataLoader
for batch_idx, (batch_data, batch_lables) in enumerate(dataloader):
    print(f"批次 {batch_idx + 1}")
    print("数据:", batch_data)
    print("标签:", batch_lables)
    if batch_idx == 2:  # 仅显示前3个批次
        break
