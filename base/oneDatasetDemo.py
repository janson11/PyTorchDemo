import torch
from torch.utils.data import Dataset


# 自定义数据集
class MyDataSet(Dataset):
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
data = torch.randn(100, 5)  # 100个样本，每个样本有5个特征
labels = torch.randint(0, 2, (100,))  # 100个标签，取值为0或者1

# 实例化数据集
dataset = MyDataSet(data, labels)

# 测试数据集
print("数据集大小:", len(dataset))
print("第0个样本:", dataset[0])
