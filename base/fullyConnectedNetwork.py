import torch
import torch.nn as nn
import torch.nn.functional as F


# 简单的全连接神经网络
# 定义一个简单的神经网络模型
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        # 定义一个输入层到隐藏层的全连接层
        self.fc1 = nn.Linear(2, 2)  # 输入 2 个特征，输出 2 个特征
        self.fc2 = nn.Linear(2, 1)  # 输入 2 个特征，输出 1 个预测值

    def forward(self, x):
        # 前向传播过程
        x = torch.relu(self.fc1(x))  # 使用ReLU激活函数
        x = self.fc2(x)  # 输出层
        return x


# 创建模型实例
model = SimpleNN()

# 打印模型
print(model)

#  ReLU激活
input_tensor = torch.rand(2, 3)
output = F.relu(input_tensor)
print("ReLU激活 \n", output)

# Sigmoid激活
output = torch.sigmoid(input_tensor)
print("Sigmoid激活 \n", output)
# Tanh激活
output = torch.tanh(input_tensor)
print("Tanh激活 \n", output)


# 均方误差损失
criterion = nn.MSELoss()

# 交叉熵损失
criterion = nn.CrossEntropyLoss()

# 二分类交叉熵损失
criterion = nn.BCEWithLogitsLoss()
