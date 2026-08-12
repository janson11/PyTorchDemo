# 1. 导入必要库
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt

# 数据集：字符序列预测（Hello -> Elloh）
char_set = list("hello")
char_to_idx = {c: i for i, c in enumerate(char_set)}
idx_to_char = {i: c for i, c in enumerate(char_set)}

# 数据准备
input_str = "hello"
target_str = "elloh"
input_data = [char_to_idx[c] for c in input_str]
target_data = [char_to_idx[c] for c in target_str]

# 转换为独热编码
input_one_hot = np.eye(len(char_set))[input_data]

# 转换为Pytorch Tensor
inputs = torch.tensor(input_one_hot, dtype=torch.float32)
targets = torch.tensor(target_data, dtype=torch.long)

# 模型超参数
input_size = len(char_set)
hidden_size = 8
output_size = len(char_set)
num_epochs = 200
learning_rate = 0.1


# 2. 定义RNN模型
class SimpleRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleRNN, self).__init__()
        # 定义RNN层
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
        # 定义全连接层
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x, hidden):
        # x: (batch_size,seq_len,input_size)
        out, hidden = self.rnn(x, hidden)
        # out, _ = self.rnn(x)  # out:(batch_size,seq_len,hidden_size)
        # 取序列最后一个时间步的输出作为模型的输出
        # out = out[:, -1, :]  # (batch_size, hidden_size)
        out = self.fc(out)  # 全连接层
        return out, hidden


# # 3 创建训练数据
# # 生成一些随机序列数据
# num_samples = 1000
# seq_len = 10
# input_size = 5
# output_size = 2  # 假设二分类问题
#
# # 随机生成输入数据(batch_size,seq_len,input_size)
# X = torch.randn(num_samples, seq_len, input_size)
# # 随机生成目标标签(batch_size,output_size)
# Y = torch.randn(0, output_size, (num_samples,))
#
# # 创建数据加载器
# dataset = TensorDataset(X, Y)
# train_loader = DataLoader(dataset, batch_size=32, shuffle=True)

# 4.定义损失函数与优化器
# 模型实例化
model = SimpleRNN(input_size, hidden_size, output_size)

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()  # 多分类交叉熵损失
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# 5、训练模型
losses = []
hidden = None  # 初始化隐藏状态为None

for epoch in range(num_epochs):
    optimizer.zero_grad()

    # 前向传播
    outputs, hidden = model(inputs.unsqueeze(0), hidden)
    hidden = hidden.detach()  # 防止梯度爆炸

    # 计算损失
    loss = criterion(outputs.view(-1, output_size), targets)
    loss.backward()
    optimizer.step()
    losses.append(loss.item())

    if (epoch + 1) % 20 == 0:
        print(
            f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

# 6、测试模型
with torch.no_grad():
    test_hidden = None
    test_outputs, _ = model(inputs.unsqueeze(0), test_hidden)
    predicted = torch.argmax(test_outputs, dim=2).squeeze().numpy()
    print("Input sequence: ", ''.join([idx_to_char[i] for i in input_data]))
    print("Predicted sequence: ", ''.join([idx_to_char[i] for i in predicted]))

# 7、可视化损失
plt.plot(losses, label='Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('RNN Training Loss Over Epochs')
plt.legend()
plt.show()
