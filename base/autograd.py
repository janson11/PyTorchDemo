import torch

# 创建一个需要计算梯度的张量
x = torch.randn(2, 2, requires_grad=True)
print(x)

# 执行某些操作
y = x + 2
z = y * y * 3
out = z.mean()
print(out)

# 反向传播，计算梯度
out.backward()

# 查看 x的梯度
print(x.grad)
# 在神经网络训练中，自动求导主要用于实现反向传播算法。
# 反向传播是一种通过计算损失函数关于网络参数的梯度来训练神经网络的方法。在每次迭代中，网络的前向传播会计算输出和损失，然后反向传播会计算损失关于每个参数的梯度，并使用这些梯度来更新参数
