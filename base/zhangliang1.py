import torch

# 创建一个 2 X 3 的全 0 张量
a = torch.zeros(2, 3)
print(a)

# 创建一个 2 X 3 的全 1 张量
b = torch.ones(2, 3)
print(b)

# 创建一个 2 X 3 的随机数张量
c = torch.rand(2, 3)
print(c)

# 从NumPy 数组创建张量
import numpy as np

numpy_array = np.array([[1, 2], [3, 4]])
tensor_from_numpy = torch.from_numpy(numpy_array)
print(tensor_from_numpy)

# 在指定设备(CPU/GPU) 上创建张量
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
d = torch.randn(2, 3, device=device)
print(d)

# 张量相加
e = torch.randn(2, 3)
f = torch.randn(2, 3)
print("张量相加:")
print(e + f)

# 逐元素乘法
print(e * f)

# 张量的转置
g = torch.randn(3, 2)
print(g.t())

# 张量的形状
print(g.shape)

# 创建一个需要梯度的张量
tensor_requires_grad = torch.tensor([1.0], requires_grad=True)

# 进行一些操作
tensor_result = tensor_requires_grad * 2

# 计算梯队
tensor_result.backward()
print(tensor_requires_grad.grad) # 输出梯度
