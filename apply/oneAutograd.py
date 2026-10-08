import torch

x = torch.tensor(2.0, requires_grad=True)
y = torch.tensor(3.0, requires_grad=True)

# 前向传播：z = x² + 3y
z = x ** 2 + y * 3

# 反向传播：自动计算 dz/dx 和 dz/dy
z.backward()

# 查看梯度
print(x.grad)  # tensor(4.)  ← dz/dx = 2x = 2×2 = 4
print(y.grad)   # tensor(3.)  ← dz/dy = 3


# print(z)
# print(z.grad_fn)
#
# print(z.grad_fn.next_functions)