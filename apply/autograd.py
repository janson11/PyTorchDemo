import torch

# 创建一个需要追踪梯度的张量(requires_grad=False)
x = torch.tensor(3.0, requires_grad=True)
print(x)
print(x.requires_grad)

# 也可以在创建后修改
y = torch.tensor(2.0)
print(y.requires_grad)
y.requires_grad_(True)
print(y.requires_grad)

z = x * y
print(z.requires_grad)
