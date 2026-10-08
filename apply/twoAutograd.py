import torch

x = torch.tensor(2.0, requires_grad=True)

# 第一次反向传播
loss = x ** 2
loss.backward()
print(x.grad)

# 第二次反向传播(没有清零)
loss = x ** 2
loss.backward()
print(x.grad)

# 正确做法：每次反向传播前先清零
x.grad.zero_()
loss = x ** 2
loss.backward()
print(x.grad)
