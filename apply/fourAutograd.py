import torch

x = torch.tensor(3.0, requires_grad=True)

# 在no_grad上下文中，所有运算不会被追踪梯度
with torch.no_grad():
    y = x ** 2
    print(y.requires_grad)  # False ← 不再追踪梯度
    print(y.grad_fn)  # None  ← 没有计算图节点

# 退出 no_grad 上下文后，恢复正常追踪
z = x ** 2
print(z.requires_grad)

# 常见用途：模型评估时包裹整个推理过程
model = torch.nn.Linear(10, 1)
inputs = torch.randn(32, 10)

with torch.no_grad():
    output = model(inputs)  # 不构建计算图，速度更快，内存更省
