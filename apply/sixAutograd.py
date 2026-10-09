import torch

x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
y = x ** 2 + x ** 3  # y 有 grad_fn

# detach 后的张量与 y 共享数据，但脱离计算图
y_detached = y.detach()
print(y_detached.requires_grad)  # False
print(y_detached.grad_fn)  # None

# 转为 numpy（带梯度的张量不能直接转）
y_detached.numpy()

# 记录损失值时应 detach（避免保留计算图消耗内存）
losses = []
for i in range(3):
    loss = (x ** 2).sum()
    losses.append(loss.detach().item())  # .item() 将标量张量转为 Python float
    loss.backward()
    x.grad.zero_()

print(losses)
