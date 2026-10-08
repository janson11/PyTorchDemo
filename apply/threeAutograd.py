import torch

x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)

# 前向传播：y 是向量
y = x ** 2

# 非标量必须传入 gradient 参数（形状与 y 相同）
# gradient 可以理解为"损失对 y 的梯度"
y.backward(gradient=torch.ones_like(y))

print(x.grad)
# tensor([2., 4., 6.])  ← dy/dx = 2x，逐元素计算

# 如果上游梯度不是全 1（例如加权）
x.grad.zero_()
y = x ** 2  # 重新前向传播，构建新的计算图
y.backward(gradient=torch.tensor([1.0, 0.5, 2.0]))  # 不同权重
# 实际计算：x.grad = 2x * gradient = [2×1, 4×0.5, 6×2]
print(x.grad)
# tensor([2., 2., 12.])

# 更常见的做法：先 sum/mean 变为标量，再 backward()
x.grad.zero_()
loss = (x ** 2).sum()  # 将向量聚合为标量
loss.backward()
print(x.grad)
# tensor([2., 4., 6.])  ← 与第一种写法等价
