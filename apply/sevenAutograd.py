import torch

x = torch.tensor(2.0, requires_grad=True)
y = x ** 3  # y = x³

# 第一次backward(保留计算图)
y.backward(retain_graph=True)
print(x.grad)

# 第二次backward(计算图仍然存在)
x.grad.zero_()
y.backward(retain_graph=True)
print(x.grad)

# 最后一次不需要再保留
x.grad.zero_()
y.backward()  # 此后计算图被释放
print(x.grad)

# 再次尝试 backward 会报错（计算图已释放）
# y.backward()   # &#x274c; RuntimeError: Trying to backward through the graph a second time
