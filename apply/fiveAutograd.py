import torch
import torch.nn as nn

model = nn.Linear(10, 1)


@torch.no_grad()
def predict(model, x):
    """推理函数，不需要计算梯度"""
    return model(x)


x = torch.randn(5, 10)
output = predict(model, x)
print(output.requires_grad)
