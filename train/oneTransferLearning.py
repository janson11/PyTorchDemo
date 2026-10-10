import torch
import torchvision.models as models

# 加载预训练模型（自动下载权重）
# PyTorch >= 0.13 推荐新写法：使用 weights 参数
from torchvision.models import ResNet50_Weights

model = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)

# 旧写法 (仍然邮箱，但会收到deprecation警告)
model = models.resnet50(pretrained=True)
print("旧写法：", model)
# 不加载预训练权重(仅使用网络结构)
model = models.resnet50(weights=None)
# 打印完整结构
print("打印完整结构:", model)

# 只查看最后几层(分类头)
print("只查看最后几层(分类头):", model.fc)

# 统计参数量
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"总参数量: {total_params:,}")
print(f"可训练参数: {trainable_params:,}")
