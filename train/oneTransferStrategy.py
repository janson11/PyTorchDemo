"""
1 策略一：特征提取（冻结全部）
冻结预训练模型的全部参数，只训练新替换的分类头。

适合数据量极少（几百张），或任务与源任务高度相似的场景。
"""
import pylab as p
import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import ResNet18_Weights

NUM_CLASSES = 5  # 目标任务的类别数

# step 1:加载预训练模型
model = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)

# step 2 :冻结所有参数
for param in model.parameters():
    param.requires_grad = False

# step 3: 替换分类头 (这部分参数默认 requires_grad = True)
in_features = model.fc.in_features  # 512
model.fc = nn.Linear(in_features, NUM_CLASSES)

# 验证：只有分类头是可训练的
trainable = [(n, p.shape() for n, p in model.named_parameters() if p.requires_grad)]
print(f"可训练层数:{len(trainable)}")

for name, shape in trainable:
    print(f"{name}:{shape}")

# Step 4 : 优化器只传可训练参数(更高效)
optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3)

# 或等价的更清晰写法
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
