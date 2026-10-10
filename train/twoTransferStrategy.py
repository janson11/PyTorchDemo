"""
微调（Fine-tuning）
解冻全部或部分预训练层，使用较小的学习率整体训练。

适合数据量中等，或任务与源任务有所不同的场景。
"""

import torch
import torch.nn as nn
import torchvision.models as models

NUM_CLASSES = 10
model = models.resnet50(weights='IMAGENET1K_V2')

# 方式A  :全量微调(解冻所有层)
# 先冻结
for param in model.parameters():
    param.requires_grad = False

# 再解冻(等价全量微调，此写法常用于逐步解冻场景)
for param in model.parameters():
    param.requires_grad = True

# 替换分类头
model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)

# 全量微调： 主干用小学习率，头部用大学习率
optimizer = torch.optim.SGD(model.parameters(), lr=1e-4, momentum=0.9)

# 方式B：解冻最后N层（部分微调）
model = models.resnet50(weights='IMAGENET1K_V2')

# 先全部冻结
for param in model.parameters():
    param.requires_grad = False

# 只解冻layer4 和fc(ResNet 的最后一个Block 和分类头)
for param in model.layer4.parameters():
    param.requires_grad = True

model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)  # fc 默认可训练
print("可训练参数:")

for name, param in model.named_parameters():
    if param.requires_grad:
        print(f"  {name}")
