import torch
import torch.nn as nn
import torchvision.models as models

NUM_CLASSES = 8
model = models.resnet50(weights='IMAGENET1K_V2')
model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)

# 方案一：两组学习率(主干 VS 头部)
optimizer = torch.optim.Adam([
    {'params': model.fc.parameters(), 'lr': 1e-3},  # 分类头：大学习率
    {'params': [p for n, p in model.named_parameters() if not n.startswith('fc')], 'lr': 1e-3},  # 主干：小学习率
])

# 方案二：逐层递减学习率(最精细)
# 越靠近输出的层，学习率越大
layer_groups = [
    (model.layer1, 1e-5),  # 最浅层，最小学习率
    (model.layer2, 3e-5),
    (model.layer3, 1e-4),
    (model.layer4, 3e-4),  # 最深主干层
    (model.fc, 1e-3)  # 分类头，最大学习率
]

param_groups = [
    {'params': layer.parameters(), 'lr': lr}
    for layer, lr in layer_groups
]
optimizer = torch.optim.Adam(param_groups)

# 方案三： 逐步解冻(Gradual Unfreezing)
# 训练初期只训练头部，逐步解冻更多层(fastai推荐)
model = models.resnet50(weights='IMAGENET1K_V2')
for param in model.parameters():
    param.requires_grad = False
model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)


def unfreeze_layers(model, num_layers):
    """解冻 ResNet 最后 num_layers 个 layer block"""
    layers = [model.layer4, model.layer3, model.layer2, model.layer1]
    for i in range(min(num_layers, len(layers))):
        for param in layers[i].parameters():
            param.requires_grad = True


# Epoch 1-5：只训练分类头
# Epoch 6-10：解冻layer4
unfreeze_layers(model, num_layers=1)

# Epoch 11+:解冻更多层
unfreeze_layers(model, num_layers=3)
