# 实战示例：图像分类流程
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# -----------1、数据准备---------------
# 定义数据变换
train_transforms = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# 加载数据集
train_set = datasets.CIFAR10(root='./data', train=True, download=True, transform=train_transforms)

# 创建数据加载器
train_loader = DataLoader(train_set, batch_size=32, shuffle=True)

# -----------2、模型训练---------------
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models

# 使用预训练模型
model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

# 修改最后一层(适应 CIFAR10 的10分类)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 10)

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

# 训练循环
for epoch in range(10):
    for images, labels in train_loader:
        outputs = model(images)
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

# 导出为ONNX格式
model.eval()
dummy_input = torch.randn(1, 3, 224, 224)
torch.onnx.export(
    model,
    dummy_input,
    "model.onnx",
    input_names=['input'],
    output_names=['output'],
)
