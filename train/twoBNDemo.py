# CNN图像分类(BN为主)
import torch
import torch.nn as nn


class ConvBNBlock(nn.Module):
    """标准卷积块：Conv → BN → ReLU"""

    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),  # bias=False：BN 的 beta 已起偏置作用
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)


class SmallCNN(nn.Module):
    """用于 CIFAR-10 的小型 CNN"""

    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            ConvBNBlock(3, 32),  # 32×32 → 32×32
            ConvBNBlock(32, 64, stride=2),  # 32×32 → 16×16
            ConvBNBlock(64, 128, stride=2),  # 16×16 → 8×8
            ConvBNBlock(128, 256, stride=2),  # 8×8 → 4×4
            nn.AdaptiveAvgPool2d(1)  # 4×4 → 1×1
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.5),  # 只在全连接层用 Dropout
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        return self.classifier(self.features(x))


model = SmallCNN(num_classes=10)
x = torch.randn(16, 3, 32, 32)
model.eval()
print(model(x).shape)
