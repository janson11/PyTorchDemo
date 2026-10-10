"""
完整实战：图像二分类
以猫狗分类为例，展示从数据准备到训练评估的完整迁移学习流程：
"""
# 核心一句话：
# **`import xxx`：导入整个模块 / 包，使用时必须写全名；**
# **`from xxx import yyy`：从模块里只导入指定对象（函数、类、变量），直接用名字，不用写模块前缀。**
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchvision import datasets, transforms, datasets
from torch.utils.data import DataLoader, random_split
from torchvision.models import EfficientNet_B0_Weights, EfficientNet
import torchvision.models as models

# 配置
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_CLASSES = 2
BATCH_SIZE = 32
EPOCHS = 20
BASE_LR = 1e-3
DATA_DIR = 'data/cats_and_dogs'

# 数据准备
train_tfm = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.3, contrast=0.4),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

val_tfm = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

full_dataset = datasets.ImageFolder(DATA_DIR, transform=train_tfm)

n_val = int(len(full_dataset) * 0.2)
n_train = len(full_dataset) - n_val
train_set, val_set = random_split(full_dataset, [n_train, n_val])
val_set.dataset = datasets.ImageFolder(DATA_DIR, transforms=val_tfm)

train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

# 构建迁移模型
weights = EfficientNet_B0_Weights.IMAGENET1K_V1
model = models.efficientnet_b0(weights=weights)
