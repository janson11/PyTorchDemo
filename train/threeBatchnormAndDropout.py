from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms


DATA_DIR = Path(__file__).resolve().parents[1] / "apply" / "data"
CHECKPOINT = Path(__file__).resolve().parent / "checkpoints" / "best1.pth"
EPOCHS = 50
BATCH_SIZE = 128


class ConvBNBlock(nn.Module):
    """标准卷积块：Conv → BN → ReLU"""

    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),  # bias=False：BN 的 beta 已起偏置作用
            nn.ReLU(inplace=True),
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
            nn.AdaptiveAvgPool2d(1),  # 4×4 → 1×1
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.5),  # 只在全连接层用 Dropout
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


def make_loaders(device, train_samples=None, val_samples=None):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ])
    train_data = datasets.CIFAR10(DATA_DIR, train=True, download=True, transform=transform)
    val_data = datasets.CIFAR10(DATA_DIR, train=False, download=True, transform=transform)
    if train_samples is not None:
        train_data = Subset(train_data, range(min(train_samples, len(train_data))))
    if val_samples is not None:
        val_data = Subset(val_data, range(min(val_samples, len(val_data))))
    options = {"batch_size": BATCH_SIZE, "pin_memory": device.type == "cuda"}
    return (
        DataLoader(train_data, shuffle=True, **options),
        DataLoader(val_data, shuffle=False, **options),
    )


def run_epoch(model, loader, criterion, device, optimizer=None):
    train = optimizer is not None
    if train:
        model.train()  # ← BN 用 batch 统计，Dropout 激活
    else:
        model.eval()  # ← BN 用 running 统计，Dropout 关闭

    total_loss, correct = 0.0, 0
    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            if train:
                optimizer.zero_grad()

            outputs = model(imgs)
            loss = criterion(outputs, labels)

            if train:
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * imgs.size(0)
            correct += (outputs.argmax(1) == labels).sum().item()

    n = len(loader.dataset)
    return total_loss / n, correct / n


def main(epochs=EPOCHS, train_samples=None, val_samples=None, checkpoint=CHECKPOINT):
    if epochs < 1:
        raise ValueError("epochs must be positive")
    if train_samples is not None and train_samples < 1:
        raise ValueError("train_samples must be positive")
    if val_samples is not None and val_samples < 1:
        raise ValueError("val_samples must be positive")

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    train_loader, val_loader = make_loaders(device, train_samples, val_samples)
    model = SmallCNN(num_classes=10).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
    checkpoint = Path(checkpoint)
    checkpoint.parent.mkdir(parents=True, exist_ok=True)

    best_acc = -1.0
    for epoch in range(1, epochs + 1):
        train_loss, train_acc = run_epoch(model, train_loader, criterion, device, optimizer)
        val_loss, val_acc = run_epoch(model, val_loader, criterion, device)
        scheduler.step()
        print(f"Epoch {epoch:2d}/{epochs} | "
              f"Train {train_loss:.4f}/{train_acc:.4f} | "
              f"Val {val_loss:.4f}/{val_acc:.4f}")

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), checkpoint)


if __name__ == "__main__":
    main()
