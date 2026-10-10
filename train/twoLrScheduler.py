"""MNIST training example with linear warmup and cosine learning-rate decay."""

from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms


EPOCHS = 100
WARMUP_EPOCHS = 5
BASE_LR = 1e-3
MIN_LR = 1e-6
DATA_DIR = Path(__file__).resolve().parent / "data"
CHECKPOINT_DIR = Path(__file__).resolve().parent / "checkpoints" / "two_lr_scheduler"


class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, 128),
            nn.ReLU(),
            nn.Linear(128, 10),
        )

    def forward(self, inputs):
        return self.network(inputs)


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def make_loaders(device, train_samples=None, val_samples=None):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ])
    train_data = datasets.MNIST(DATA_DIR, train=True, download=True, transform=transform)
    val_data = datasets.MNIST(DATA_DIR, train=False, download=True, transform=transform)
    if train_samples is not None:
        train_data = Subset(train_data, range(min(train_samples, len(train_data))))
    if val_samples is not None:
        val_data = Subset(val_data, range(min(val_samples, len(val_data))))
    options = {"batch_size": 256, "pin_memory": device.type == "cuda"}
    return (
        DataLoader(train_data, shuffle=True, **options),
        DataLoader(val_data, shuffle=False, **options),
    )


def main(epochs=EPOCHS, warmup_epochs=WARMUP_EPOCHS, checkpoint_dir=CHECKPOINT_DIR,
         train_samples=None, val_samples=None):
    if not 0 < warmup_epochs < epochs:
        raise ValueError("warmup_epochs must be positive and less than epochs")

    device = get_device()
    print(f"训练设备：{device}")
    train_loader, val_loader = make_loaders(device, train_samples, val_samples)
    model = MyModel().to(device)
    optimizer = optim.AdamW(model.parameters(), lr=BASE_LR, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    warmup = LinearLR(optimizer, start_factor=0.1, total_iters=warmup_epochs)
    cosine = CosineAnnealingLR(optimizer, T_max=epochs - warmup_epochs, eta_min=MIN_LR)
    scheduler = SequentialLR(optimizer, [warmup, cosine], milestones=[warmup_epochs])

    checkpoint_dir = Path(checkpoint_dir)
    latest_path = checkpoint_dir / "latest_model.pth"
    best_path = checkpoint_dir / "best_model.pth"
    start_epoch, best_acc = 0, -1.0
    if latest_path.exists():
        checkpoint = torch.load(latest_path, map_location=device, weights_only=True)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        start_epoch = checkpoint["epoch"] + 1
        best_acc = checkpoint["best_acc"]
        print(f"恢复自 Epoch {start_epoch}，最佳准确率 {best_acc:.4f}")
    else:
        print("从头开始训练")

    for epoch in range(start_epoch, epochs):
        model.train()
        total_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(inputs), labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item() * inputs.size(0)

        model.eval()
        correct = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                correct += (model(inputs).argmax(1) == labels).sum().item()

        avg_loss = total_loss / len(train_loader.dataset)
        val_acc = correct / len(val_loader.dataset)
        cur_lr = scheduler.get_last_lr()[0]
        print(f"Epoch {epoch + 1:3d}/{epochs} | "
              f"Loss: {avg_loss:.4f} | Acc: {val_acc:.4f} | LR: {cur_lr:.2e}")

        scheduler.step()
        improved = val_acc > best_acc
        best_acc = max(best_acc, val_acc)
        checkpoint = {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "best_acc": best_acc,
        }
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        if improved:
            torch.save(checkpoint, best_path)
            print(f"  保存最优模型，Acc: {best_acc:.4f}")
        torch.save(checkpoint, latest_path)

    print(f"\n训练完成，最佳验证准确率: {best_acc:.4f}")


if __name__ == "__main__":
    main()
