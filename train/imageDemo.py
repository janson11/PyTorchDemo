"""Cat/dog transfer learning with a local image folder or cached CIFAR-10."""

from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, models, transforms
from torchvision.models import EfficientNet_B0_Weights

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = Path(__file__).resolve().parent / "data" / "cats_and_dogs"
CIFAR_DIR = PROJECT_ROOT / "apply" / "data"
CHECKPOINT = Path(__file__).resolve().parent / "best_image_model.pth"
BATCH_SIZE = 32
EPOCHS = 20
BASE_LR = 1e-3
class CatDogCIFAR(Dataset):
    def __init__(self, root, train, transform):
        self.dataset = datasets.CIFAR10(root, train=train, download=False, transform=transform)
        self.indices = [i for i, label in enumerate(self.dataset.targets) if label in (3, 5)]
        self.classes = ["cat", "dog"]

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        image, label = self.dataset[self.indices[index]]
        return image, 0 if label == 3 else 1


def make_val_transform():
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])


def make_datasets(data_dir=DATA_DIR):
    train_tfm = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.3, contrast=0.4),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    val_tfm = make_val_transform()
    data_dir = Path(data_dir)
    if data_dir.is_dir():
        train_all = datasets.ImageFolder(data_dir, transform=train_tfm)
        val_all = datasets.ImageFolder(data_dir, transform=val_tfm)
        if len(train_all.classes) != 2:
            raise ValueError(f"{data_dir} must contain exactly two class folders")
        if len(train_all) < 2:
            raise ValueError(f"{data_dir} needs at least two images")
        indices = torch.randperm(len(train_all), generator=torch.Generator().manual_seed(42)).tolist()
        n_val = max(1, int(len(indices) * 0.2))
        return Subset(train_all, indices[n_val:]), Subset(val_all, indices[:n_val])

    if not (CIFAR_DIR / "cifar-10-batches-py").is_dir():
        raise FileNotFoundError(
            f"No cat/dog images at {data_dir}. Create cat/ and dog/ subfolders "
            f"there, or download CIFAR-10 into {CIFAR_DIR}."
        )
    print(f"未找到 {data_dir}，使用现有 CIFAR-10 猫/狗图片（32x32）演示。")
    return CatDogCIFAR(CIFAR_DIR, True, train_tfm), CatDogCIFAR(CIFAR_DIR, False, val_tfm)


def make_model():
    model = models.efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
    for parameter in model.features.parameters():
        parameter.requires_grad = False
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(nn.Dropout(p=0.2), nn.Linear(in_features, 2))
    return model


def run_epoch(model, loader, criterion, device, optimizer=None):
    model.train(optimizer is not None)
    total_loss, correct = 0.0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        with torch.set_grad_enabled(optimizer is not None):
            logits = model(images)
            loss = criterion(logits, labels)
            if optimizer is not None:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        total_loss += loss.item() * len(labels)
        correct += (logits.argmax(1) == labels).sum().item()
    return total_loss / len(loader.dataset), correct / len(loader.dataset)


def sample_image(dataset):
    if isinstance(dataset, Subset):
        return sample_image_at(dataset.dataset, dataset.indices[0])
    return sample_image_at(dataset, 0)


def sample_image_at(dataset, index):
    if isinstance(dataset, Subset):
        return sample_image_at(dataset.dataset, dataset.indices[index])
    if isinstance(dataset, CatDogCIFAR):
        return Image.fromarray(dataset.dataset.data[dataset.indices[index]])
    return dataset.samples[index][0]


def predict(image, model, class_names, device):
    if not isinstance(image, Image.Image):
        image = Image.open(image)
    tensor = make_val_transform()(image.convert("RGB")).unsqueeze(0).to(device)

    with torch.inference_mode():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1)[0]
        pred = probs.argmax().item()

    for cls, prob in zip(class_names, probs.cpu().tolist()):
        print(f" {cls}: {prob:.4f}")
    print(f"预测结果：{class_names[pred]}")
    return class_names[pred]


def main(epochs=EPOCHS, data_dir=DATA_DIR, max_train_samples=None, max_val_samples=None,
         checkpoint=CHECKPOINT, image_path=None):
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    train_set, val_set = make_datasets(data_dir)
    if max_train_samples is not None:
        train_set = Subset(train_set, range(min(max_train_samples, len(train_set))))
    if max_val_samples is not None:
        val_set = Subset(val_set, range(min(max_val_samples, len(val_set))))
    options = {"batch_size": BATCH_SIZE, "num_workers": 4, "pin_memory": device.type == "cuda"}
    train_loader = DataLoader(train_set, shuffle=True, **options)
    val_loader = DataLoader(val_set, shuffle=False, **options)

    model = make_model().to(device)
    optimizer = Adam(model.classifier.parameters(), lr=BASE_LR)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    print(f"训练设备：{device}")
    print("=== 阶段一：只训练分类头 ===")
    best_acc = -1.0
    for epoch in range(1, epochs + 1):
        if epoch == 6:
            print("=== 阶段二：解冻主干全量微调 ===")
            for parameter in model.features.parameters():
                parameter.requires_grad = True
            optimizer.add_param_group({"params": model.features.parameters(), "lr": BASE_LR * 0.1})

        train_loss, train_acc = run_epoch(model, train_loader, criterion, device, optimizer)
        val_loss, val_acc = run_epoch(model, val_loader, criterion, device)
        print(f"Epoch {epoch:2d}/{epochs} | "
              f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f} | "
              f"LR: {optimizer.param_groups[0]['lr']:.2e}")
        scheduler.step()
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), checkpoint)
            print(f"  保存最优模型 Acc={best_acc:.4f}")
    print(f"\n训练完成，最佳验证准确率: {best_acc:.4f}")

    model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
    model.eval()

    base_dataset = train_set
    while isinstance(base_dataset, Subset):
        base_dataset = base_dataset.dataset
    image = image_path if image_path is not None else sample_image(val_set)
    predict(image, model, base_dataset.classes, device)


if __name__ == "__main__":
    main()
