import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

# 定义数据预处理
transform = transforms.Compose([
    transforms.ToTensor(),# 转换为张量
    transforms.Normalize((0.5,), (0.5,)) # 标准化
])

# 加载训练数据集
train_dataset = torchvision.datasets.MNIST(root='./data', train=True, transform=transform, download=True)

# 使用DataLoader 加载数据
train_loader = DataLoader(dataset=train_dataset, batch_size=32, shuffle=True)

# 查看一个批次的数据
data_iter = iter(train_loader)
images, labels = next(data_iter)
print(f"批次图像大小: {images.shape}")  # 输出形状为 [batch_size, 1, 28, 28]
print(f"批次标签: {labels}")