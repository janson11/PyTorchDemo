import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.datasets
import torchvision.transforms as transforms

# 定义数据转换
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

classes = ('plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # 卷积层1:输入3通道(RGB) 输出6通道，5X5卷积核
        self.conv1 = nn.Conv2d(3, 6, kernel_size=5)
        # 池化层 ：2X2窗口，步长2
        self.pool = nn.MaxPool2d(2, 2)
        # 卷积层2：输入6通道，输出16通道，5X5卷积核
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        # 全连接层1：输入16*5*5 输出120
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        # 全连接层2 :输入120，输出84
        self.fc2 = nn.Linear(120, 84)
        # 全连接层3： 输入84，输出10（对应10个类别）
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        # 第一层卷积 + ReLU  +池化
        x = self.pool(F.relu(self.conv1(x)))
        # 第二层卷积 + ReLU  +池化
        x = self.pool(F.relu(self.conv2(x)))
        # 展平特征图
        x = x.view(-1, 16 * 5 * 5)
        # 全连接层+ ReLU
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        # 输出层
        x = self.fc3(x)
        return x


import torch.optim as optim


def main():
    # 加载 CIFAR-10 数据集。创建 DataLoader 必须在主进程中进行，
    # 这样 multiprocessing 的 spawn 模式不会重复执行训练代码。
    trainset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform
    )
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=4, shuffle=True, num_workers=2
    )

    testset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform
    )
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=4, shuffle=False, num_workers=2
    )

    # 实例化网络
    net = Net()

    # 保存模型
    path = './cifar_net.pth'
    torch.save(net.state_dict(), path)
    net.load_state_dict(torch.load(path, weights_only=True))

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    for epoch in range(10):
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            if i % 2000 == 1999:
                print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
                running_loss = 0.0

    print('Finished Training')

    # 使用模型进行预测
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in testloader:
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print(f'Accuracy on test images: {100 * correct / total:.2f}%')

    outputs = net(images)
    _, predicted = torch.max(outputs.data, 1)
    print('Predicted: ', ' '.join(f'{classes[predicted[j]]:5s}' for j in range(4)))


if __name__ == '__main__':
    main()
