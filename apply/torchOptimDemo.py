import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR

# 直接运行脚本时，将项目根目录加入模块搜索路径。
if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from base.datasetDemo import dataloader


# 1. 定义一个简单的模型
class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        # datasetDemo 中每条数据有 2 个特征，标签为 0 或 1。
        self.fc1 = nn.Linear(2, 2)

    def forward(self, x):
        return self.fc1(x)


# 配置
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
EPOCHS = 100
BATCH_SIZE = 32
LR = 1e-3

# 创建模型并移动到设备
model = SimpleNet().to(DEVICE)

# 损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=0.01)

# 学习率调度器(余弦退火)
scheduler = CosineAnnealingLR(optimizer, EPOCHS, eta_min=1e-6)

# 训练循环
best_acc = 0.0
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0.0
    correct = 0.0

    for inputs, labels in dataloader:
        inputs = inputs.to(DEVICE)
        # CrossEntropyLoss 的类别索引需要使用整数类型。
        labels = labels.to(device=DEVICE, dtype=torch.long)

        # 清空梯度(推荐使用set_to_none = True)
        optimizer.zero_grad(set_to_none=True)

        # 前向传播
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        # 反向传播
        loss.backward()

        # 梯度裁剪（防止梯度爆炸）
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        # 更新参数
        optimizer.step()

        # 统计
        total_loss += loss.item()
        correct += (outputs.argmax(1) == labels).sum().item()

    # 更新学习率
    scheduler.step()

    # 打印训练信息
    avg_loss = total_loss / len(dataloader)
    accuracy = correct / len(dataloader.dataset)
    current_lr = scheduler.get_last_lr()[0]
    print(f"Epoch {epoch + 1}/{EPOCHS} | Loss: {avg_loss:.4f} | "
          f"Acc: {accuracy:.4f} | LR: {current_lr:.6f}")

    # 保存最佳模型
    if accuracy > best_acc:
        best_acc = accuracy
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'best_acc': best_acc,
        }, 'best_model.pth')

print(f"训练完成,最佳准确率：{best_acc:.4f}")
