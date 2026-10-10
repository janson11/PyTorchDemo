import torch
import torch.nn as nn
import torch.optim as optim

model = nn.Linear(10, 1)
optimizer = optim.SGD(model.parameters(), lr=0.1)

# 1 .创建调度器，传入optimizer
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

for epoch in range(1, 100):
    # 2. 训练
    train(model, optimizer)

    # 3. 每个epoch结束后调用scheduler.step()
    scheduler.step()
    # 4.查看当前学习率
    current_lr = scheduler.get_last_lr()[0]
    print(f"Epoch {epoch + 1}, LR: {current_lr:.6f}")
