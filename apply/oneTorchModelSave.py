import torch
import torch.nn as nn
import torch.optim as optim


# 定义一个简单模型
class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.fc1 = nn.Linear(10, 2)

    def forward(self, x):
        return self.fc1(x)


# 初始化
model = SimpleModel()
optimizer = optim.SGD(model.parameters(), lr=0.01)
criterion = nn.CrossEntropyLoss()

# 模拟训练过程
for epoch in range(5):
    # 模拟训练步骤
    inputs = torch.randn(32, 10)
    labels = torch.randint(0, 2, (32,))

    optimizer.zero_grad()
    outputs = model(inputs)
    loss = criterion(outputs, labels)
    loss.backward()
    optimizer.step()

    # 每2个epoch保存一次检查点
    if epoch % 2 == 0:
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss.item()
        }
        torch.save(checkpoint, f'checkpoint_epoch{epoch}.pth')
        print(f'Checkpoint saved at epoch {epoch}')

# 最终保存
torch.save(model.state_dict(), f'final_model.pth')

# 加载示例
loaded_model = SimpleModel()
loaded_model.load_state_dict(torch.load(f'final_model.pth'))
loaded_model.eval()

# 测试加载的模型
test_input = torch.randn(1, 10)
with torch.no_grad():
    output = loaded_model(test_input)
print(f"Test output: {output}")
