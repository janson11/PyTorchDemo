import torch
import torch.nn as nn

# 实例一：全连接分类网络（BN + Dropout）

class MLPClassifier(nn.Module):
    """
    带 BatchNorm1d 和 Dropout 的全连接分类器
    适用于表格数据分类任务
    """

    def __init__(self, in_dim, hidden_dims, num_classes, dropout=0.5):
        super().__init__()
        layers = []
        prev_dim = in_dim

        for hidden_dim in hidden_dims:
            layers += [
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
            ]
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


model = MLPClassifier(
    in_dim=784,
    hidden_dims=[512, 256, 128],
    num_classes=10,
    dropout=0.4)

# 验证输出形状
x = torch.randn(32, 784)
model.eval()
print(model(x).shape)
