import torch
import torch.nn as nn

# 输入形状（N，C) 或（N，C，L）
# N = batch_size C = 特征数/通道数 ,L = 序列长度

# BatchNorm1d用于全连接层/序列数据
bn1d = nn.BatchNorm1d(
    num_features=128,  # 特征/通道数
    eps=1e-05,  # 防止除零的小常熟
    momentum=0.1,  # 滑动平均的动量
    affine=True,  # 是否学习gamma 和beat
    track_running_stats=True  # 是否追踪运行时均值/方差
)

# 全链接层后使用
x = torch.randn(32, 128)
out = bn1d(x)
print(out.shape)

# 序列数据（如1D卷积后）
x_seq = torch.randn(32, 128, 50)  # (batch, channels, seq_len)
out_seq = bn1d(x_seq)
print(out_seq.shape)

# BatchNorm2d:用于CNN图像特征图
bn2d = nn.BatchNorm2d(num_features=64)  # 64个通道

x = torch.randn(32, 64, 28, 28)  # (batch, channels, height, width)
out = bn2d(x)
print(out.shape)

# BatchNorm3d:用于3D卷积(视频/医学影像)
# 输入形状（N，C，D，H，W）
bn3d = nn.BatchNorm3d(num_features=32)

x = torch.randn(4, 32, 16, 32, 32)  # (batch, channels, depth, h, w)
out = bn3d(x)
print(out.shape)
