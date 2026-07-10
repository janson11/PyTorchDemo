import torch

# 当前安装的PyTorch库的版本
print(torch.__version__)
# 检查 CUDA 是否可用，即你的系统有NVIDIA的GPU
print(torch.cuda.is_available())

# MPS 可用性检测（M1/M2/M3/M4 芯片专用）
print("MPS 是否可用:", torch.backends.mps.is_available())
print("MPS 是否构建启用:", torch.backends.mps.is_built())

# 设备自动选择
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print("当前使用设备：", device)

# 测试：把张量放到GPU(MPS)上运算
x = torch.randn(1000,1000).to(device)
y = x @ x.T
print("张量运算完成，设备:", y.device)