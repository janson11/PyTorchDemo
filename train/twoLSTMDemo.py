import torch
import torch.nn as nn

# ── batch_first=False（默认）────────────────────────
lstm = nn.LSTM(input_size=32, hidden_size=64, batch_first=False)

# 输入形状
seq_len, batch_size, input_size = 10, 4, 32
x = torch.randn(seq_len, batch_size, input_size)

output, (h_n, c_n) = lstm(x)

print(f"output形状:{output.shape}")
print(f"h_n形状:{h_n.shape}")
print(f"c_n形状:{c_n.shape}")

lstm_bf = nn.LSTM(input_size=32, hidden_size=64, batch_first=True)
x = torch.randn(seq_len, batch_size, input_size)
output, (h_n, c_n) = lstm_bf(x)
print(f"output形状:{output.shape}")
print(f"h_n形状:{h_n.shape}")

# ── 多层双向 LSTM 的输出形状 ──────────────────────
lstm_bd = nn.LSTM(input_size=32, hidden_size=64, num_layers=3, bidirectional=True, batch_first=True)
x = torch.randn(seq_len, batch_size, input_size)
output, (h_n, c_n) = lstm_bd(x)
print(f"output形状:{output.shape}")

print(f"h_n形状:{h_n.shape}")
