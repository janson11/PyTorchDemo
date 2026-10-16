import torch
import torch.nn as nn

lstm = nn.LSTM(
    input_size=64,    # 每个时间步输入向量的维度
    hidden_size=128,  # 隐藏状态(以及细胞状态)的维度
    num_layers=2,     # 堆叠层数，默认是1
    bias=True,        # 是否使用偏置项，默认是True
    batch_first=False,# 输入/输出 shape 中batch 是否在第一维,默认是False
    dropout=0.0,      # 层间dropout 概率（仅在 num_layers > 1 时生效）
    bidirectional=False, # 是否使用双向 LSTM，默认 False
    proj_size=0)  # 投影层维度（LSTM with projection），默认 0 表示不使用
# 查看参数量
total_params = sum(p.numel() for p in lstm.parameters())
print(f"LSTM 参数量: {total_params:,}")