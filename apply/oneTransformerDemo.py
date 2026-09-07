import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.models.video.mvit import PositionalEncoding


# 定义Tranformer模型

class TransformerModel(nn.Module):
    def __init__(self, input_dim, model_dim, num_heads, num_layers, output_dim):
        super(TransformerModel, self).__init__()

        # 词嵌入：将词索引映射为model_dim维向量
        self.embedding = nn.Embedding(input_dim, model_dim)

        # 位置编码：可学习的位置向量，最大支持长度 1000
        self.positional_encoding = nn.Parameter(
            torch.zeros(1, 1000, model_dim)
        )

        # PyTorch 内置Tranformer（包含编码器 + 解码器）
        self.transformer = nn.Transformer(
            d_model=model_dim,  # 向量维度
            nhead=num_heads,  # 多头注意力的头数
            num_encoder_layers=num_layers,  # 编码器层数
            num_decoder_layers=num_layers  # 解码器层数
        )

        # 最终线性层：将向量映射回词汇表大小(用于预测下一个词)
        self.fc = nn.Linear(model_dim, output_dim)

    def forward(self, src, tgt):
        src_seq_length = src.size(1)
        tgt_seq_length = tgt.size(1)

        # 词嵌入 + 位置编码(两者相加)
        src = self.embedding(src) + self.positional_encoding[:, :src_seq_length, :]
        tgt = self.embedding(tgt) + self.positional_encoding[:, :tgt_seq_length, :]

        # 通过Transformer (编码器读src,解码器生成tgt)
        transformer_output = self.transformer(src, tgt)

        # 线性层输出每个位置的词汇概率
        output = self.fc(transformer_output)
        return output


# --- 超参数设置
input_dim = 10000  # 词汇表大小(共有多个不同的词)
model_dim = 512  # 每个词的向量维度 （原论文使用512）
num_heads = 8  # 多头注意力头数(需能整除model_dim)
num_layers = 6  # 编码器/解码器层数(原文使用6)
output_dim = 10000  # 输出维度(与词汇表大小相同)

# ----初始化模型、损失函数和优化器----
model = TransformerModel(input_dim, model_dim, num_heads, num_layers, output_dim)
criterion = nn.CrossEntropyLoss()  # 多分类交叉熵损失
optimizer = optim.Adam(model.parameters(), lr=0.001)  # Adam优化器

# --- 构造示例数据(实际使用时换成真实语料)---

# src:源序列(如中文) shape = (序列长度=10,批量大小=32)
src = torch.randint(0, input_dim, (10, 32))
# tgt:目标序列(如英文) shape = (序列长度=20,批量大小=32)
tgt = torch.randint(0, input_dim, (20, 32))

# ---前向传播
output = model(src, tgt)
# output.shape = (20,32,10000) # 每个位置对词汇表的预测分布

# --- 计算损失
loss = criterion(output.view(-1, output_dim), tgt.view(-1))

# --- 反向传播 + 更新权重
optimizer.zero_grad()  # 清空上一步的梯度
loss.backward()  # 计算梯度
optimizer.step()  # 更新参数

print(f"损失值:{loss.item():.4f}")