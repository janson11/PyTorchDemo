import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import math
import copy

"""
多头注意力机制：将输入分割成多个头，每个头独立计算注意力，最后将结果合并。

缩放点积注意力：计算查询和键的点积，缩放后使用 softmax 计算注意力权重，最后对值进行加权求和。

掩码：用于屏蔽无效位置（如填充部分）。
"""


# 多头注意力机制
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0, "d_model必须能被num_heads整除"

        self.d_model = d_model  # 模型维度
        self.num_heads = num_heads  # 注意力头数
        self.d_k = d_model // num_heads  # 每个头的维度

        # 定义线性变换层
        self.W_q = nn.Linear(d_model, d_model)  # 查询变换
        self.W_k = nn.Linear(d_model, d_model)  # 键变换
        self.W_v = nn.Linear(d_model, d_model)  # 值变换
        self.W_o = nn.Linear(d_model, d_model)  # 输出变换

    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        """
        计算缩放点积注意力
        输入形状：
            Q: (batch_size, num_heads, seq_length, d_k)
            K, V: 同Q
        输出形状： (batch_size, num_heads, seq_length, d_k)
        """
        # 计算注意力分数(Q和K的点积)
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)

        # 应用掩码
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, -1e9)

        # 计算注意力权重(softmax归一化)
        attn_probs = torch.softmax(attn_scores, dim=-1)

        # 对值向量加权求和
        output = torch.matmul(attn_probs, V)
        return output

    def split_heads(self, x):
        """
        将输入张量分割为多个头
        输入形状: (batch_size, seq_length, d_model)
        输出形状: (batch_size, num_heads, seq_length, d_k)
        """
        batch_size, seq_length, d_model = x.size()
        if d_model != self.d_model:
            raise ValueError(
                f"Expected the last dimension to be {self.d_model}, got {d_model}."
            )
        return x.view(batch_size, seq_length, self.num_heads, self.d_k).transpose(1, 2)

    def combine_heads(self, x):
        """
        将多个头的输出合并回原始形状
        输入形状: (batch_size, num_heads, seq_length, d_k)
        输出形状: (batch_size, seq_length, d_model)
        """
        batch_size, num_heads, seq_length, d_k = x.size()
        return x.transpose(1, 2).contiguous().view(batch_size, seq_length, self.d_model)

    def forward(self, Q, K, V, mask=None):
        """
        前向传播
        输入形状: Q/K/V: (batch_size, seq_length, d_model)
        输出形状: (batch_size, seq_length, d_model)
        """
        # 线性变换并分割多头
        Q = self.split_heads(self.W_q(Q))
        K = self.split_heads(self.W_k(K))
        V = self.split_heads(self.W_v(V))

        # 计算注意力
        attn_output = self.scaled_dot_product_attention(Q, K, V, mask)

        # 合并多头并输出变换
        output = self.W_o(self.combine_heads(attn_output))
        return output


# 位置前馈网络：由两个全连接层和一个 ReLU 激活函数组成，用于进一步处理注意力机制的输出。
class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff):
        super(PositionwiseFeedForward, self).__init__()
        self.fc1 = nn.Linear(d_model, d_ff)  # 第一层全连接
        self.fc2 = nn.Linear(d_ff, d_model)  # 第二层全连接
        self.relu = nn.ReLU()  # 激活函数

    def forward(self, x):
        # 前馈网络的计算
        return self.fc2(self.relu(self.fc1(x)))


"""
位置编码
位置编码用于注入输入序列中每个 token 的位置信息。

使用不同频率的正弦和余弦函数来生成位置编码。
"""


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_seq_length):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_seq_length, d_model)  # 初始化位置编码矩阵
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)  # 偶数位置使用正弦函数
        pe[:, 1::2] = torch.cos(position * div_term)  # 奇数位置使用、余弦函数
        self.register_buffer('pe', pe.unsqueeze(0))  # 注册为缓冲区

    def forward(self, x):
        # 将位置编码添加到输入中
        if x.size(1) > self.pe.size(1):
            raise ValueError(
                f"Sequence length {x.size(1)} exceeds max_seq_length {self.pe.size(1)}."
            )
        return x + self.pe[:, :x.size(1)]


#     编码器层：包含一个自注意力机制和一个前馈网络，每个子层后接残差连接和层归一化。
class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)  # 自注意力机制
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff)  # 前馈网络
        self.norm1 = nn.LayerNorm(d_model)  # 层归一化
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)  # Dropout

    def forward(self, x, mask):
        # 自注意力机制
        attn_output = self.self_attn(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))  # 残差连接和层归一化
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        return x


# 解码器层：包含一个自注意力机制、一个交叉注意力机制和一个前馈网络，每个子层后接残差连接和层归一化。
class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout):
        super(DecoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)  # 自注意力机制
        self.cross_attn = MultiHeadAttention(d_model, num_heads)  # 交叉注意力机制
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff)  # 前馈网络
        self.norm1 = nn.LayerNorm(d_model)  # 层归一化
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)  # Dropout

    def forward(self, x, enc_output, src_mask, tgt_mask):
        # 自注意力机制
        attn_output = self.self_attn(x, x, x, tgt_mask)
        x = self.norm1(x + self.dropout(attn_output))  # 残差连接和层归一化

        # 交叉注意力机制
        attn_output = self.cross_attn(x, enc_output, enc_output, src_mask)
        x = self.norm2(x + self.dropout(attn_output))  # 残差连接和层归一化

        # 前馈网络
        ff_output = self.feed_forward(x)
        x = self.norm3(x + self.dropout(ff_output))  # 残差连接和层归一化
        return x


# 构建完整的 Transformer 模型
class Transformer(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model, num_heads, num_layers, d_ff, max_seq_length, dropout):
        super(Transformer, self).__init__()
        self.encoder_embedding = nn.Embedding(src_vocab_size, d_model)  # 编码器词嵌入
        self.decoder_embedding = nn.Embedding(tgt_vocab_size, d_model)  # 解码器词嵌入
        self.positional_encoding = PositionalEncoding(d_model, max_seq_length)  # 位置编码

        # 编码器和解码器层
        self.encoder_layer = nn.ModuleList([EncoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])
        self.decoder_layer = nn.ModuleList([DecoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])

        self.output_projection = nn.Linear(d_model, tgt_vocab_size)  # 最终的全连接层
        self.dropout = nn.Dropout(dropout)  # Dropout

    def generate_mask(self, src, tgt):
        # 源掩码：屏蔽填充符（假设填充符索引为0）
        # 形状：（batch_size,1,1,seq_length）
        src_mask = (src != 0).unsqueeze(1).unsqueeze(2)

        # 目标掩码：屏蔽填充符和未来信息
        # 形状：（batch_size,1,seq_length,seq_length）
        tgt_padding_mask = (tgt != 0).unsqueeze(1).unsqueeze(2)
        seq_length = tgt.size(1)

        # 生成上三角矩阵掩码，防止解码时看到未来信息
        nopeak_mask = torch.tril(
            torch.ones(seq_length, seq_length, dtype=torch.bool, device=tgt.device)
        ).unsqueeze(0).unsqueeze(1)
        tgt_mask = tgt_padding_mask & nopeak_mask  # 合并填充掩码和未来信息掩码
        return src_mask, tgt_mask

    def forward(self, src, tgt):
        # 生成掩码
        src_mask, tgt_mask = self.generate_mask(src, tgt)

        # 编码器部分
        src_embedded = self.dropout(self.positional_encoding(self.encoder_embedding(src)))
        enc_output = src_embedded
        for enc_layer in self.encoder_layer:
            enc_output = enc_layer(enc_output, src_mask)

        # 解码器部分
        tgt_embedded = self.dropout(self.positional_encoding(self.decoder_embedding(tgt)))
        dec_output = tgt_embedded
        for dec_layer in self.decoder_layer:
            dec_output = dec_layer(dec_output, enc_output, src_mask, tgt_mask)

        # 最终输出
        output = self.output_projection(dec_output)
        return output


"""
Transformer 模型：包含编码器和解码器部分，每个部分由多个层堆叠而成。

掩码生成：用于屏蔽无效位置和未来信息。

前向传播：依次通过编码器和解码器，最后通过全连接层输出。
class Transformer(nn.Module):
    def __init__(
        self, 
        src_vocab_size,  # 源语言词汇表大小（如英文单词数）
        tgt_vocab_size,  # 目标语言词汇表大小（如中文单词数）
        d_model=512,     # 模型维度（每个词向量的长度）
        num_heads=8,     # 多头注意力的头数
        num_layers=6,    # 编码器/解码器的堆叠层数
        d_ff=2048,       # 前馈网络隐藏层维度
        max_seq_length=100, # 最大序列长度（用于位置编码）
        dropout=0.1      # Dropout概率
    ):
"""

"""
训练 PyTorch Transformer 模型
使用随机数据训练模型，计算损失并更新参数。
"""
def compute_next_token_loss(model, criterion, src, tgt):
    if tgt.size(1) < 2:
        raise ValueError("Target sequences must contain at least two tokens.")

    decoder_input = tgt[:, :-1]
    target = tgt[:, 1:]
    output = model(src, decoder_input)
    loss = criterion(
        output.contiguous().view(-1, output.size(-1)), target.contiguous().view(-1)
    )
    return output, target, loss


def train_step(model, optimizer, criterion, src, tgt):
    model.train()
    optimizer.zero_grad()
    _, _, loss = compute_next_token_loss(model, criterion, src, tgt)
    loss.backward()
    optimizer.step()
    return loss.detach()


def evaluate(model, criterion, src, tgt):
    model.eval()
    with torch.no_grad():
        _, _, loss = compute_next_token_loss(model, criterion, src, tgt)
    return loss


def run_demo(epochs=100, batch_size=64):
    src_vocab_size = 5000
    tgt_vocab_size = 5000
    d_model = 512
    num_heads = 8
    num_layers = 6
    d_ff = 2048
    max_seq_length = 100
    dropout = 0.1

    transformer = Transformer(
        src_vocab_size,
        tgt_vocab_size,
        d_model,
        num_heads,
        num_layers,
        d_ff,
        max_seq_length,
        dropout,
    )
    src_data = torch.randint(1, src_vocab_size, (batch_size, max_seq_length))
    tgt_data = torch.randint(1, tgt_vocab_size, (batch_size, max_seq_length))
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    optimizer = torch.optim.Adam(
        transformer.parameters(), lr=0.001, betas=(0.9, 0.98), eps=1e-9
    )

    for epoch in range(epochs):
        loss = train_step(transformer, optimizer, criterion, src_data, tgt_data)
        print(f"Epoch {epoch + 1}, Loss: {loss.item()}")

    val_src_data = torch.randint(1, src_vocab_size, (batch_size, max_seq_length))
    val_tgt_data = torch.randint(1, tgt_vocab_size, (batch_size, max_seq_length))
    val_loss = evaluate(transformer, criterion, val_src_data, val_tgt_data)
    print(f"Validation Loss: {val_loss.item()}")


if __name__ == "__main__":
    run_demo()
