import csv
import re
from collections import Counter
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pack_padded_sequence, pad_sequence
from torch.optim.lr_scheduler import ReduceLROnPlateau

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
DATA_DIR = Path(__file__).resolve().parent.parent / "data"
CHECKPOINT = Path(__file__).resolve().parent / "checkpoints" / "best_bilstm.pth"


def tokenize(text):
    return re.findall(r"\b\w+\b", text.lower())


def read_reviews(path):
    with path.open(newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        if reader.fieldnames is None or not {"text", "label"} <= set(reader.fieldnames):
            raise ValueError(f"{path} must have text and label columns")
        reviews = []
        for row in reader:
            if row["label"] not in {"0", "1"}:
                raise ValueError(f"{path} has an invalid label: {row['label']!r}")
            reviews.append((row["text"], int(row["label"])))
    if not reviews:
        raise ValueError(f"{path} has no reviews")
    return reviews


def build_vocab(reviews, max_size=50000):
    counts = Counter(token for text, _ in reviews for token in tokenize(text))
    vocab = {"<pad>": 0, "<unk>": 1}
    for token, _ in counts.most_common(max_size - len(vocab)):
        vocab[token] = len(vocab)
    return vocab


# -----模型定义-----------------
class BiLSTMClassifier(nn.Module):
    """
    双向 LSTM 文本分类模型
    结构：Embedding -> BiLSTM -> Dropout -> FC
    """

    def __init__(self, vocab_size, embed_dim, hidden_size, num_layers, num_classes, dropout=0.5, padding_idx=0):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=padding_idx)

        # 使用预训练词向量时：
        # self.embedding = nn.Embedding.from_pretrained(pretrained_vectors)

        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )
        self.dropout = nn.Dropout(dropout)

        # 双向LSTM:拼接正向和反向最后隐藏状态
        self.fc = nn.Linear(hidden_size * 2, num_classes)

    def forward(self, x, lengths):
        embedded = self.dropout(self.embedding(x))
        packed = pack_padded_sequence(embedded, lengths.cpu(), batch_first=True, enforce_sorted=False)

        _, (h_n, _) = self.lstm(packed)
        # h_n: (num_layers * 2, batch_size, hidden_size)
        # 取最后一层的正向和反向隐藏状态并拼接
        h_forward = h_n[-2, :, :]
        h_backward = h_n[-1, :, :]
        h_combined = torch.cat([h_forward, h_backward], dim=-1)
        out = self.dropout(h_combined)
        return self.fc(out)


# -------------自定义Dataset
class TextDataset(Dataset):
    def __init__(self, texts, labels, vocab, max_len=200):
        self.texts = texts
        self.labels = labels
        self.vocab = vocab
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        tokens = tokenize(self.texts[idx])[:self.max_len]
        ids = [self.vocab.get(t, 1) for t in tokens] or [1]
        return torch.tensor(ids, dtype=torch.long), torch.tensor(self.labels[idx], dtype=torch.long)

    @staticmethod
    def collate_fn(batch):
        """自定义 collate：对变长序列补零，记录真实长度"""
        sequences, labels = zip(*batch)
        lengths = torch.tensor([len(s) for s in sequences])
        padded = pad_sequence(sequences, batch_first=True, padding_value=0)
        labels = torch.stack(labels)
        return padded, lengths, labels

def train_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss, correct = 0.0, 0
    for texts, lengths, labels in loader:
        texts, labels = texts.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(texts, lengths)
        loss = criterion(outputs, labels)
        loss.backward()
        # 梯度裁剪，防止RNN的梯度爆炸
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item() * len(labels)
        correct += (outputs.argmax(1) == labels).sum().item()

    n = len(loader.dataset)
    return total_loss / n, correct / n


def eval_epoch(model, loader, criterion):
    model.eval()
    total_loss, correct = 0.0, 0
    with torch.no_grad():
        for texts, lengths, labels in loader:
            texts, labels = texts.to(device), labels.to(device)
            outputs = model(texts, lengths)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * len(labels)
            correct += (outputs.argmax(1) == labels).sum().item()
    n = len(loader.dataset)
    return total_loss / n, correct / n

# ---初始化与训练
EMBED_DIM   = 128
HIDDEN_SIZE = 256
NUM_LAYERS  = 2
NUM_CLASSES = 2
DROPOUT     = 0.5
EPOCHS      = 15

def main():
    torch.manual_seed(0)
    train_reviews = read_reviews(DATA_DIR / "train.csv")
    val_reviews = read_reviews(DATA_DIR / "test.csv")
    vocab = build_vocab(train_reviews)
    train_loader = DataLoader(
        TextDataset(*zip(*train_reviews), vocab), batch_size=8, shuffle=True,
        collate_fn=TextDataset.collate_fn,
    )
    val_loader = DataLoader(
        TextDataset(*zip(*val_reviews), vocab), batch_size=8,
        collate_fn=TextDataset.collate_fn,
    )

    model = BiLSTMClassifier(len(vocab), EMBED_DIM, HIDDEN_SIZE,
                             NUM_LAYERS, NUM_CLASSES, DROPOUT).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    scheduler = ReduceLROnPlateau(optimizer, mode="max", patience=3, factor=0.5)

    best_acc = -1.0
    for epoch in range(1, EPOCHS + 1):
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion)
        val_loss, val_acc = eval_epoch(model, val_loader, criterion)
        scheduler.step(val_acc)

        print(f"Epoch {epoch:2d}/{EPOCHS} | "
              f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f}")

        if val_acc > best_acc:
            best_acc = val_acc
            CHECKPOINT.parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), CHECKPOINT)
            print(f"  -> 保存最优模型到 {CHECKPOINT}，Val Acc: {best_acc:.4f}")


if __name__ == "__main__":
    main()
