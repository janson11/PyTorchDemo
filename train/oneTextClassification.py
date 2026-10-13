"""使用 bert-base-chinese 的中文情感分类示例。"""

import os

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LinearLR
from torch.utils.data import DataLoader, Dataset
from transformers import BertForSequenceClassification, BertTokenizer


DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else
    "mps" if torch.backends.mps.is_available() else "cpu"
)
MODEL_NAME = "bert-base-chinese"
NUM_CLASSES = 3
MAX_LEN = 128
BATCH_SIZE = 16
EPOCHS = 5
LR = 2e-5

TRAIN_TEXTS = [
    "这部电影太棒了！", "电影精彩，值得推荐", "画面漂亮，故事很感人", "我非常喜欢这部影片",
    "服务很差，失望", "服务态度太差了", "体验糟糕，再也不来了", "质量很差，非常失望",
    "一般般，没什么特别", "感觉还可以，没有惊喜", "表现普通，谈不上喜欢", "整体中规中矩",
]
TRAIN_LABELS = [2, 2, 2, 2, 0, 0, 0, 0, 1, 1, 1, 1]


class SentimentDataset(Dataset):
    def __init__(self, texts, labels, tokenizer):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, index):
        encoding = self.tokenizer(
            self.texts[index],
            max_length=MAX_LEN,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": torch.tensor(self.labels[index], dtype=torch.long),
        }


def load_model():
    """在线加载模型；可通过 HF_ENDPOINT 指向镜像站点。"""
    endpoint = os.getenv("HF_ENDPOINT")
    if endpoint:
        print(f"使用 Hugging Face endpoint：{endpoint}")
    print(f"正在加载 {MODEL_NAME}，首次运行会下载模型文件...")
    tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
    model = BertForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=NUM_CLASSES,
        hidden_dropout_prob=0.1,
    )
    return tokenizer, model


def predict_sentiment(text, model, tokenizer, id2label):
    model.eval()
    encoding = tokenizer(
        text,
        max_length=MAX_LEN,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
    batch = {key: value.to(DEVICE) for key, value in encoding.items()}
    with torch.inference_mode():
        label_id = model(**batch).logits.argmax(dim=1).item()
    return id2label[label_id]


def main():
    torch.manual_seed(0)
    tokenizer, model = load_model()
    model = model.to(DEVICE)

    # 冻结 BERT 底部 6 层，降低小数据集上的过拟合和训练成本。
    for layer in model.bert.encoder.layer[:6]:
        for parameter in layer.parameters():
            parameter.requires_grad = False

    dataset = SentimentDataset(TRAIN_TEXTS, TRAIN_LABELS, tokenizer)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    no_decay = {"bias", "LayerNorm.weight"}
    optimizer_groups = [
        {
            "params": [
                parameter for name, parameter in model.named_parameters()
                if parameter.requires_grad and not any(key in name for key in no_decay)
            ],
            "weight_decay": 0.01,
        },
        {
            "params": [
                parameter for name, parameter in model.named_parameters()
                if parameter.requires_grad and any(key in name for key in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]
    optimizer = AdamW(optimizer_groups, lr=LR)
    criterion = nn.CrossEntropyLoss()
    scheduler = LinearLR(optimizer, start_factor=0.1, total_iters=max(1, EPOCHS // 2))

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss, correct = 0.0, 0
        for batch in loader:
            batch = {key: value.to(DEVICE) for key, value in batch.items()}
            optimizer.zero_grad()
            logits = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
            ).logits
            loss = criterion(logits, batch["labels"])
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item()
            correct += (logits.argmax(1) == batch["labels"]).sum().item()
        scheduler.step()
        print(f"Epoch {epoch}/{EPOCHS} | Loss: {total_loss / len(loader):.4f} | "
              f"Acc: {correct / len(dataset):.4f}")

    id2label = {0: "负面", 1: "中性", 2: "正面"}
    for text in ("这部电影很精彩", "服务态度太差了"):
        print(f"{text} -> {predict_sentiment(text, model, tokenizer, id2label)}")


if __name__ == "__main__":
    main()
