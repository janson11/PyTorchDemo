# 文本分类实战
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import (
    BertTokenizer,
    BertForSequenceClassification,
    get_linear_schedule_with_warmup,
)

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
MODEL_NAME = "bert-base-chinese"
NUM_CLASSES = 3  # 情感分类：正/中/负
MAX_LEN = 128
BATCH_SIZE = 16
EPOCHS = 5
LR = 2e-5  # BERT 微调的推荐学习率范围：1e-5 ~ 5e-5


# 自定义 Dataset
class SentimentDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            'labels': torch.tensor(self.labels[idx], dtype=torch.long),
        }


# 加载 BERT预训练模型
tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
model = BertForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=NUM_CLASSES,
    hidden_dropout_prob=0.1,
)
model = model.to(DEVICE)

# 冻结底部N层(可选)
# BERT-base 有12层Tranformer，可以冻结前几层节省计算
FREEZE_LAYERS = 6  # 冻结前6层

for i, layer in enumerate(model.bert.encoder.layer):
    if i < FREEZE_LAYERS:
        for param in layer.parameters():
            param.requires_grad = False

print(f"已冻结前 {FREEZE_LAYERS} 层，减少约 {FREEZE_LAYERS / 12 * 100:.0f}% 的参数更新")

# 数据加载
# 示例数据（实际替换为真实数据集）
train_texts = ["这部电影太棒了！", "服务很差，失望", "一般般，没什么特别"]
train_labels = [2, 0, 1]  # 0:负 1:中 2:正

train_dataset = SentimentDataset(train_texts, train_labels, tokenizer, max_len=MAX_LEN)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

# 优化器与调度器
# BERT 标配 AdamW + 线性 warmup
no_decay = ['bias', 'LayerNorm.weight']
optimizer_grouped = [
    {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
    {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
]
optimizer = AdamW(optimizer_grouped, lr=LR)

total_step = len(train_loader) * EPOCHS
warmup_steps = int(total_step * 0.1)  # 10% 的步数用于预热
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_step)

# 训练循环
for epoch in range(1, EPOCHS + 1):
    model.train()
    total_loss, correct = 0.0, 0

    for batch in train_loader:
        input_ids = batch["input_ids"].to(DEVICE)
        attention_mask = batch["attention_mask"].to(DEVICE)
        labels = batch["labels"].to(DEVICE)

        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)

        loss = outputs.loss  # BERT 内部已计算 CE loss
        logits = outputs.logits

        loss.backward()
        # BERT 微调标配:梯度裁剪，防止梯度爆炸
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()
        correct += (logits.argmax(1) == labels).sum().item()

    avg_loss = total_loss / len(train_loader)
    acc = correct / len(train_loader.dataset)
    print(f"Epoch {epoch}/{EPOCHS} | Loss: {avg_loss:.4f} | Acc: {acc:.4f}")


# 推理
def predict_sentiment(text, model, tokenizer, id2label):
    model.eval()
    encoding = tokenizer(
        text, max_length=MAX_LEN, padding="max_length", truncation=True, return_tensors="pt",
    )
