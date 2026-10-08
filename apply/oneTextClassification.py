import csv
import re
from collections import Counter
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset


DATA_DIR = Path(__file__).resolve().parent.parent / "data"
PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"


def tokenize(text):
    return re.findall(r"\b\w+\b", text.lower())


def read_reviews(path):
    with path.open(newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        if reader.fieldnames is None or not {"text", "label"} <= set(reader.fieldnames):
            raise ValueError(f"{path} must have text and label columns")
        reviews = []
        for row in reader:
            label = row["label"]
            if label not in {"0", "1"}:
                raise ValueError(f"{path} has a label other than 0 or 1: {label!r}")
            reviews.append((row["text"], float(label)))
    if not reviews:
        raise ValueError(f"{path} has no reviews")
    return reviews


def build_vocab(reviews, max_size=25000):
    counts = Counter(token for text, _ in reviews for token in tokenize(text))
    vocab = {PAD_TOKEN: 0, UNK_TOKEN: 1}
    for token, _ in counts.most_common(max_size - len(vocab)):
        vocab[token] = len(vocab)
    return vocab


class ReviewDataset(Dataset):
    def __init__(self, reviews, vocab):
        self.reviews = reviews
        self.vocab = vocab

    def __len__(self):
        return len(self.reviews)

    def __getitem__(self, index):
        text, label = self.reviews[index]
        tokens = tokenize(text)
        ids = [self.vocab.get(token, self.vocab[UNK_TOKEN]) for token in tokens]
        return torch.tensor(ids or [self.vocab[UNK_TOKEN]]), label


def collate_reviews(batch):
    texts, labels = zip(*batch)
    lengths = torch.tensor([len(text) for text in texts])
    padded = nn.utils.rnn.pad_sequence(texts)
    return padded, lengths, torch.tensor(labels, dtype=torch.float32)


class SentimentLSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim=32, hidden_dim=32, n_layers=2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=n_layers, bidirectional=True)
        self.fc = nn.Linear(hidden_dim * 2, 1)
        self.dropout = nn.Dropout(0.2)

    def forward(self, text, text_lengths):
        embedded = self.dropout(self.embedding(text))
        packed = nn.utils.rnn.pack_padded_sequence(
            embedded, text_lengths.cpu(), enforce_sorted=False
        )
        _, (hidden, _) = self.lstm(packed)
        hidden = self.dropout(torch.cat((hidden[-2], hidden[-1]), dim=1))
        return self.fc(hidden).squeeze(1)


def accuracy(logits, labels):
    return ((logits >= 0) == (labels >= 0.5)).float().mean().item()


def run_epoch(model, iterator, criterion, optimizer=None):
    model.train(optimizer is not None)
    total_loss = total_acc = 0.0
    for text, lengths, labels in iterator:
        with torch.set_grad_enabled(optimizer is not None):
            logits = model(text, lengths)
            loss = criterion(logits, labels)
            if optimizer is not None:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        total_loss += loss.item() * len(labels)
        total_acc += accuracy(logits, labels) * len(labels)
    return total_loss / len(iterator.dataset), total_acc / len(iterator.dataset)


def predict_sentiment(model, sentence, vocab):
    tokens = tokenize(sentence)
    ids = [vocab.get(token, vocab[UNK_TOKEN]) for token in tokens]
    text = torch.tensor(ids or [vocab[UNK_TOKEN]]).unsqueeze(1)
    model.eval()
    with torch.no_grad():
        return torch.sigmoid(model(text, torch.tensor([len(text)]))).item()


def main():
    train_path = DATA_DIR / "train.csv"
    test_path = DATA_DIR / "test.csv"
    if not train_path.is_file() or not test_path.is_file():
        raise FileNotFoundError(f"Expected {train_path} and {test_path}")

    torch.manual_seed(0)
    train_reviews = read_reviews(train_path)
    test_reviews = read_reviews(test_path)
    vocab = build_vocab(train_reviews)
    train_loader = DataLoader(
        ReviewDataset(train_reviews, vocab), batch_size=8, shuffle=True,
        collate_fn=collate_reviews,
    )
    test_loader = DataLoader(
        ReviewDataset(test_reviews, vocab), batch_size=8, collate_fn=collate_reviews,
    )

    model = SentimentLSTM(len(vocab))
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.BCEWithLogitsLoss()
    for epoch in range(10):
        train_loss, train_acc = run_epoch(model, train_loader, criterion, optimizer)
        print(f"Epoch {epoch + 1}: train loss={train_loss:.4f}, accuracy={train_acc:.2%}")
    test_loss, test_acc = run_epoch(model, test_loader, criterion)
    print(f"Test: loss={test_loss:.4f}, accuracy={test_acc:.2%}")

    positive_review = "This movie was fantastic! I really enjoyed it."
    negative_review = "The film was terrible and boring."
    print(f"Positive review score: {predict_sentiment(model, positive_review, vocab):.4f}")
    print(f"Negative review score: {predict_sentiment(model, negative_review, vocab):.4f}")


if __name__ == "__main__":
    main()
