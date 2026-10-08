import torch

from apply.oneTextClassification import (
    ReviewDataset,
    SentimentLSTM,
    build_vocab,
    collate_reviews,
    predict_sentiment,
    read_reviews,
    run_epoch,
)


def test_csv_reviews_support_unknown_words_and_variable_lengths(tmp_path):
    path = tmp_path / "reviews.csv"
    path.write_text("text,label\nGreat movie,1\nBad,0\n", encoding="utf-8")

    reviews = read_reviews(path)
    vocab = build_vocab(reviews)
    dataset = ReviewDataset(reviews, vocab)
    text, lengths, labels = collate_reviews([dataset[1], dataset[0]])

    assert text.shape == (2, 2)
    assert lengths.tolist() == [1, 2]
    assert text[1, 0].item() == vocab["<pad>"]
    assert labels.tolist() == [0.0, 1.0]
    assert ReviewDataset([("Unseen", 1)], vocab)[0][0].tolist() == [vocab["<unk>"]]


def test_lstm_trains_on_unsorted_batches():
    reviews = [("bad", 0), ("very good", 1)]
    vocab = build_vocab(reviews)
    dataset = ReviewDataset(reviews, vocab)
    loader = torch.utils.data.DataLoader(dataset, batch_size=2, collate_fn=collate_reviews)
    model = SentimentLSTM(len(vocab), embedding_dim=4, hidden_dim=4)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    initial_weight = model.fc.weight.detach().clone()

    loss, accuracy = run_epoch(model, loader, torch.nn.BCEWithLogitsLoss(), optimizer)

    assert torch.isfinite(torch.tensor(loss))
    assert 0 <= accuracy <= 1
    assert not torch.equal(initial_weight, model.fc.weight)
    assert 0 <= predict_sentiment(model, "unknown word", vocab) <= 1
