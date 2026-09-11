import subprocess
import sys
from pathlib import Path

import pytest
import torch

from apply.TransformerModel import (
    MultiHeadAttention,
    PositionalEncoding,
    Transformer,
    compute_next_token_loss,
    evaluate,
    train_step,
)


def make_model(max_seq_length=6):
    return Transformer(
        src_vocab_size=32,
        tgt_vocab_size=32,
        d_model=8,
        num_heads=2,
        num_layers=1,
        d_ff=16,
        max_seq_length=max_seq_length,
        dropout=0.0,
    )


def make_batch(device="cpu"):
    src = torch.randint(1, 32, (2, 5), device=device)
    tgt = torch.randint(1, 32, (2, 6), device=device)
    return src, tgt


def test_split_and_combine_heads_preserve_default_model_dimension():
    attention = MultiHeadAttention(d_model=512, num_heads=8)
    x = torch.randn(2, 3, 512)

    split = attention.split_heads(x)

    assert split.shape == (2, 8, 3, 64)
    torch.testing.assert_close(attention.combine_heads(split), x)


def test_attention_rejects_non_divisible_head_configuration():
    with pytest.raises(AssertionError, match="d_model"):
        MultiHeadAttention(d_model=10, num_heads=3)


def test_attention_supports_self_and_cross_attention_with_finite_outputs():
    attention = MultiHeadAttention(d_model=8, num_heads=2)
    query = torch.randn(2, 4, 8)
    key_value = torch.randn(2, 5, 8)
    mask = torch.ones(2, 1, 1, 5, dtype=torch.bool)

    self_output = attention(query, query, query, torch.ones(2, 1, 1, 4, dtype=torch.bool))
    cross_output = attention(query, key_value, key_value, mask)

    assert self_output.shape == (2, 4, 8)
    assert cross_output.shape == (2, 4, 8)
    assert torch.isfinite(self_output).all()
    assert torch.isfinite(cross_output).all()


def test_transformer_projects_logits_and_uses_encoder_feed_forward():
    model = make_model()
    src, tgt = make_batch()

    output = model(src, tgt[:, :-1])
    output.sum().backward()

    assert output.shape == (2, 5, 32)
    gradient = model.encoder_layer[0].feed_forward.fc1.weight.grad
    assert gradient is not None
    assert torch.isfinite(gradient).all()


def test_next_token_training_and_validation_use_right_shifted_labels():
    model = make_model()
    src, tgt = make_batch()
    criterion = torch.nn.CrossEntropyLoss(ignore_index=0)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    initial_weight = model.output_projection.weight.detach().clone()

    output, target, loss = compute_next_token_loss(model, criterion, src, tgt)
    train_loss = train_step(model, optimizer, criterion, src, tgt)
    validation_loss = evaluate(model, criterion, src, tgt)

    assert output.shape == (2, 5, 32)
    torch.testing.assert_close(target, tgt[:, 1:])
    assert output.reshape(-1, 32).shape[0] == target.numel()
    assert torch.isfinite(loss)
    assert torch.isfinite(train_loss)
    assert torch.isfinite(validation_loss)
    assert not torch.equal(initial_weight, model.output_projection.weight)


def test_masks_are_boolean_causal_and_match_input_device():
    model = make_model()
    src, tgt = make_batch()

    src_mask, tgt_mask = model.generate_mask(src, tgt)

    assert src_mask.shape == (2, 1, 1, 5)
    assert tgt_mask.shape == (2, 1, 6, 6)
    assert src_mask.dtype is torch.bool
    assert tgt_mask.dtype is torch.bool
    assert src_mask.device == src.device
    assert tgt_mask.device == tgt.device
    assert not tgt_mask[0, 0].triu(diagonal=1).any()


def test_positional_encoding_checks_maximum_sequence_length():
    encoding = PositionalEncoding(d_model=8, max_seq_length=3)

    assert encoding(torch.zeros(1, 3, 8)).shape == (1, 3, 8)
    with pytest.raises(ValueError, match="exceeds max_seq_length 3"):
        encoding(torch.zeros(1, 4, 8))


def test_module_import_has_no_training_side_effects():
    root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [sys.executable, "-c", "import apply.TranformerModel"],
        cwd=root,
        capture_output=True,
        text=True,
        check=True,
    )

    assert result.stdout == ""
    assert result.stderr == ""


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_masks_and_forward_pass_support_cuda():
    model = make_model().cuda()
    src, tgt = make_batch(device="cuda")

    src_mask, tgt_mask = model.generate_mask(src, tgt)
    output = model(src, tgt[:, :-1])

    assert src_mask.device.type == "cuda"
    assert tgt_mask.device.type == "cuda"
    assert output.device.type == "cuda"
