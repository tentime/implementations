"""
Tests for the BERT MLM implementation.
Run with: python -m pytest test_bert.py -v
"""

import torch
import torch.nn as nn
import pytest
from bert import BertMLM, mask_tokens, MASK_ID, PAD_ID


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

VOCAB_SIZE = 100
D_MODEL = 32
NUM_HEADS = 4
NUM_LAYERS = 2
D_FF = 64
MAX_LEN = 64


def make_model():
    return BertMLM(
        vocab_size=VOCAB_SIZE,
        d_model=D_MODEL,
        num_heads=NUM_HEADS,
        num_layers=NUM_LAYERS,
        d_ff=D_FF,
        max_len=MAX_LEN,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_mask_token_ratio():
    """Exactly 15% of tokens selected for prediction (labels != -100)."""
    seq_len = 100
    # Use a large batch so the ratio is close to 0.15
    token_ids = torch.randint(4, VOCAB_SIZE, (50, seq_len))
    _, labels = mask_tokens(token_ids, VOCAB_SIZE, mask_prob=0.15, seed=42)

    total = token_ids.numel()
    selected = (labels != -100).sum().item()
    ratio = selected / total

    # Allow ±2% tolerance (stochastic, but with 5000 tokens it should be tight)
    assert abs(ratio - 0.15) < 0.02, f"Expected ~15% selected, got {ratio:.3f}"


def test_mask_strategy_distribution():
    """
    Of selected positions: ~80% MASK_ID, ~10% random, ~10% unchanged.
    Run mask_tokens 1000 times on a 100-token sequence, check distributions.
    Each bucket should be within 5% of expected.
    """
    seq_len = 100
    runs = 1000
    mask_count = 0
    random_count = 0
    unchanged_count = 0
    total_selected = 0

    # Use a fixed base so random replacement is detectable: all tokens are the same id
    base_id = 10
    token_ids = torch.full((1, seq_len), base_id, dtype=torch.long)

    for i in range(runs):
        masked, labels = mask_tokens(token_ids, VOCAB_SIZE, mask_prob=0.15, seed=None)
        selected_mask = labels != -100
        selected_original = token_ids[selected_mask]
        selected_output = masked[selected_mask]

        n = selected_mask.sum().item()
        total_selected += n

        for orig, out in zip(selected_original.tolist(), selected_output.tolist()):
            if out == MASK_ID:
                mask_count += 1
            elif out == orig:
                unchanged_count += 1
            else:
                random_count += 1

    total = total_selected
    mask_frac = mask_count / total
    random_frac = random_count / total
    unchanged_frac = unchanged_count / total

    assert abs(mask_frac - 0.80) < 0.05, f"MASK fraction: expected ~0.80, got {mask_frac:.3f}"
    assert abs(random_frac - 0.10) < 0.05, f"random fraction: expected ~0.10, got {random_frac:.3f}"
    assert abs(unchanged_frac - 0.10) < 0.05, f"unchanged fraction: expected ~0.10, got {unchanged_frac:.3f}"


def test_weight_tying():
    """
    model.token_embedding.embedding.weight IS model.mlm_head.projection.weight
    (same Python object, same data pointer).
    """
    model = make_model()
    assert model.mlm_head.projection.weight is model.token_embedding.embedding.weight, (
        "Weight tying broken: MLMHead.projection.weight is not the same tensor object "
        "as TokenEmbedding.embedding.weight"
    )


def test_output_shape():
    """forward returns (batch, seq_len, vocab_size)."""
    model = make_model()
    batch, seq_len = 3, 20
    token_ids = torch.randint(4, VOCAB_SIZE, (batch, seq_len))
    logits = model(token_ids)
    assert logits.shape == (batch, seq_len, VOCAB_SIZE), (
        f"Expected shape {(batch, seq_len, VOCAB_SIZE)}, got {logits.shape}"
    )


def test_loss_on_masked_only():
    """
    When labels has -100 for all positions except one,
    cross-entropy loss is nonzero (because one position is masked).
    """
    model = make_model()
    batch, seq_len = 1, 20
    token_ids = torch.randint(4, VOCAB_SIZE, (batch, seq_len))

    # Manually construct labels with only position 5 unmasked
    labels = torch.full_like(token_ids, -100)
    labels[0, 5] = token_ids[0, 5].item()

    logits = model(token_ids)  # (1, seq_len, vocab_size)
    loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
    loss = loss_fn(logits.view(-1, VOCAB_SIZE), labels.view(-1))

    assert loss.item() > 0.0, "Loss should be nonzero when one position is masked"
    # Also make sure it's a finite scalar
    assert torch.isfinite(loss), "Loss should be finite"
