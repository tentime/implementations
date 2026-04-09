"""
Tests for the GPT implementation.
Run with: python -m pytest test_gpt.py -v
"""

import torch
import torch.nn as nn
import pytest
from gpt import GPT


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

VOCAB_SIZE = 50
D_MODEL = 32
NUM_HEADS = 4
NUM_LAYERS = 2
MAX_LEN = 64


def make_model(dropout=0.0):
    return GPT(
        vocab_size=VOCAB_SIZE,
        d_model=D_MODEL,
        num_heads=NUM_HEADS,
        num_layers=NUM_LAYERS,
        max_len=MAX_LEN,
        dropout=dropout,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_causal_mask_is_lower_triangular():
    """The mask buffer is torch.tril(torch.ones(max_len, max_len))."""
    model = make_model()
    # The mask lives on the CausalSelfAttention in each GPTBlock
    attn = model.blocks[0].attn
    mask = attn.mask
    expected = torch.tril(torch.ones(MAX_LEN, MAX_LEN))
    assert mask.shape == (MAX_LEN, MAX_LEN), (
        f"Expected mask shape ({MAX_LEN}, {MAX_LEN}), got {mask.shape}"
    )
    assert torch.allclose(mask, expected), "Mask is not lower-triangular"


def test_output_shape():
    """GPT.forward on (batch, seq_len) gives (batch, seq_len, vocab_size)."""
    model = make_model()
    batch, seq_len = 4, 20
    idx = torch.randint(0, VOCAB_SIZE, (batch, seq_len))
    logits = model(idx)
    assert logits.shape == (batch, seq_len, VOCAB_SIZE), (
        f"Expected {(batch, seq_len, VOCAB_SIZE)}, got {logits.shape}"
    )


def test_generation_length():
    """generate(prompt, max_new_tokens=20) appends exactly 20 new tokens."""
    model = make_model()
    prompt_len = 5
    max_new = 20
    prompt = torch.randint(0, VOCAB_SIZE, (1, prompt_len))
    output = model.generate(prompt, max_new_tokens=max_new, temperature=1.0)
    assert output.shape == (1, prompt_len + max_new), (
        f"Expected length {prompt_len + max_new}, got {output.shape[1]}"
    )


def test_temperature_greedy():
    """temperature=0 (argmax) always produces the same output for the same prompt."""
    model = make_model()
    prompt = torch.randint(0, VOCAB_SIZE, (1, 5))
    out1 = model.generate(prompt.clone(), max_new_tokens=10, temperature=0)
    out2 = model.generate(prompt.clone(), max_new_tokens=10, temperature=0)
    assert torch.equal(out1, out2), (
        "Greedy generation (temperature=0) should be deterministic but gave different outputs"
    )


def test_loss_decreases():
    """After 100 steps on a single repeated sequence, loss decreases."""
    torch.manual_seed(0)
    model = make_model()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()

    # A single short sequence, repeated as a batch
    seq = torch.randint(0, VOCAB_SIZE, (1, 32))
    x = seq[:, :-1]   # input: all but last token
    y = seq[:, 1:]    # target: all but first token

    losses = []
    model.train()
    for _ in range(100):
        logits = model(x)
        loss = loss_fn(logits.view(-1, VOCAB_SIZE), y.view(-1))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    first_loss = sum(losses[:5]) / 5     # average of first 5
    last_loss = sum(losses[-5:]) / 5     # average of last 5
    assert last_loss < first_loss, (
        f"Loss did not decrease: first={first_loss:.4f}, last={last_loss:.4f}"
    )


def test_weight_tying():
    """model.lm_head.weight IS model.token_embedding.weight (same Python object)."""
    model = make_model()
    assert model.lm_head.weight is model.token_embedding.weight, (
        "Weight tying broken: lm_head.weight is not the same tensor as token_embedding.weight"
    )
