"""
Unit tests for the transformer implementation.
Run with: python -m pytest test_transformer.py -v
"""

import math
import torch
import torch.nn as nn
import torch.optim as optim
import pytest

from attention import scaled_dot_product_attention, MultiHeadAttention, SinusoidalPositionalEncoding
from transformer import Transformer, make_causal_mask


# ---------------------------------------------------------------------------
# Configuration constants
# ---------------------------------------------------------------------------

BATCH = 2
SEQ_Q = 5
SEQ_K = 7
D_MODEL = 32
NUM_HEADS = 4
D_K = D_MODEL // NUM_HEADS  # 8
D_FF = 64
NUM_LAYERS = 2
SRC_VOCAB = 20
TGT_VOCAB = 20
MAX_LEN = 50


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_attention_output_shape():
    """scaled_dot_product_attention returns correct (batch, heads, seq_q, d_v) shape."""
    Q = torch.randn(BATCH, NUM_HEADS, SEQ_Q, D_K)
    K = torch.randn(BATCH, NUM_HEADS, SEQ_K, D_K)
    V = torch.randn(BATCH, NUM_HEADS, SEQ_K, D_K)

    output, weights = scaled_dot_product_attention(Q, K, V)

    assert output.shape == (BATCH, NUM_HEADS, SEQ_Q, D_K), \
        f"Expected output shape {(BATCH, NUM_HEADS, SEQ_Q, D_K)}, got {output.shape}"
    assert weights.shape == (BATCH, NUM_HEADS, SEQ_Q, SEQ_K), \
        f"Expected weights shape {(BATCH, NUM_HEADS, SEQ_Q, SEQ_K)}, got {weights.shape}"


def test_causal_mask_blocks_future():
    """
    With a causal (lower-triangular) mask, attention weights on future positions
    must be exactly 0.0 (because -inf → 0 after softmax).
    """
    seq_len = 6
    mask = make_causal_mask(seq_len, device=torch.device("cpu"))  # (seq_len, seq_len)
    # Expand to (1, 1, seq_len, seq_len) for the attention function
    mask_4d = mask.unsqueeze(0).unsqueeze(0)

    Q = torch.randn(1, 1, seq_len, D_K)
    K = torch.randn(1, 1, seq_len, D_K)
    V = torch.randn(1, 1, seq_len, D_K)

    _, weights = scaled_dot_product_attention(Q, K, V, mask=mask_4d)
    # weights: (1, 1, seq_len, seq_len)

    # For each query position i, all key positions j > i should have weight 0
    for i in range(seq_len):
        for j in range(i + 1, seq_len):
            w = weights[0, 0, i, j].item()
            assert abs(w) < 1e-6, \
                f"Future position ({i},{j}) should be masked, got weight {w:.6f}"

    # And the lower-triangular weights should be positive (attended)
    for i in range(seq_len):
        for j in range(0, i + 1):
            w = weights[0, 0, i, j].item()
            assert w > 0, \
                f"Past/current position ({i},{j}) should have positive weight, got {w:.6f}"


def test_sinusoidal_pe_shape():
    """SinusoidalPositionalEncoding buffer has shape (1, max_len, d_model)."""
    max_len = 100
    pe_module = SinusoidalPositionalEncoding(D_MODEL, max_len=max_len, dropout=0.0)
    pe = pe_module.pe

    assert pe.shape == (1, max_len, D_MODEL), \
        f"Expected PE shape (1, {max_len}, {D_MODEL}), got {pe.shape}"


def test_sinusoidal_pe_values():
    """
    Check specific PE values from the formula.

    PE(pos=0, 2i=0)   = sin(0 / 10000^0) = sin(0) = 0.0
    PE(pos=0, 2i+1=1) = cos(0 / 10000^0) = cos(0) = 1.0
    PE(pos=1, 2i=0)   = sin(1 / 10000^0) = sin(1) ≈ 0.8415
    """
    pe_module = SinusoidalPositionalEncoding(D_MODEL, max_len=MAX_LEN, dropout=0.0)
    pe = pe_module.pe  # (1, max_len, d_model)

    # pos=0, dim=0 → sin(0) = 0.0
    val_00 = pe[0, 0, 0].item()
    assert abs(val_00 - 0.0) < 1e-5, \
        f"PE[0,0,0] should be sin(0)=0.0, got {val_00:.6f}"

    # pos=0, dim=1 → cos(0) = 1.0
    val_01 = pe[0, 0, 1].item()
    assert abs(val_01 - 1.0) < 1e-5, \
        f"PE[0,0,1] should be cos(0)=1.0, got {val_01:.6f}"

    # pos=1, dim=0 → sin(1 / 10000^(0/D_MODEL)) = sin(1) ≈ 0.8415
    val_10 = pe[0, 1, 0].item()
    expected = math.sin(1.0)
    assert abs(val_10 - expected) < 1e-5, \
        f"PE[0,1,0] should be sin(1)={expected:.4f}, got {val_10:.6f}"


def test_transformer_output_shape():
    """End-to-end forward pass returns (batch, tgt_len, tgt_vocab_size)."""
    model = Transformer(
        src_vocab_size=SRC_VOCAB,
        tgt_vocab_size=TGT_VOCAB,
        d_model=D_MODEL,
        num_heads=NUM_HEADS,
        num_layers=NUM_LAYERS,
        d_ff=D_FF,
        max_len=MAX_LEN,
        dropout=0.0,
    )
    model.eval()

    src = torch.randint(0, SRC_VOCAB, (BATCH, SEQ_K))   # (batch, src_len)
    tgt = torch.randint(0, TGT_VOCAB, (BATCH, SEQ_Q))   # (batch, tgt_len)

    with torch.no_grad():
        logits = model(src, tgt)

    assert logits.shape == (BATCH, SEQ_Q, TGT_VOCAB), \
        f"Expected logits shape {(BATCH, SEQ_Q, TGT_VOCAB)}, got {logits.shape}"


def test_copy_task():
    """
    The transformer should be able to learn to copy a short sequence.
    After 300 gradient steps on a tiny copy task, loss should be < 0.5.

    Task: given [SOS, a, b, c], predict [a, b, c, EOS].
    Vocabulary: 0=PAD, 1=SOS, 2=EOS, 3..12 = ten distinct tokens.
    """
    torch.manual_seed(42)
    VOCAB = 13
    SOS_IDX = 1
    EOS_IDX = 2
    SEQ_LEN = 4   # number of content tokens (excluding SOS/EOS)
    BATCH_SZ = 16

    model = Transformer(
        src_vocab_size=VOCAB,
        tgt_vocab_size=VOCAB,
        d_model=32,
        num_heads=4,
        num_layers=2,
        d_ff=64,
        max_len=20,
        dropout=0.0,
    )
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    model.train()
    final_loss = None
    for step in range(300):
        # Generate random sequences to copy
        content = torch.randint(3, VOCAB, (BATCH_SZ, SEQ_LEN))  # (B, SEQ_LEN)
        sos = torch.full((BATCH_SZ, 1), SOS_IDX, dtype=torch.long)
        eos = torch.full((BATCH_SZ, 1), EOS_IDX, dtype=torch.long)

        src = torch.cat([sos, content, eos], dim=1)       # (B, SEQ_LEN+2)
        tgt_in  = torch.cat([sos, content], dim=1)        # (B, SEQ_LEN+1)
        tgt_out = torch.cat([content, eos], dim=1)        # (B, SEQ_LEN+1)

        optimizer.zero_grad()
        logits = model(src, tgt_in)                       # (B, SEQ_LEN+1, VOCAB)
        loss = criterion(logits.reshape(-1, VOCAB), tgt_out.reshape(-1))
        loss.backward()
        optimizer.step()
        final_loss = loss.item()

    assert final_loss < 0.5, \
        f"Copy task loss should be < 0.5 after 300 steps, got {final_loss:.4f}"


def test_multi_head_attention_shape():
    """MultiHeadAttention output has the same shape as input."""
    mha = MultiHeadAttention(D_MODEL, NUM_HEADS)
    x = torch.randn(BATCH, SEQ_Q, D_MODEL)
    out, weights = mha(x)

    assert out.shape == (BATCH, SEQ_Q, D_MODEL), \
        f"Expected MHA output shape {(BATCH, SEQ_Q, D_MODEL)}, got {out.shape}"
    assert weights.shape == (BATCH, NUM_HEADS, SEQ_Q, SEQ_Q), \
        f"Expected weights shape {(BATCH, NUM_HEADS, SEQ_Q, SEQ_Q)}, got {weights.shape}"
