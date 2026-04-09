"""
Unit tests for the seq2seq attention components.
Run with: python -m pytest test_attention.py -v
"""

import torch
import pytest

from attention import BahdanauAttention
from encoder import LSTMCell, BidirectionalEncoder


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

ENCODER_DIM = 16
DECODER_DIM = 12
ATTN_DIM = 8
SEQ_LEN = 7
INPUT_SIZE = 10
HIDDEN_SIZE = 8


@pytest.fixture
def attention():
    return BahdanauAttention(ENCODER_DIM, DECODER_DIM, ATTN_DIM)


@pytest.fixture
def encoder():
    return BidirectionalEncoder(INPUT_SIZE, HIDDEN_SIZE)


@pytest.fixture
def lstm_cell():
    return LSTMCell(INPUT_SIZE, HIDDEN_SIZE)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_attention_weights_sum_to_one(attention):
    """Softmax output over seq_len positions must sum to 1.0."""
    encoder_outputs = torch.randn(SEQ_LEN, ENCODER_DIM)
    decoder_hidden = torch.randn(DECODER_DIM)

    _, attn_weights = attention(encoder_outputs, decoder_hidden)

    assert attn_weights.shape == (SEQ_LEN,), \
        f"Expected attn_weights shape ({SEQ_LEN},), got {attn_weights.shape}"
    total = attn_weights.sum().item()
    assert abs(total - 1.0) < 1e-5, \
        f"Attention weights should sum to 1.0, got {total:.6f}"


def test_context_vector_shape(attention):
    """Context vector must have shape (encoder_dim,)."""
    encoder_outputs = torch.randn(SEQ_LEN, ENCODER_DIM)
    decoder_hidden = torch.randn(DECODER_DIM)

    context, attn_weights = attention(encoder_outputs, decoder_hidden)

    assert context.shape == (ENCODER_DIM,), \
        f"Expected context shape ({ENCODER_DIM},), got {context.shape}"


def test_encoder_output_shape(encoder):
    """Bidirectional encoder returns all_hidden (seq_len, 2*hidden) and final_h (2*hidden,)."""
    inputs = torch.randn(SEQ_LEN, INPUT_SIZE)
    all_hidden, final_h = encoder(inputs)

    expected_hidden_dim = 2 * HIDDEN_SIZE
    assert all_hidden.shape == (SEQ_LEN, expected_hidden_dim), \
        f"Expected all_hidden shape ({SEQ_LEN}, {expected_hidden_dim}), got {all_hidden.shape}"
    assert final_h.shape == (expected_hidden_dim,), \
        f"Expected final_h shape ({expected_hidden_dim},), got {final_h.shape}"


def test_lstm_cell_gates_range(lstm_cell):
    """
    Forget, input, and output gates must lie in [0, 1] (they pass through sigmoid).
    We verify this indirectly by checking the cell's output magnitudes:
      - h_t = o_t * tanh(c_t), so |h_t| <= 1
      - c_t is unbounded in theory, but for random init shouldn't explode in one step
    We also directly test the gate values by temporarily inspecting them.
    """
    torch.manual_seed(0)
    x = torch.randn(INPUT_SIZE)
    h = torch.randn(HIDDEN_SIZE)
    c = torch.randn(HIDDEN_SIZE)

    # Run the cell
    h_new, c_new = lstm_cell(x, h, c)

    assert h_new.shape == (HIDDEN_SIZE,)
    assert c_new.shape == (HIDDEN_SIZE,)

    # h_t = o_t * tanh(c_t); since o_t in [0,1] and tanh in [-1,1], |h_t| <= 1
    assert h_new.abs().max().item() <= 1.0 + 1e-6, \
        f"h_t values should be in [-1, 1], max abs was {h_new.abs().max().item():.4f}"

    # Verify gates directly by computing them the same way
    combined = torch.cat([h, x], dim=-1)
    f_t = torch.sigmoid(lstm_cell.Wf(combined))
    i_t = torch.sigmoid(lstm_cell.Wi(combined))
    o_t = torch.sigmoid(lstm_cell.Wo(combined))

    for name, gate in [("forget", f_t), ("input", i_t), ("output", o_t)]:
        assert gate.min().item() >= 0.0 - 1e-6 and gate.max().item() <= 1.0 + 1e-6, \
            f"{name} gate values should be in [0, 1], got min={gate.min():.4f} max={gate.max():.4f}"


def test_attention_is_differentiable(attention):
    """Gradients flow back through the attention mechanism."""
    encoder_outputs = torch.randn(SEQ_LEN, ENCODER_DIM, requires_grad=True)
    decoder_hidden = torch.randn(DECODER_DIM, requires_grad=True)

    context, _ = attention(encoder_outputs, decoder_hidden)
    loss = context.sum()
    loss.backward()

    assert encoder_outputs.grad is not None, "No gradient for encoder_outputs"
    assert decoder_hidden.grad is not None, "No gradient for decoder_hidden"


def test_attention_different_inputs_differ(attention):
    """Different decoder hidden states should produce different attention distributions."""
    torch.manual_seed(1)
    encoder_outputs = torch.randn(SEQ_LEN, ENCODER_DIM)
    h1 = torch.randn(DECODER_DIM)
    h2 = torch.randn(DECODER_DIM)

    _, w1 = attention(encoder_outputs, h1)
    _, w2 = attention(encoder_outputs, h2)

    assert not torch.allclose(w1, w2), \
        "Different decoder states should produce different attention weights"
