"""
Pytest tests for rnn.py, lstm.py, and gru.py.
"""

import numpy as np
import pytest

from rnn import VanillaRNN, bptt
from lstm import LSTMCell, LSTM, backward_lstm
from gru import GRUCell, GRU


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

INPUT_SIZE  = 10
HIDDEN_SIZE = 16
OUTPUT_SIZE = 10
SEQ_LEN     = 10


def make_random_sequence(T=SEQ_LEN, I=INPUT_SIZE):
    xs = [np.random.randn(I) for _ in range(T)]
    ys = list(np.random.randint(0, OUTPUT_SIZE, size=T))
    return xs, ys


# ---------------------------------------------------------------------------
# test_lstm_gates_sigmoid_range
# ---------------------------------------------------------------------------

def test_lstm_gates_sigmoid_range():
    np.random.seed(0)
    cell = LSTMCell(INPUT_SIZE, HIDDEN_SIZE)
    x_t = np.random.randn(INPUT_SIZE)
    h_prev = np.random.randn(HIDDEN_SIZE)
    c_prev = np.random.randn(HIDDEN_SIZE)

    _, _, cache = cell.forward(x_t, h_prev, c_prev)

    for gate_name in ("f_t", "i_t", "o_t"):
        gate = cache[gate_name]
        assert np.all(gate >= 0.0) and np.all(gate <= 1.0), (
            f"Gate '{gate_name}' has values outside [0, 1]: min={gate.min():.4f}, max={gate.max():.4f}"
        )


# ---------------------------------------------------------------------------
# test_gru_output_shape
# ---------------------------------------------------------------------------

def test_gru_output_shape():
    np.random.seed(1)
    cell = GRUCell(INPUT_SIZE, HIDDEN_SIZE)
    x_t = np.random.randn(INPUT_SIZE)
    h_prev = np.random.randn(HIDDEN_SIZE)

    h_t, _ = cell.forward(x_t, h_prev)
    assert h_t.shape == (HIDDEN_SIZE,), (
        f"Expected h_t shape ({HIDDEN_SIZE},), got {h_t.shape}"
    )


# ---------------------------------------------------------------------------
# test_bptt_gradient_clip
# ---------------------------------------------------------------------------

def test_bptt_gradient_clip():
    np.random.seed(2)
    rnn = VanillaRNN(INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE)
    xs, ys = make_random_sequence()
    h0 = np.zeros(HIDDEN_SIZE)

    grads, _ = bptt(rnn, xs, ys, h0, truncate=25)

    total_norm = np.sqrt(sum(np.sum(g ** 2) for g in grads.values()))
    assert total_norm <= 5.0 + 1e-7, (
        f"Gradient norm after clipping should be ≤ 5.0, got {total_norm:.6f}"
    )


# ---------------------------------------------------------------------------
# test_lstm_output_shape
# ---------------------------------------------------------------------------

def test_lstm_output_shape():
    np.random.seed(3)
    lstm = LSTM(INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE)
    xs = [np.random.randn(INPUT_SIZE) for _ in range(SEQ_LEN)]

    hs, cs, caches = lstm.forward(xs)

    assert len(hs) == SEQ_LEN, f"Expected {SEQ_LEN} hidden states, got {len(hs)}"
    for t, h in enumerate(hs):
        assert h.shape == (HIDDEN_SIZE,), (
            f"Hidden state at t={t} has shape {h.shape}, expected ({HIDDEN_SIZE},)"
        )
