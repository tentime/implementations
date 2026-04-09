"""
GRU cell with explicit gate equations.
NumPy only.
"""

import numpy as np


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))


def softmax(x):
    e = np.exp(x - x.max())
    return e / e.sum()


class GRUCell:
    """
    Gated Recurrent Unit — Cho et al. 2014.

    Equations (concat convention: [h_prev, x_t]):
        z_t    = sigmoid(W_z @ [h_prev, x_t] + b_z)       update gate
        r_t    = sigmoid(W_r @ [h_prev, x_t] + b_r)       reset gate
        h_cand = tanh(W_h @ [r_t * h_prev, x_t] + b_h)   candidate state
        h_t    = (1 - z_t) * h_prev + z_t * h_cand
    """

    def __init__(self, input_size, hidden_size):
        self.I = input_size
        self.H = hidden_size
        combined = input_size + hidden_size
        scale = 0.01

        # Update gate (H × combined)
        self.W_z = np.random.randn(hidden_size, combined) * scale
        self.b_z = np.zeros(hidden_size)

        # Reset gate
        self.W_r = np.random.randn(hidden_size, combined) * scale
        self.b_r = np.zeros(hidden_size)

        # Candidate (note: uses r_t * h_prev concat x_t, so still combined size)
        self.W_h = np.random.randn(hidden_size, combined) * scale
        self.b_h = np.zeros(hidden_size)

    def forward(self, x_t, h_prev):
        """
        x_t    : (I,)
        h_prev : (H,)
        Returns h_t : (H,), cache for backward
        """
        z_input = np.concatenate([h_prev, x_t])     # (I+H,) — for z and r gates

        z_t = sigmoid(self.W_z @ z_input + self.b_z)
        r_t = sigmoid(self.W_r @ z_input + self.b_r)

        # Candidate uses reset gate applied to h_prev
        h_input = np.concatenate([r_t * h_prev, x_t])   # (I+H,)
        h_cand = np.tanh(self.W_h @ h_input + self.b_h)

        h_t = (1.0 - z_t) * h_prev + z_t * h_cand

        cache = {
            "z_t": z_t, "r_t": r_t, "h_cand": h_cand,
            "h_prev": h_prev, "x_t": x_t,
            "z_input": z_input, "h_input": h_input,
        }
        return h_t, cache


class GRU:
    """
    Single-layer GRU over a sequence with output projection.
    """

    def __init__(self, input_size, hidden_size, output_size):
        self.cell = GRUCell(input_size, hidden_size)
        self.H = hidden_size
        self.I = input_size
        self.O = output_size

        self.W_hy = np.random.randn(hidden_size, output_size) * 0.01
        self.by = np.zeros(output_size)

    def forward(self, xs, h0=None):
        """
        xs : list of (I,) arrays
        Returns:
            hs     : list of (H,) hidden states
            caches : list of per-step caches
        """
        h = np.zeros(self.H) if h0 is None else h0
        hs, caches = [], []
        for x_t in xs:
            h, cache = self.cell.forward(x_t, h)
            hs.append(h.copy())
            caches.append(cache)
        return hs, caches

    def output(self, h_t):
        logits = h_t @ self.W_hy + self.by
        return softmax(logits)
