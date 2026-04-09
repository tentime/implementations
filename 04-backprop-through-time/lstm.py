"""
LSTM cell with explicit gate equations and manual BPTT.
NumPy only.
"""

import numpy as np


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))


def softmax(x):
    e = np.exp(x - x.max())
    return e / e.sum()


class LSTMCell:
    """
    Single LSTM cell with separate weight matrices for each gate.
    This is more verbose than a concatenated form but pedagogically clearer.

    Input convention: concat(h_prev, x_t) — hidden first, then input.
    """

    def __init__(self, input_size, hidden_size):
        self.I = input_size
        self.H = hidden_size
        combined = input_size + hidden_size

        scale = 0.01

        # Forget gate weights (H × combined) + bias (H,)
        self.W_f = np.random.randn(hidden_size, combined) * scale
        self.b_f = np.ones(hidden_size) * 1.0   # init forget bias to 1 (common practice)

        # Input gate
        self.W_i = np.random.randn(hidden_size, combined) * scale
        self.b_i = np.zeros(hidden_size)

        # Candidate cell (gate g)
        self.W_g = np.random.randn(hidden_size, combined) * scale
        self.b_g = np.zeros(hidden_size)

        # Output gate
        self.W_o = np.random.randn(hidden_size, combined) * scale
        self.b_o = np.zeros(hidden_size)

    def forward(self, x_t, h_prev, c_prev):
        """
        One LSTM step.

        x_t    : (I,)
        h_prev : (H,)
        c_prev : (H,)

        Returns:
            h_t    : (H,)
            c_t    : (H,)
            cache  : dict of intermediates needed for backward
        """
        z = np.concatenate([h_prev, x_t])   # (I+H,)

        f_t = sigmoid(self.W_f @ z + self.b_f)    # forget gate
        i_t = sigmoid(self.W_i @ z + self.b_i)    # input gate
        g_t = np.tanh(self.W_g @ z + self.b_g)    # candidate cell
        o_t = sigmoid(self.W_o @ z + self.b_o)    # output gate

        c_t = f_t * c_prev + i_t * g_t
        h_t = o_t * np.tanh(c_t)

        cache = {
            "z": z,
            "f_t": f_t, "i_t": i_t, "g_t": g_t, "o_t": o_t,
            "c_t": c_t, "c_prev": c_prev,
            "h_prev": h_prev,
            "tanh_ct": np.tanh(c_t),
        }
        return h_t, c_t, cache


class LSTM:
    """
    Single-layer LSTM over a sequence with a linear output projection.
    """

    def __init__(self, input_size, hidden_size, output_size):
        self.cell = LSTMCell(input_size, hidden_size)
        self.H = hidden_size
        self.I = input_size
        self.O = output_size

        # Output projection
        self.W_hy = np.random.randn(hidden_size, output_size) * 0.01
        self.by = np.zeros(output_size)

    def forward(self, xs, h0=None, c0=None):
        """
        Run over a sequence.

        xs : list of (I,) arrays
        Returns:
            hs     : list of (H,) hidden states
            cs     : list of (H,) cell states
            caches : list of gate caches (one per step)
        """
        H = self.H
        h = np.zeros(H) if h0 is None else h0
        c = np.zeros(H) if c0 is None else c0

        hs, cs, caches = [], [], []
        for x_t in xs:
            h, c, cache = self.cell.forward(x_t, h, c)
            hs.append(h.copy())
            cs.append(c.copy())
            caches.append(cache)
        return hs, cs, caches

    def output(self, h_t):
        logits = h_t @ self.W_hy + self.by
        return softmax(logits)


def backward_lstm(lstm, xs, ys, states):
    """
    BPTT for the LSTM.

    xs     : list of (I,) one-hot inputs   (length T)
    ys     : list of int class indices     (length T)
    states : (hs, cs, caches) from lstm.forward()

    Returns:
        grads : dict of gradient arrays for all parameters
        loss  : scalar cross-entropy

    This is the educational core: the gradients flow through the output
    projection, then through h_t = o_t * tanh(c_t), then through the
    cell state (the "conveyor belt") via dc_t/dt, bypassing the
    saturating gates and allowing gradients to persist many steps.
    """
    cell = lstm.cell
    hs, cs, caches = states
    T = len(xs)

    # --- Forward loss ---
    loss = 0.0
    ps = []
    for t in range(T):
        p = lstm.output(hs[t])
        ps.append(p)
        loss -= np.log(p[ys[t]] + 1e-8)
    loss /= T

    # --- Gradient accumulators ---
    dW_f = np.zeros_like(cell.W_f)
    dW_i = np.zeros_like(cell.W_i)
    dW_g = np.zeros_like(cell.W_g)
    dW_o = np.zeros_like(cell.W_o)
    db_f = np.zeros_like(cell.b_f)
    db_i = np.zeros_like(cell.b_i)
    db_g = np.zeros_like(cell.b_g)
    db_o = np.zeros_like(cell.b_o)
    dW_hy = np.zeros_like(lstm.W_hy)
    dby   = np.zeros_like(lstm.by)

    dh_next = np.zeros(lstm.H)
    dc_next = np.zeros(lstm.H)

    for t in reversed(range(T)):
        cache = caches[t]
        f_t, i_t, g_t, o_t = cache["f_t"], cache["i_t"], cache["g_t"], cache["o_t"]
        c_t, c_prev, tanh_ct = cache["c_t"], cache["c_prev"], cache["tanh_ct"]
        z = cache["z"]

        # Output-layer gradient (cross-entropy + softmax)
        dy = ps[t].copy()
        dy[ys[t]] -= 1.0
        dy /= T

        dW_hy += np.outer(hs[t], dy)
        dby   += dy

        # dh from output layer + gradient from next timestep
        dh = lstm.W_hy @ dy + dh_next

        # Gradients through h_t = o_t * tanh(c_t)
        do_t = dh * tanh_ct
        dc_t = dh * o_t * (1.0 - tanh_ct ** 2) + dc_next

        # Gradients through c_t = f_t * c_prev + i_t * g_t
        df_t = dc_t * c_prev
        di_t = dc_t * g_t
        dg_t = dc_t * i_t
        dc_next = dc_t * f_t

        # Gradients through gate activations
        df_pre = df_t * f_t * (1.0 - f_t)   # sigmoid'
        di_pre = di_t * i_t * (1.0 - i_t)
        dg_pre = dg_t * (1.0 - g_t ** 2)    # tanh'
        do_pre = do_t * o_t * (1.0 - o_t)

        # Gradients w.r.t. weight matrices
        dW_f += np.outer(df_pre, z)
        dW_i += np.outer(di_pre, z)
        dW_g += np.outer(dg_pre, z)
        dW_o += np.outer(do_pre, z)
        db_f += df_pre
        db_i += di_pre
        db_g += dg_pre
        db_o += do_pre

        # Gradient back to z = [h_prev, x_t] — only h_prev part needed
        dz = (cell.W_f.T @ df_pre
              + cell.W_i.T @ di_pre
              + cell.W_g.T @ dg_pre
              + cell.W_o.T @ do_pre)
        dh_next = dz[:lstm.H]   # first H elements are h_prev

    grads = {
        "W_f": dW_f, "W_i": dW_i, "W_g": dW_g, "W_o": dW_o,
        "b_f": db_f, "b_i": db_i, "b_g": db_g, "b_o": db_o,
        "W_hy": dW_hy, "by": dby,
    }

    # Gradient norm clipping
    total_norm = np.sqrt(sum(np.sum(g ** 2) for g in grads.values()))
    clip = 5.0
    if total_norm > clip:
        scale = clip / (total_norm + 1e-8)
        for k in grads:
            grads[k] *= scale

    return grads, loss


def sgd_lstm(lstm, grads, lr=0.01):
    cell = lstm.cell
    cell.W_f -= lr * grads["W_f"]
    cell.W_i -= lr * grads["W_i"]
    cell.W_g -= lr * grads["W_g"]
    cell.W_o -= lr * grads["W_o"]
    cell.b_f -= lr * grads["b_f"]
    cell.b_i -= lr * grads["b_i"]
    cell.b_g -= lr * grads["b_g"]
    cell.b_o -= lr * grads["b_o"]
    lstm.W_hy -= lr * grads["W_hy"]
    lstm.by   -= lr * grads["by"]
