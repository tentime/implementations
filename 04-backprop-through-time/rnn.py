"""
Vanilla RNN with manual Backpropagation Through Time (BPTT).
NumPy only.
"""

import numpy as np


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))


def softmax(x):
    e = np.exp(x - x.max())
    return e / e.sum()


class VanillaRNN:
    def __init__(self, input_size, hidden_size, output_size):
        self.I = input_size
        self.H = hidden_size
        self.O = output_size

        scale_h = 0.01
        scale_x = 0.01

        # Recurrent weight matrix (H × H)
        self.W_hh = np.random.randn(hidden_size, hidden_size) * scale_h
        # Input-to-hidden (I × H) — note: x @ W_xh gives (H,)
        self.W_xh = np.random.randn(input_size, hidden_size) * scale_x
        # Hidden-to-output (H × O)
        self.W_hy = np.random.randn(hidden_size, output_size) * scale_x

        self.bh = np.zeros(hidden_size)
        self.by = np.zeros(output_size)

    def forward_step(self, x_t, h_prev):
        """
        One RNN step.
        x_t    : (I,)
        h_prev : (H,)
        Returns h_t : (H,)
        """
        pre_act = x_t @ self.W_xh + h_prev @ self.W_hh + self.bh
        h_t = np.tanh(pre_act)
        return h_t

    def forward(self, xs):
        """
        Run forward over a sequence.
        xs : list of (I,) arrays
        Returns list of (H,) hidden states h_1 … h_T
        """
        h = np.zeros(self.H)
        hs = []
        for x_t in xs:
            h = self.forward_step(x_t, h)
            hs.append(h.copy())
        return hs

    def output(self, h_t):
        """Linear projection + softmax for character prediction."""
        logits = h_t @ self.W_hy + self.by
        return softmax(logits)


def bptt(rnn, xs, ys, h0, truncate=25):
    """
    Full BPTT for VanillaRNN with gradient norm clipping.

    xs : list of (I,) one-hot input vectors  (length T)
    ys : list of int class indices            (length T)
    h0 : (H,) initial hidden state
    truncate : maximum steps to unroll (for efficiency; use T for full BPTT)

    Returns:
        grads : dict of gradient arrays for each parameter
        loss  : scalar cross-entropy loss
    """
    T = len(xs)

    # --- Forward pass ---
    hs = [None] * (T + 1)
    hs[0] = h0.copy()
    ps = [None] * T
    loss = 0.0

    for t in range(T):
        hs[t + 1] = rnn.forward_step(xs[t], hs[t])
        ps[t] = rnn.output(hs[t + 1])
        loss -= np.log(ps[t][ys[t]] + 1e-8)

    loss /= T

    # --- Backward pass ---
    dW_hh = np.zeros_like(rnn.W_hh)
    dW_xh = np.zeros_like(rnn.W_xh)
    dW_hy = np.zeros_like(rnn.W_hy)
    dbh   = np.zeros_like(rnn.bh)
    dby   = np.zeros_like(rnn.by)

    dh_next = np.zeros(rnn.H)

    for t in reversed(range(T)):
        # Gradient of cross-entropy + softmax
        dy = ps[t].copy()
        dy[ys[t]] -= 1.0
        dy /= T

        # Output layer gradients
        dW_hy += np.outer(hs[t + 1], dy)
        dby   += dy

        # Gradient flowing back into h_{t+1}
        dh = rnn.W_hy @ dy + dh_next

        # Gradient through tanh
        dtanh = (1.0 - hs[t + 1] ** 2) * dh

        # Accumulate parameter gradients
        dW_hh += np.outer(hs[t], dtanh)
        dW_xh += np.outer(xs[t], dtanh)
        dbh   += dtanh

        # Pass gradient back one more step
        dh_next = rnn.W_hh.T @ dtanh

        # Truncate BPTT at `truncate` steps
        if T - 1 - t >= truncate:
            break

    grads = {
        "W_hh": dW_hh,
        "W_xh": dW_xh,
        "W_hy": dW_hy,
        "bh":   dbh,
        "by":   dby,
    }

    # --- Gradient norm clipping ---
    total_norm = np.sqrt(sum(np.sum(g ** 2) for g in grads.values()))
    clip_threshold = 5.0
    if total_norm > clip_threshold:
        scale = clip_threshold / (total_norm + 1e-8)
        for key in grads:
            grads[key] *= scale

    return grads, loss


def sgd(rnn, grads, lr=0.01):
    """Apply SGD updates to all RNN parameters."""
    rnn.W_hh -= lr * grads["W_hh"]
    rnn.W_xh -= lr * grads["W_xh"]
    rnn.W_hy -= lr * grads["W_hy"]
    rnn.bh   -= lr * grads["bh"]
    rnn.by   -= lr * grads["by"]
