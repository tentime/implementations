"""
GloVe — Global Vectors for Word Representation, with Adagrad optimizer.
NumPy only.
"""

import numpy as np
from collections import defaultdict


def build_cooccurrence(tokens, window=5):
    """
    Build a symmetric co-occurrence dictionary {(i, j): count}.
    Both (i, j) and (j, i) are incremented for each context pair.
    """
    cooc = defaultdict(float)
    n = len(tokens)
    for center_pos, center_id in enumerate(tokens):
        lo = max(0, center_pos - window)
        hi = min(n, center_pos + window + 1)
        for ctx_pos in range(lo, hi):
            if ctx_pos == center_pos:
                continue
            ctx_id = tokens[ctx_pos]
            # Harmonic weighting by distance (optional but common)
            dist = abs(ctx_pos - center_pos)
            cooc[(center_id, ctx_id)] += 1.0 / dist
    return dict(cooc)


class GloVe:
    def __init__(self, vocab_size, embed_dim=50):
        scale = 0.01
        self.V = vocab_size
        self.d = embed_dim

        # Main word and context vectors
        self.W = np.random.randn(vocab_size, embed_dim) * scale   # (V, d)
        self.C = np.random.randn(vocab_size, embed_dim) * scale   # (V, d)

        # Bias terms
        self.bw = np.zeros(vocab_size)   # (V,)
        self.bc = np.zeros(vocab_size)   # (V,)

        # Adagrad squared-gradient accumulators
        self.sq_W = np.ones((vocab_size, embed_dim))
        self.sq_C = np.ones((vocab_size, embed_dim))
        self.sq_bw = np.ones(vocab_size)
        self.sq_bc = np.ones(vocab_size)

        # Weighting function hyperparameters
        self.x_max = 100.0
        self.alpha = 0.75

    def _weighting(self, x):
        """f(x) = min(1, (x / x_max)^alpha)"""
        return np.minimum(1.0, (x / self.x_max) ** self.alpha)

    def loss(self, pairs):
        """
        Weighted least-squares GloVe loss over all co-occurrence pairs.

        pairs: list of (i, j, X_ij)
        """
        total = 0.0
        for i, j, x_ij in pairs:
            f = self._weighting(x_ij)
            diff = np.dot(self.W[i], self.C[j]) + self.bw[i] + self.bc[j] - np.log(x_ij)
            total += f * diff * diff
        return total

    def train_step(self, pairs, lr=0.05):
        """One Adagrad step over all co-occurrence pairs."""
        eps = 1e-8
        for i, j, x_ij in pairs:
            f = self._weighting(x_ij)
            diff = np.dot(self.W[i], self.C[j]) + self.bw[i] + self.bc[j] - np.log(x_ij)
            common = 2.0 * f * diff

            # Gradients
            grad_wi = common * self.C[j]
            grad_cj = common * self.W[i]
            grad_bwi = common
            grad_bcj = common

            # Adagrad accumulator update
            self.sq_W[i] += grad_wi ** 2
            self.sq_C[j] += grad_cj ** 2
            self.sq_bw[i] += grad_bwi ** 2
            self.sq_bc[j] += grad_bcj ** 2

            # Parameter update
            self.W[i] -= lr * grad_wi / (np.sqrt(self.sq_W[i]) + eps)
            self.C[j] -= lr * grad_cj / (np.sqrt(self.sq_C[j]) + eps)
            self.bw[i] -= lr * grad_bwi / (np.sqrt(self.sq_bw[i]) + eps)
            self.bc[j] -= lr * grad_bcj / (np.sqrt(self.sq_bc[j]) + eps)

    def get_word_vector(self, word_id):
        """Standard GloVe practice: average of word and context vectors."""
        return (self.W[word_id] + self.C[word_id]) / 2.0
