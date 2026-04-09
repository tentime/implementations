"""
Word2Vec Skip-Gram with Negative Sampling — NumPy only.
"""

import numpy as np


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))


class Vocabulary:
    def __init__(self):
        self.word2id = {}
        self.id2word = {}
        self.word_counts = {}

    def build(self, corpus_text):
        tokens = corpus_text.lower().split()
        for word in tokens:
            self.word_counts[word] = self.word_counts.get(word, 0) + 1
        for idx, word in enumerate(sorted(self.word_counts.keys())):
            self.word2id[word] = idx
            self.id2word[idx] = word
        return tokens

    def __len__(self):
        return len(self.word2id)

    def make_training_pairs(self, tokens, window=2):
        """Generate (center_word_id, context_word_id) pairs for Skip-Gram."""
        pairs = []
        ids = [self.word2id[t] for t in tokens if t in self.word2id]
        for i, center_id in enumerate(ids):
            lo = max(0, i - window)
            hi = min(len(ids), i + window + 1)
            for j in range(lo, hi):
                if j != i:
                    pairs.append((center_id, ids[j]))
        return pairs


class NegativeSampler:
    def __init__(self, vocab, k=5):
        self.k = k
        words = sorted(vocab.word2id.keys(), key=lambda w: vocab.word2id[w])
        counts = np.array([vocab.word_counts[w] for w in words], dtype=np.float64)
        powered = counts ** 0.75
        self.dist = powered / powered.sum()
        self.vocab_size = len(vocab)

    def sample(self, n):
        """Draw n negative sample word ids from the unigram^0.75 distribution."""
        return np.random.choice(self.vocab_size, size=n, p=self.dist)


class SkipGram:
    def __init__(self, vocab_size, embed_dim=50):
        scale = 0.01
        self.V = vocab_size
        self.d = embed_dim
        self.W_in = np.random.randn(vocab_size, embed_dim) * scale   # input  (center) embeddings
        self.W_out = np.random.randn(vocab_size, embed_dim) * scale  # output (context) embeddings

    def forward(self, center_id, context_id, neg_ids):
        """
        Compute sigmoid scores for the positive pair and each negative pair.

        Returns:
            pos_score : scalar sigmoid(W_in[center] · W_out[context])
            neg_scores: array of shape (k,) — one per negative sample
        """
        v_c = self.W_in[center_id]                  # (d,)
        u_pos = self.W_out[context_id]               # (d,)
        u_neg = self.W_out[neg_ids]                  # (k, d)

        pos_score = sigmoid(np.dot(v_c, u_pos))
        neg_scores = sigmoid(np.dot(u_neg, v_c))    # (k,)
        return pos_score, neg_scores

    def backward(self, center_id, context_id, neg_ids, pos_score, neg_scores):
        """
        Gradients of the negative-sampling binary cross-entropy loss:
            L = -log(sigma(v·u_pos)) - sum_k log(sigma(-v·u_neg_k))

        Returns dict of gradient arrays (not yet applied).
        """
        v_c = self.W_in[center_id]
        u_pos = self.W_out[context_id]
        u_neg = self.W_out[neg_ids]   # (k, d)

        # Gradient w.r.t. v_c (W_in[center_id])
        d_v = -(1 - pos_score) * u_pos + np.sum((1 - neg_scores)[:, None] * u_neg * (-1) * (-1), axis=0)
        # Simplify: d_v = (pos_score - 1)*u_pos + sum_k (neg_scores_k)*u_neg_k
        d_v = (pos_score - 1.0) * u_pos + np.sum(neg_scores[:, None] * u_neg, axis=0)

        # Gradient w.r.t. u_pos (W_out[context_id])
        d_u_pos = (pos_score - 1.0) * v_c

        # Gradient w.r.t. each u_neg (W_out[neg_ids])  shape: (k, d)
        d_u_neg = neg_scores[:, None] * v_c[None, :]

        return {
            "d_v": d_v,
            "d_u_pos": d_u_pos,
            "d_u_neg": d_u_neg,
        }

    def update(self, center_id, context_id, neg_ids, grads, lr):
        """Apply sparse in-place gradient descent updates."""
        self.W_in[center_id] -= lr * grads["d_v"]
        self.W_out[context_id] -= lr * grads["d_u_pos"]
        # Sparse update: accumulate by index (handles duplicate neg ids)
        np.add.at(self.W_out, neg_ids, -lr * grads["d_u_neg"])

    def get_embedding(self, word_id):
        return self.W_in[word_id].copy()

    def most_similar(self, word_id, top_k=5):
        """Cosine similarity of W_in[word_id] against all rows of W_in."""
        query = self.W_in[word_id]
        norms = np.linalg.norm(self.W_in, axis=1) + 1e-8
        query_norm = np.linalg.norm(query) + 1e-8
        sims = self.W_in @ query / (norms * query_norm)
        # Exclude the query word itself
        sims[word_id] = -np.inf
        top_ids = np.argsort(sims)[::-1][:top_k]
        return top_ids, sims[top_ids]
