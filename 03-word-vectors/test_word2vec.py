"""
Pytest tests for word2vec.py and glove.py.
"""

import numpy as np
import pytest

from word2vec import Vocabulary, NegativeSampler, SkipGram
from glove import GloVe, build_cooccurrence


# ---------------------------------------------------------------------------
# Shared corpus (same as train.py)
# ---------------------------------------------------------------------------

CORPUS = """
the king sat on the throne the queen stood beside the king
the man walked to the castle the woman ran from the castle
paris is the capital of france london is the capital of england
the cat sat on the mat the dog sat on the floor
good things come to those who wait bad things come to those who rush
the king and the queen ruled the land with good judgment
a man and a woman walked through the city of paris
the cat chased the dog around the castle wall
france and england are good neighbors who sometimes disagree
the throne of the king was made of gold and good wood
""" * 20


def make_vocab_and_pairs():
    vocab = Vocabulary()
    tokens = vocab.build(CORPUS)
    pairs = vocab.make_training_pairs(tokens, window=2)
    return vocab, tokens, pairs


# ---------------------------------------------------------------------------
# test_negative_sampler_shape
# ---------------------------------------------------------------------------

def test_negative_sampler_shape():
    vocab, tokens, _ = make_vocab_and_pairs()
    k = 5
    sampler = NegativeSampler(vocab, k=k)
    samples = sampler.sample(k)
    assert samples.shape == (k,), f"Expected shape ({k},), got {samples.shape}"


# ---------------------------------------------------------------------------
# test_embedding_shapes
# ---------------------------------------------------------------------------

def test_embedding_shapes():
    vocab, _, _ = make_vocab_and_pairs()
    V = len(vocab)
    d = 50
    model = SkipGram(vocab_size=V, embed_dim=d)
    assert model.W_in.shape == (V, d), f"W_in shape mismatch: {model.W_in.shape}"
    assert model.W_out.shape == (V, d), f"W_out shape mismatch: {model.W_out.shape}"


# ---------------------------------------------------------------------------
# test_loss_decreases_word2vec
# ---------------------------------------------------------------------------

def test_loss_decreases_word2vec():
    np.random.seed(42)
    vocab, tokens, pairs = make_vocab_and_pairs()
    sampler = NegativeSampler(vocab, k=5)
    model = SkipGram(vocab_size=len(vocab), embed_dim=50)

    pairs_shuffled = pairs[:]
    np.random.shuffle(pairs_shuffled)

    def compute_avg_loss(model, pairs_subset, sampler):
        total = 0.0
        for center_id, context_id in pairs_subset:
            neg_ids = sampler.sample(sampler.k)
            pos_score, neg_scores = model.forward(center_id, context_id, neg_ids)
            total += -np.log(pos_score + 1e-8) - np.sum(np.log(1.0 - neg_scores + 1e-8))
        return total / len(pairs_subset)

    eval_pairs = pairs_shuffled[:200]

    loss_at_0 = compute_avg_loss(model, eval_pairs, sampler)

    # Train for 200 steps
    lr = 0.025
    for center_id, context_id in pairs_shuffled[:200]:
        neg_ids = sampler.sample(sampler.k)
        pos_score, neg_scores = model.forward(center_id, context_id, neg_ids)
        grads = model.backward(center_id, context_id, neg_ids, pos_score, neg_scores)
        model.update(center_id, context_id, neg_ids, grads, lr)

    loss_at_200 = compute_avg_loss(model, eval_pairs, sampler)

    assert loss_at_200 < loss_at_0, (
        f"Loss did not decrease: step-0={loss_at_0:.4f}, step-200={loss_at_200:.4f}"
    )


# ---------------------------------------------------------------------------
# test_glove_loss_decreases
# ---------------------------------------------------------------------------

def test_glove_loss_decreases():
    np.random.seed(42)
    vocab, tokens, _ = make_vocab_and_pairs()

    token_ids = [vocab.word2id[t] for t in tokens if t in vocab.word2id]
    cooc = build_cooccurrence(token_ids, window=5)
    pairs = [(i, j, x) for (i, j), x in cooc.items()]

    glove = GloVe(vocab_size=len(vocab), embed_dim=50)

    loss_at_0 = glove.loss(pairs)

    for _ in range(50):
        glove.train_step(pairs, lr=0.05)

    loss_at_50 = glove.loss(pairs)

    assert loss_at_50 < loss_at_0, (
        f"GloVe loss did not decrease: step-0={loss_at_0:.4f}, step-50={loss_at_50:.4f}"
    )
