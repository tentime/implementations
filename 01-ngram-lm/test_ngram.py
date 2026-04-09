"""
Pytest tests for ngram_lm.py.
"""

import numpy as np
import pytest
from ngram_lm import NgramLM, Tokenizer

# ---------------------------------------------------------------------------
# Small corpus used across multiple tests
# ---------------------------------------------------------------------------

SMALL_CORPUS = """
the cat sat on the mat
the cat ate the rat
the dog sat on the log
the rat ran from the cat
the dog ran after the cat
"""

# Sentence drawn from the same distribution (overlapping vocabulary).
IN_DIST_TEXT = "the cat sat on the mat"

# Random character soup — guaranteed high perplexity.
RANDOM_SOUP = "xqz wvb jkp mrf ltz qvb xzp wjk"


# ---------------------------------------------------------------------------
# Helper: build and train a model on the small corpus
# ---------------------------------------------------------------------------

def make_model(corpus=SMALL_CORPUS):
    m = NgramLM()
    m.train(corpus)
    return m


# ---------------------------------------------------------------------------
# Test 1: bigram probabilities sum to 1
# ---------------------------------------------------------------------------

def test_bigram_probabilities_sum_to_one():
    """
    For every unigram context that appears in the corpus, the Laplace-smoothed
    bigram probabilities over the full vocabulary must sum to 1.0 (within 1e-9).

    This verifies the normalisation identity:
        sum_w  (C(ctx,w)+1) / (C(ctx)+V)  == 1
    which holds exactly when the denominator is C(ctx)+V and we sum over V words.
    """
    model = make_model()
    vocab = list(model.tokenizer.vocab.keys())
    V = len(vocab)

    # Collect all unigram contexts that actually appeared in training.
    contexts_seen = set()
    for (w1, w2) in model.bigram_counts:
        contexts_seen.add((w1,))

    assert len(contexts_seen) > 0, "No bigram contexts found — corpus may be empty."

    for ctx in contexts_seen:
        total = sum(model.laplace_prob(ctx, w) for w in vocab)
        assert abs(total - 1.0) < 1e-9, (
            f"Bigram probs from context {ctx} sum to {total}, expected 1.0"
        )


# ---------------------------------------------------------------------------
# Test 2: perplexity is lower for in-distribution text
# ---------------------------------------------------------------------------

def test_perplexity_lower_for_in_distribution_text():
    """
    Perplexity on a sentence drawn from the training distribution should be
    strictly less than perplexity on random character soup.

    Both Laplace and Kneser-Ney are tested.
    """
    model = make_model()

    for smoothing in ("laplace", "kneser_ney"):
        ppl_in  = model.perplexity(IN_DIST_TEXT, smoothing=smoothing)
        ppl_out = model.perplexity(RANDOM_SOUP,  smoothing=smoothing)
        assert ppl_in < ppl_out, (
            f"[{smoothing}] Expected in-dist perplexity ({ppl_in:.2f}) < "
            f"OOD perplexity ({ppl_out:.2f})"
        )


# ---------------------------------------------------------------------------
# Test 3: generated text starts with the given prefix
# ---------------------------------------------------------------------------

def test_generate_prefix():
    """
    The first tokens of the generated sequence must match the prefix words
    supplied to generate(), regardless of what follows.
    """
    np.random.seed(7)
    model = make_model()

    prefix = "the cat"
    result = model.generate(prefix, n_words=10, smoothing="laplace", temperature=1.0)
    words = result.split()

    assert words[0] == "the", f"Expected 'the' as first token, got '{words[0]}'"
    assert words[1] == "cat", f"Expected 'cat' as second token, got '{words[1]}'"


# ---------------------------------------------------------------------------
# Test 4: Kneser-Ney perplexity <= Laplace perplexity on held-out text
# ---------------------------------------------------------------------------

def test_kneser_ney_lower_perplexity_than_laplace():
    """
    Kneser-Ney smoothing should assign probability mass more efficiently than
    Laplace, yielding equal or lower perplexity on held-out text that shares
    vocabulary with the training set.

    We train on the full small corpus and evaluate on a held-out sentence that
    uses known vocabulary in a plausible order.
    """
    model = make_model()
    held_out = "the dog ran from the mat"

    ppl_laplace = model.perplexity(held_out, smoothing="laplace")
    ppl_kn      = model.perplexity(held_out, smoothing="kneser_ney")

    assert ppl_kn <= ppl_laplace + 1e-6, (
        f"Expected KN perplexity ({ppl_kn:.4f}) <= "
        f"Laplace perplexity ({ppl_laplace:.4f})"
    )
