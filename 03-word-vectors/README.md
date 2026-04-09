# 03 — Word Vectors: Word2Vec and GloVe

## Context

Tomas Mikolov's 2013 papers introduced Word2Vec and changed how NLP thinks about meaning. The core insight: if you train a shallow neural network to predict context words from a center word (or vice versa), the learned weight matrix turns out to encode semantic relationships geometrically. Words that appear in similar contexts end up near each other in vector space — and the geometry generalizes: `king - man + woman ≈ queen`.

There are two variants of Word2Vec. CBOW predicts the center word from context; Skip-Gram predicts context words from the center. Skip-Gram works better for rare words and is what this implementation uses.

**Word2Vec vs. GloVe.** Both produce similar-quality embeddings, but they approach the problem differently. Word2Vec is a *predictive* model — it trains by sampling pairs and maximizing a classification objective. GloVe (Pennington et al., 2014) is a *reconstruction* model — it builds a global co-occurrence matrix first, then factorizes it using weighted least-squares. GloVe's loss has a clean statistical interpretation: the ratio of co-occurrence probabilities encodes meaning more reliably than raw counts. In practice both converge to embeddings of similar quality; GloVe is often faster to train on large corpora because it operates on the compressed co-occurrence matrix rather than re-scanning the corpus.

---

## What this code does

Run `python train.py` and you will see:

1. **Word2Vec trains for 500 steps** over Skip-Gram pairs with negative sampling. Loss is printed every 100 steps — it should fall noticeably over the first 200 steps.
2. **GloVe trains for 100 steps** with Adagrad. Loss is printed every 20 steps.
3. **Most-similar queries** for `king`, `queen`, `good`, and `bad`. Even on this tiny corpus the neighbors are semantically plausible (e.g. `queen` near `king`, `bad` near `good`).
4. **Analogy demo**: `king - man + woman` — the nearest neighbor of the resulting vector is printed. Results vary with random seed; on this corpus `queen` or `throne` often appear.
5. **word_vectors.png** — a 2D PCA scatter of 20 words. Semantic clusters (royalty, places, animals) are loosely visible even with 500 training steps on 200 unique tokens.

---

## Key implementation details

**The 3/4-power trick in negative sampling.** The negative samples are drawn from a unigram distribution raised to the 0.75 power: `p(w) ∝ freq(w)^0.75`. Why 0.75 and not 1.0? Sampling proportional to raw frequency over-represents very common words like "the" — they dominate negatives and produce uninformative gradients. Raising to a power < 1 smooths the distribution toward uniformity. Mikolov found 0.75 empirically works better than 0.5 or 1.0. It is a hyper-parameter with no deep theoretical justification, just an effective heuristic.

**Sparse updates.** Only the embedding rows that participated in a training step receive gradient updates. `W_in` is updated only for the center word; `W_out` only for the context word and the sampled negatives. `np.add.at` is used rather than `+=` because negative samples can repeat, and `+=` would silently apply only the last update for duplicated indices.

**GloVe weighted loss.** The weighting function `f(x) = min(1, (x/x_max)^alpha)` serves two purposes. First, it caps the influence of very frequent pairs (e.g. "the the") which otherwise dominate the loss and push vectors toward trivial solutions. Second, it down-weights rare pairs whose co-occurrence counts are noisy. This is analogous in spirit (though not in derivation) to Kneser-Ney smoothing in n-gram models: both discount frequent events and redistribute probability mass more evenly. The Adagrad optimizer is standard for GloVe — it adapts the learning rate per-parameter, which matters because embedding rows for rare words receive fewer updates and benefit from a relatively larger step size.

---

## What's deliberately omitted

- **Subword tokenization (BPE).** Word2Vec operates on whole tokens; rare and unknown words get no embedding. FastText extends Skip-Gram with character n-gram subword embeddings to handle morphology and OOV words — that is the natural next step.
- **Hierarchical softmax.** An alternative to negative sampling that replaces the output layer with a Huffman-coded binary tree over the vocabulary, reducing the per-step cost from O(V) to O(log V). Negative sampling is simpler and nearly as effective in practice.
- **The SGNS objective paper details.** Levy & Goldberg (2014) showed that SGNS implicitly factorizes the PMI matrix shifted by log(k). This theoretical connection between Word2Vec and matrix factorization explains why Word2Vec and GloVe converge to similar results despite looking very different on the surface.
- **Dynamic window sizing.** The original Word2Vec samples the window size uniformly from [1, max_window] at each position, giving closer context words higher expected weight. This implementation uses a fixed window.
