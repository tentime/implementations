"""
Training script: Word2Vec and GloVe on a mini-corpus.
Produces word_vectors.png with a 2D PCA scatter of 20 words.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from word2vec import Vocabulary, NegativeSampler, SkipGram, sigmoid
from glove import GloVe, build_cooccurrence

# ---------------------------------------------------------------------------
# Mini corpus
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


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def w2v_step_loss(model, center_id, context_id, neg_ids):
    """Compute Word2Vec negative-sampling loss (scalar) for one training pair."""
    pos_score, neg_scores = model.forward(center_id, context_id, neg_ids)
    loss = -np.log(pos_score + 1e-8) - np.sum(np.log(1.0 - neg_scores + 1e-8))
    return loss


# ---------------------------------------------------------------------------
# Train Word2Vec
# ---------------------------------------------------------------------------

np.random.seed(42)

vocab = Vocabulary()
tokens = vocab.build(CORPUS)
pairs = vocab.make_training_pairs(tokens, window=2)

sampler = NegativeSampler(vocab, k=5)
model = SkipGram(vocab_size=len(vocab), embed_dim=50)

print("=== Word2Vec Training ===")
lr = 0.025
step = 0
running_loss = 0.0
report_every = 100
max_steps = 500

# Shuffle pairs once
pairs_arr = pairs[:]
np.random.shuffle(pairs_arr)

for center_id, context_id in pairs_arr:
    neg_ids = sampler.sample(sampler.k)
    pos_score, neg_scores = model.forward(center_id, context_id, neg_ids)
    loss = -np.log(pos_score + 1e-8) - np.sum(np.log(1.0 - neg_scores + 1e-8))
    running_loss += loss

    grads = model.backward(center_id, context_id, neg_ids, pos_score, neg_scores)
    model.update(center_id, context_id, neg_ids, grads, lr)

    step += 1
    if step % report_every == 0:
        avg = running_loss / report_every
        print(f"  step {step:4d}  avg loss: {avg:.4f}")
        running_loss = 0.0

    if step >= max_steps:
        break

print()


# ---------------------------------------------------------------------------
# Train GloVe
# ---------------------------------------------------------------------------

np.random.seed(42)

token_ids = [vocab.word2id[t] for t in tokens if t in vocab.word2id]
cooc = build_cooccurrence(token_ids, window=5)
pairs_glove = [(i, j, x) for (i, j), x in cooc.items()]

glove = GloVe(vocab_size=len(vocab), embed_dim=50)

print("=== GloVe Training ===")
for glove_step in range(1, 101):
    glove.train_step(pairs_glove, lr=0.05)
    if glove_step % 20 == 0:
        l = glove.loss(pairs_glove)
        print(f"  step {glove_step:3d}  loss: {l:.4f}")

print()


# ---------------------------------------------------------------------------
# Most-similar words
# ---------------------------------------------------------------------------

print("=== Most Similar (Word2Vec) ===")
query_words = ["king", "queen", "good", "bad"]
for word in query_words:
    if word not in vocab.word2id:
        print(f"  '{word}' not in vocabulary")
        continue
    wid = vocab.word2id[word]
    top_ids, sims = model.most_similar(wid, top_k=3)
    neighbors = [(vocab.id2word[i], f"{s:.3f}") for i, s in zip(top_ids, sims)]
    print(f"  {word:8s} → {neighbors}")

print()


# ---------------------------------------------------------------------------
# Analogy: king - man + woman
# ---------------------------------------------------------------------------

print("=== Analogy: king - man + woman ===")
required = ["king", "man", "woman"]
if all(w in vocab.word2id for w in required):
    v_king  = model.get_embedding(vocab.word2id["king"])
    v_man   = model.get_embedding(vocab.word2id["man"])
    v_woman = model.get_embedding(vocab.word2id["woman"])

    analogy_vec = v_king - v_man + v_woman

    # Cosine similarity against all W_in vectors
    norms = np.linalg.norm(model.W_in, axis=1) + 1e-8
    analogy_norm = np.linalg.norm(analogy_vec) + 1e-8
    sims = model.W_in @ analogy_vec / (norms * analogy_norm)

    # Exclude the source words
    for w in required:
        sims[vocab.word2id[w]] = -np.inf

    best_id = int(np.argmax(sims))
    print(f"  king - man + woman ≈ '{vocab.id2word[best_id]}'  (sim={sims[best_id]:.3f})")
else:
    missing = [w for w in required if w not in vocab.word2id]
    print(f"  Missing words: {missing}")

print()


# ---------------------------------------------------------------------------
# 2D PCA scatter (no sklearn)
# ---------------------------------------------------------------------------

print("=== Saving word_vectors.png ===")

plot_words = [
    "king", "queen", "man", "woman",
    "cat", "dog", "castle", "throne",
    "france", "england", "paris", "london",
    "good", "bad", "gold", "land",
    "city", "sat", "walked", "capital",
]
plot_words = [w for w in plot_words if w in vocab.word2id][:20]

embeddings = np.stack([model.get_embedding(vocab.word2id[w]) for w in plot_words])

# Manual PCA: center, SVD, project onto top-2 components
embeddings_centered = embeddings - embeddings.mean(axis=0)
U, S, Vt = np.linalg.svd(embeddings_centered, full_matrices=False)
coords_2d = embeddings_centered @ Vt[:2].T   # (N, 2)

fig, ax = plt.subplots(figsize=(10, 8))
ax.scatter(coords_2d[:, 0], coords_2d[:, 1], s=40, color="steelblue", alpha=0.7)
for i, word in enumerate(plot_words):
    ax.annotate(
        word,
        (coords_2d[i, 0], coords_2d[i, 1]),
        fontsize=11,
        xytext=(5, 5),
        textcoords="offset points",
    )
ax.set_title("Word Embeddings — PCA projection (Word2Vec Skip-Gram)", fontsize=13)
ax.set_xlabel("PC 1")
ax.set_ylabel("PC 2")
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("word_vectors.png", dpi=120)
plt.close()
print("  Saved to word_vectors.png")
