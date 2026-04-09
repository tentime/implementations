"""
Train a small BERT-MLM model on a short word-level corpus.

After training, probe the model: given "the [MASK] sat on the mat",
predict the top-5 tokens for the masked position.
"""

import torch
import torch.nn as nn
from bert import BertMLM, mask_tokens, MASK_ID

# ---------------------------------------------------------------------------
# Corpus — ~200 words repeated 5× for a bit more data
# ---------------------------------------------------------------------------
CORPUS = """the quick brown fox jumps over the lazy dog the dog slept all day
the fox ran through the forest the forest was dark and deep the trees were tall
the cat sat on the mat the mat was red and warm the cat purred loudly
a bird flew over the river the river was cold and fast fish swam below
the sun rose over the mountains the mountains were covered in snow clouds gathered
above the valley below the valley a small village sat quiet and still in the morning""" * 5

# ---------------------------------------------------------------------------
# Tokenization — word level
# ---------------------------------------------------------------------------
words = CORPUS.lower().split()

# Build vocabulary; reserve ids 0–3 for special tokens
# PAD=0, MASK=1, BOS=2, EOS=3
SPECIAL_TOKENS = ["[PAD]", "[MASK]", "[BOS]", "[EOS]"]
unique_words = sorted(set(words))
vocab = {tok: idx for idx, tok in enumerate(SPECIAL_TOKENS)}
for w in unique_words:
    if w not in vocab:
        vocab[w] = len(vocab)
id_to_token = {v: k for k, v in vocab.items()}

vocab_size = len(vocab)
print(f"Vocabulary size: {vocab_size}")

# Encode the full corpus as a flat list of token ids
token_ids = [vocab[w] for w in words]

# ---------------------------------------------------------------------------
# Hyperparameters
# ---------------------------------------------------------------------------
SEQ_LEN = 32
BATCH_SIZE = 16
D_MODEL = 64
NUM_HEADS = 4
NUM_LAYERS = 2
D_FF = 128
MAX_LEN = 64
NUM_STEPS = 1000
LR = 3e-4
MASK_PROB = 0.15

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
model = BertMLM(
    vocab_size=vocab_size,
    d_model=D_MODEL,
    num_heads=NUM_HEADS,
    num_layers=NUM_LAYERS,
    d_ff=D_FF,
    max_len=MAX_LEN,
).to(device)

total_params = sum(p.numel() for p in model.parameters())
print(f"Model parameters: {total_params:,}")

optimizer = torch.optim.Adam(model.parameters(), lr=LR)
loss_fn = nn.CrossEntropyLoss(ignore_index=-100)

# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------
corpus_tensor = torch.tensor(token_ids, dtype=torch.long)
n_tokens = len(corpus_tensor)


def sample_batch(batch_size, seq_len):
    """Sample random fixed-length windows from the corpus."""
    starts = torch.randint(0, n_tokens - seq_len, (batch_size,))
    batch = torch.stack([corpus_tensor[s: s + seq_len] for s in starts])
    return batch


print("\nTraining...")
model.train()
for step in range(1, NUM_STEPS + 1):
    batch = sample_batch(BATCH_SIZE, SEQ_LEN).to(device)
    masked, labels = mask_tokens(batch, vocab_size, mask_prob=MASK_PROB)

    logits = model(masked)  # (batch, seq_len, vocab_size)

    # Flatten for cross-entropy: ignore_index=-100 handles unmasked positions
    loss = loss_fn(logits.view(-1, vocab_size), labels.view(-1))

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if step % 200 == 0:
        print(f"  step {step:4d} | loss {loss.item():.4f}")

print("Training complete.\n")

# ---------------------------------------------------------------------------
# Probe: "the [MASK] sat on the mat" → top-5 predictions
# ---------------------------------------------------------------------------
probe_words = ["the", "[MASK]", "sat", "on", "the", "mat"]
probe_ids = []
for w in probe_words:
    if w == "[MASK]":
        probe_ids.append(MASK_ID)
    else:
        probe_ids.append(vocab.get(w, vocab.get("[PAD]")))

probe_tensor = torch.tensor([probe_ids], dtype=torch.long, device=device)
mask_position = probe_words.index("[MASK]")  # position 1

model.eval()
with torch.no_grad():
    logits = model(probe_tensor)  # (1, seq_len, vocab_size)
    mask_logits = logits[0, mask_position]  # (vocab_size,)
    probs = torch.softmax(mask_logits, dim=-1)
    top5_probs, top5_ids = probs.topk(5)

print('Probe: "the [MASK] sat on the mat"')
print("Top-5 predictions for [MASK]:")
for prob, tok_id in zip(top5_probs.tolist(), top5_ids.tolist()):
    token = id_to_token[tok_id]
    print(f"  {token!r:12s}  {prob:.4f}")
