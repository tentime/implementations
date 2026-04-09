"""
Toy task: English digit words → French digit words.

"three two seven" → "trois deux sept"

Sequences of 3-6 digit words are generated randomly.
1000 training examples, 100 test examples.
"""

import random
import torch
import torch.nn as nn
import torch.optim as optim

from transformer import Transformer

# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

EN = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"]
FR = ["zero", "un", "deux", "trois", "quatre", "cinq", "six", "sept", "huit", "neuf"]

PAD = "<PAD>"
SOS = "<SOS>"
EOS = "<EOS>"
SPECIAL = [PAD, SOS, EOS]


def generate_example(seed):
    rng = random.Random(seed)
    length = rng.randint(3, 6)
    indices = [rng.randint(0, 9) for _ in range(length)]
    src_words = [EN[i] for i in indices]
    tgt_words = [FR[i] for i in indices]
    return src_words, tgt_words


def build_dataset(n_train=1000, n_test=100):
    total = n_train + n_test
    pairs = [generate_example(i) for i in range(total)]
    return pairs[:n_train], pairs[n_train:]


def build_vocab(pairs):
    src_tokens = set()
    tgt_tokens = set()
    for src, tgt in pairs:
        src_tokens.update(src)
        tgt_tokens.update(tgt)
    src_vocab = SPECIAL + sorted(src_tokens)
    tgt_vocab = SPECIAL + sorted(tgt_tokens)
    src2idx = {w: i for i, w in enumerate(src_vocab)}
    tgt2idx = {w: i for i, w in enumerate(tgt_vocab)}
    idx2tgt = {i: w for w, i in tgt2idx.items()}
    return src2idx, tgt2idx, idx2tgt


def encode_src(words, src2idx):
    return [src2idx[SOS]] + [src2idx[w] for w in words] + [src2idx[EOS]]


def encode_tgt(words, tgt2idx):
    return [tgt2idx[SOS]] + [tgt2idx[w] for w in words] + [tgt2idx[EOS]]


def decode_tgt(indices, idx2tgt, eos_idx):
    words = []
    for idx in indices:
        if idx == eos_idx:
            break
        w = idx2tgt.get(idx, "?")
        if w not in (SOS, PAD):
            words.append(w)
    return words


# ---------------------------------------------------------------------------
# Collate / batching (pad sequences in a batch to the same length)
# ---------------------------------------------------------------------------

def pad_sequence(seqs, pad_idx):
    """seqs: list of 1-D tensors → (batch, max_len)"""
    max_len = max(s.shape[0] for s in seqs)
    out = torch.full((len(seqs), max_len), pad_idx, dtype=torch.long)
    for i, s in enumerate(seqs):
        out[i, :s.shape[0]] = s
    return out


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train():
    torch.manual_seed(42)
    random.seed(42)

    train_pairs, test_pairs = build_dataset(1000, 100)
    all_pairs = train_pairs + test_pairs
    src2idx, tgt2idx, idx2tgt = build_vocab(all_pairs)

    pad_idx   = src2idx[PAD]
    sos_idx   = tgt2idx[SOS]
    eos_idx   = tgt2idx[EOS]
    src_vocab = len(src2idx)
    tgt_vocab = len(tgt2idx)

    print(f"Source vocab size: {src_vocab}")
    print(f"Target vocab size: {tgt_vocab}")
    print(f"Train: {len(train_pairs)}, Test: {len(test_pairs)}")

    # Pre-encode all pairs as tensors
    train_tensors = [
        (
            torch.tensor(encode_src(s, src2idx), dtype=torch.long),
            torch.tensor(encode_tgt(t, tgt2idx), dtype=torch.long),
        )
        for s, t in train_pairs
    ]
    test_tensors = [
        (
            torch.tensor(encode_src(s, src2idx), dtype=torch.long),
            torch.tensor(encode_tgt(t, tgt2idx), dtype=torch.long),
            s, t,
        )
        for s, t in test_pairs
    ]

    # Model
    model = Transformer(
        src_vocab_size=src_vocab,
        tgt_vocab_size=tgt_vocab,
        d_model=64,
        num_heads=4,
        num_layers=2,
        d_ff=128,
        max_len=20,
        dropout=0.1,
    )
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)

    STEPS = 3000
    BATCH_SIZE = 32

    print(f"\nTraining for {STEPS} steps (batch size {BATCH_SIZE})...\n")

    model.train()
    for step in range(1, STEPS + 1):
        # Sample a random mini-batch
        batch_idx = random.sample(range(len(train_tensors)), BATCH_SIZE)
        src_batch = [train_tensors[i][0] for i in batch_idx]
        tgt_batch = [train_tensors[i][1] for i in batch_idx]

        src_padded = pad_sequence(src_batch, pad_idx)  # (B, src_len)
        tgt_padded = pad_sequence(tgt_batch, pad_idx)  # (B, tgt_len)

        # Input to decoder: all tokens except the last (EOS)
        tgt_in  = tgt_padded[:, :-1]   # (B, tgt_len-1)
        # Target for loss: all tokens except the first (SOS)
        tgt_out = tgt_padded[:, 1:]    # (B, tgt_len-1)

        optimizer.zero_grad()
        logits = model(src_padded, tgt_in)  # (B, tgt_len-1, tgt_vocab)

        # Reshape for CrossEntropyLoss: (B * seq_len, vocab) and (B * seq_len,)
        loss = criterion(
            logits.reshape(-1, tgt_vocab),
            tgt_out.reshape(-1),
        )
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        if step % 500 == 0:
            print(f"Step {step:4d} | Loss: {loss.item():.4f}")

    # ---------------------------------------------------------------------------
    # Evaluation: exact-match accuracy on test set
    # ---------------------------------------------------------------------------
    print("\n--- Test set evaluation ---")
    model.eval()
    correct = 0
    total = len(test_tensors)

    for src_t, tgt_t, src_words, tgt_words in test_tensors:
        pred_indices = model.greedy_decode(
            src_t.unsqueeze(0), sos_idx, eos_idx, max_len=15
        )
        pred_words = decode_tgt(pred_indices, idx2tgt, eos_idx)
        if pred_words == tgt_words:
            correct += 1

    accuracy = correct / total * 100
    print(f"Exact-match accuracy: {correct}/{total} = {accuracy:.1f}%")

    # Print a few examples
    print("\nSample predictions:")
    for src_t, tgt_t, src_words, tgt_words in test_tensors[:5]:
        pred_indices = model.greedy_decode(
            src_t.unsqueeze(0), sos_idx, eos_idx, max_len=15
        )
        pred_words = decode_tgt(pred_indices, idx2tgt, eos_idx)
        match = "OK" if pred_words == tgt_words else "X"
        print(f"  [{match}] {' '.join(src_words)}")
        print(f"       pred: {' '.join(pred_words)}")
        print(f"       gold: {' '.join(tgt_words)}")
        print()

    # Save checkpoint
    torch.save({
        "model_state": model.state_dict(),
        "src2idx": src2idx,
        "tgt2idx": tgt2idx,
        "idx2tgt": idx2tgt,
        "src_vocab": src_vocab,
        "tgt_vocab": tgt_vocab,
    }, "transformer_checkpoint.pt")
    print("Checkpoint saved to transformer_checkpoint.pt")

    return model, accuracy


if __name__ == "__main__":
    train()
