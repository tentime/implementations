"""
Stage 1: Supervised Fine-Tuning (SFT)

Fine-tune a GPT model on (prompt, ideal_completion) pairs.
This creates the starting point for RL fine-tuning.

Run: python train_sft.py
Saves: sft_model.pt
"""

import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from reward_model import GPTBase

# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

SUBJECTS = ["cat", "dog", "bird", "fish", "fox", "wolf", "bear", "deer"]
VERBS = ["sat", "ran", "jumped", "swam", "flew", "slept", "played", "waited"]
PLACES = ["mat", "park", "river", "forest", "mountain", "valley", "meadow", "cliff"]

# Seed pairs
SFT_DATA = [
    ("summarize: the cat sat on the mat", "a cat on a mat"),
    ("summarize: the dog ran through the park", "a dog in a park"),
    ("summarize: the sun rose over the mountain", "sunrise over mountain"),
]


def make_sft_data(n=80):
    random.seed(42)
    pairs = list(SFT_DATA)  # start with seed pairs
    for _ in range(n):
        s = random.choice(SUBJECTS)
        v = random.choice(VERBS)
        p = random.choice(PLACES)
        prompt = f"summarize: the {s} {v} near the {p}"
        completion = f"{s} near {p}"
        pairs.append((prompt, completion))
    return pairs


# ---------------------------------------------------------------------------
# Tokenizer (character-level, self-contained)
# ---------------------------------------------------------------------------

class CharTokenizer:
    """Minimal character-level tokenizer. Vocab built from training data."""

    PAD = 0
    BOS = 1
    EOS = 2
    RESERVED = 3  # start of real characters

    def __init__(self, texts):
        chars = sorted(set("".join(texts)))
        self.ch2id = {ch: i + self.RESERVED for i, ch in enumerate(chars)}
        self.id2ch = {v: k for k, v in self.ch2id.items()}
        self.vocab_size = self.RESERVED + len(chars)

    def encode(self, text):
        return [self.BOS] + [self.ch2id[ch] for ch in text if ch in self.ch2id] + [self.EOS]

    def decode(self, ids):
        out = []
        for i in ids:
            if i == self.EOS:
                break
            if i in self.id2ch:
                out.append(self.id2ch[i])
        return "".join(out)


def build_tokenizer(pairs):
    all_text = " ".join(p + " " + c for p, c in pairs)
    return CharTokenizer([all_text])


# ---------------------------------------------------------------------------
# SFT Model (GPT + language model head)
# ---------------------------------------------------------------------------

class SFTModel(nn.Module):
    def __init__(self, vocab_size, d_model=64, num_heads=4, num_layers=2, max_len=128):
        super().__init__()
        self.backbone = GPTBase(vocab_size, d_model, num_heads, num_layers, max_len)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, token_ids):
        """Returns logits: (B, T, vocab_size)"""
        hidden = self.backbone(token_ids)
        return self.lm_head(hidden)

    @torch.no_grad()
    def generate(self, prompt_ids, max_new_tokens=20, temperature=1.0):
        """Greedy / temperature sampling for a single prompt."""
        ids = prompt_ids.clone()
        for _ in range(max_new_tokens):
            logits = self.forward(ids.unsqueeze(0))  # (1, T, V)
            next_logits = logits[0, -1, :] / temperature
            next_id = torch.multinomial(F.softmax(next_logits, dim=-1), num_samples=1)
            ids = torch.cat([ids, next_id])
            if next_id.item() == 2:  # EOS
                break
        return ids


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def make_batch(pairs, tokenizer, device, max_len=128):
    """
    Build a batch of (input, target) pairs for next-token prediction.
    Full sequence = BOS + prompt + SEP + completion + EOS.
    Target is the sequence shifted right by 1.
    """
    sequences = []
    for prompt, completion in pairs:
        full = tokenizer.encode(prompt + " | " + completion)
        full = full[:max_len]
        sequences.append(full)

    # Pad to same length
    max_seq = max(len(s) for s in sequences)
    padded = []
    for s in sequences:
        padded.append(s + [0] * (max_seq - len(s)))

    tokens = torch.tensor(padded, dtype=torch.long, device=device)
    inputs = tokens[:, :-1]
    targets = tokens[:, 1:]
    return inputs, targets


def train_sft(n_steps=400, batch_size=16, lr=3e-3, device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    pairs = make_sft_data(n=80)
    tokenizer = build_tokenizer(pairs)

    model = SFTModel(
        vocab_size=tokenizer.vocab_size,
        d_model=64,
        num_heads=4,
        num_layers=2,
        max_len=128,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    print(f"Vocab size: {tokenizer.vocab_size}")
    print(f"Training on {len(pairs)} pairs for {n_steps} steps...")

    for step in range(1, n_steps + 1):
        # Sample random mini-batch
        indices = random.sample(range(len(pairs)), min(batch_size, len(pairs)))
        batch_pairs = [pairs[i] for i in indices]
        inputs, targets = make_batch(batch_pairs, tokenizer, device)

        logits = model(inputs)  # (B, T, V)
        loss = F.cross_entropy(
            logits.reshape(-1, tokenizer.vocab_size),
            targets.reshape(-1),
            ignore_index=0,  # PAD
        )

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if step % 50 == 0:
            print(f"  step {step:4d} | loss {loss.item():.4f}")

    # Quick generation demo
    print("\nGeneration sample:")
    prompt = "summarize: the fox slept near the river"
    prompt_ids = torch.tensor(tokenizer.encode(prompt), dtype=torch.long, device=device)
    generated = model.generate(prompt_ids, max_new_tokens=30, temperature=0.8)
    print(f"  prompt:    {prompt!r}")
    print(f"  generated: {tokenizer.decode(generated.tolist())!r}")

    # Save checkpoint
    checkpoint = {
        "model_state": model.state_dict(),
        "vocab_size": tokenizer.vocab_size,
        "tokenizer_ch2id": tokenizer.ch2id,
        "d_model": 64,
        "num_heads": 4,
        "num_layers": 2,
        "max_len": 128,
    }
    torch.save(checkpoint, "sft_model.pt")
    print("\nSaved sft_model.pt")
    return model, tokenizer


if __name__ == "__main__":
    train_sft()
