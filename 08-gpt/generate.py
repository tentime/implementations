"""
Demonstrate generation with different temperature settings.

Loads the model checkpoint saved by train.py. If no checkpoint exists,
trains a quick 500-step model first so the script is always runnable.

Prints outputs for:
  - Greedy (temperature=0, implemented as argmax)
  - temperature=0.8  (slightly conservative sampling)
  - temperature=1.4  (high-entropy, more creative / chaotic)
"""

import os
import torch
import torch.nn as nn
from gpt import GPT

CHECKPOINT_PATH = 'gpt_checkpoint.pt'

# ---------------------------------------------------------------------------
# Load or train
# ---------------------------------------------------------------------------

def quick_train():
    """Train a minimal model for 500 steps so generate.py always works."""
    print("No checkpoint found. Running a quick 500-step training run first...\n")

    TEXT = """
First Citizen: Before we proceed any further, hear me speak.
All: Speak, speak.
First Citizen: You are all resolved rather to die than to famish?
All: Resolved. resolved.
First Citizen: First, you know Caius Marcius is chief enemy to the people.
All: We know't, we know't.
First Citizen: Let us kill him, and we'll have corn at our own price.
Is't a verdict?
All: No more talking on't; let it be done: away, away!
Second Citizen: One word, good citizens.
First Citizen: We are accounted poor citizens, the patricians good.
What authority surfeits on would relieve us: if they
would yield us but the superfluity, while it were
wholesome, we might guess they relieved us humanely;
but they think we are too dear: the leanness that
afflicts us, the object of our misery, is as an
inventory to particularise their abundance; our
sufferance is a gain to them Let us revenge this with
our pikes, ere we become rakes: for the gods know I
speak this in hunger for bread, not in thirst for revenge.
""" * 8

    chars = sorted(set(TEXT))
    vocab_size = len(chars)
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for ch, i in stoi.items()}
    encode = lambda s: [stoi[c] for c in s]

    data = torch.tensor(encode(TEXT), dtype=torch.long)
    n = len(data)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    D_MODEL, NUM_HEADS, NUM_LAYERS, MAX_LEN = 64, 4, 4, 128
    model = GPT(vocab_size, D_MODEL, NUM_HEADS, NUM_LAYERS, MAX_LEN, dropout=0.1).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
    loss_fn = nn.CrossEntropyLoss()

    model.train()
    for step in range(1, 501):
        starts = torch.randint(0, n - MAX_LEN - 1, (32,))
        x = torch.stack([data[s: s + MAX_LEN] for s in starts]).to(device)
        y = torch.stack([data[s + 1: s + MAX_LEN + 1] for s in starts]).to(device)
        logits = model(x)
        loss = loss_fn(logits.view(-1, vocab_size), y.view(-1))
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step % 100 == 0:
            print(f"  step {step:3d} | loss {loss.item():.4f}")

    checkpoint = {
        'model_state': model.state_dict(),
        'vocab_size': vocab_size,
        'stoi': stoi,
        'itos': itos,
        'd_model': D_MODEL,
        'num_heads': NUM_HEADS,
        'num_layers': NUM_LAYERS,
        'max_len': MAX_LEN,
    }
    torch.save(checkpoint, CHECKPOINT_PATH)
    print(f"\nQuick checkpoint saved to {CHECKPOINT_PATH}\n")
    return checkpoint


if os.path.exists(CHECKPOINT_PATH):
    print(f"Loading checkpoint from {CHECKPOINT_PATH}")
    checkpoint = torch.load(CHECKPOINT_PATH, map_location='cpu', weights_only=False)
else:
    checkpoint = quick_train()

# ---------------------------------------------------------------------------
# Reconstruct model from checkpoint metadata
# ---------------------------------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

stoi = checkpoint['stoi']
itos = checkpoint['itos']
vocab_size = checkpoint['vocab_size']
decode = lambda ids: ''.join(itos[i] for i in ids)

model = GPT(
    vocab_size=vocab_size,
    d_model=checkpoint['d_model'],
    num_heads=checkpoint['num_heads'],
    num_layers=checkpoint['num_layers'],
    max_len=checkpoint['max_len'],
    dropout=0.0,
).to(device)
model.load_state_dict(checkpoint['model_state'])
model.eval()

# ---------------------------------------------------------------------------
# Generation with different temperatures
# ---------------------------------------------------------------------------
PROMPT = "\nFirst Citizen:"
GENERATE_LEN = 200

prompt_ids = torch.tensor([[stoi[c] for c in PROMPT]], dtype=torch.long, device=device)

print("=" * 60)
print("PROMPT:", repr(PROMPT))
print("=" * 60)

# --- Greedy (temperature=0) ---
print("\n[Greedy — temperature=0 (argmax)]\n")
greedy_out = model.generate(prompt_ids.clone(), max_new_tokens=GENERATE_LEN, temperature=0)
print(decode(greedy_out[0].tolist()))

# --- temperature=0.8 ---
print("\n[temperature=0.8 — conservative sampling]\n")
out_08 = model.generate(prompt_ids.clone(), max_new_tokens=GENERATE_LEN, temperature=0.8, top_k=40)
print(decode(out_08[0].tolist()))

# --- temperature=1.4 ---
print("\n[temperature=1.4 — high entropy, more chaotic]\n")
out_14 = model.generate(prompt_ids.clone(), max_new_tokens=GENERATE_LEN, temperature=1.4, top_k=40)
print(decode(out_14[0].tolist()))

print("\n" + "=" * 60)
print("Note: greedy output is deterministic. Temperature=1.4 produces")
print("higher-entropy samples that may look less coherent but explore")
print("more of the distribution. top_k=40 prevents very low-probability")
print("tokens from being sampled in the stochastic runs.")
