"""
Char-level GPT trained on a Shakespeare excerpt.

Trains for 2000 steps. Prints loss every 200 steps and a 100-char sample
every 500 steps. Saves a checkpoint to gpt_checkpoint.pt on completion.
"""

import torch
import torch.nn as nn
from gpt import GPT

# ---------------------------------------------------------------------------
# Dataset — char-level Shakespeare (repeated 8× for more data)
# ---------------------------------------------------------------------------
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

# ---------------------------------------------------------------------------
# Char vocabulary
# ---------------------------------------------------------------------------
chars = sorted(set(TEXT))
vocab_size = len(chars)
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for ch, i in stoi.items()}

encode = lambda s: [stoi[c] for c in s]
decode = lambda ids: ''.join(itos[i] for i in ids)

print(f"Vocabulary size (chars): {vocab_size}")

data = torch.tensor(encode(TEXT), dtype=torch.long)
n = len(data)

# ---------------------------------------------------------------------------
# Hyperparameters
# ---------------------------------------------------------------------------
SEQ_LEN = 128
BATCH_SIZE = 32
D_MODEL = 64
NUM_HEADS = 4
NUM_LAYERS = 4
MAX_LEN = 128
NUM_STEPS = 2000
LR = 3e-4

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
model = GPT(
    vocab_size=vocab_size,
    d_model=D_MODEL,
    num_heads=NUM_HEADS,
    num_layers=NUM_LAYERS,
    max_len=MAX_LEN,
    dropout=0.1,
).to(device)

total_params = sum(p.numel() for p in model.parameters())
print(f"Model parameters: {total_params:,}")

optimizer = torch.optim.Adam(model.parameters(), lr=LR)
loss_fn = nn.CrossEntropyLoss()


def get_batch(batch_size, seq_len):
    """Sample random windows from the dataset."""
    starts = torch.randint(0, n - seq_len - 1, (batch_size,))
    x = torch.stack([data[s: s + seq_len] for s in starts])
    y = torch.stack([data[s + 1: s + seq_len + 1] for s in starts])
    return x.to(device), y.to(device)


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------
print("\nTraining...")
model.train()
for step in range(1, NUM_STEPS + 1):
    x, y = get_batch(BATCH_SIZE, SEQ_LEN)

    logits = model(x)  # (batch, seq_len, vocab_size)
    # Predict the next character at every position
    loss = loss_fn(logits.view(-1, vocab_size), y.view(-1))

    optimizer.zero_grad()
    loss.backward()
    # Gradient clipping — important for stability with char-level models
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()

    if step % 200 == 0:
        print(f"  step {step:4d} | loss {loss.item():.4f}")

    if step % 500 == 0:
        # Generate a 100-char sample from the newline character as prompt
        model.eval()
        prompt_ids = torch.tensor([[stoi['\n']]], dtype=torch.long, device=device)
        generated = model.generate(prompt_ids, max_new_tokens=100, temperature=0.8)
        sample = decode(generated[0].tolist())
        print(f"\n--- Sample at step {step} ---")
        print(sample)
        print("---\n")
        model.train()

print("Training complete.")

# ---------------------------------------------------------------------------
# Save checkpoint
# ---------------------------------------------------------------------------
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
torch.save(checkpoint, 'gpt_checkpoint.pt')
print("Checkpoint saved to gpt_checkpoint.pt")
