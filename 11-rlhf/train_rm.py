"""
Stage 2: Reward Model Training

Train a reward model on synthetic preference pairs.
Chosen: shorter, more focused summaries
Rejected: longer, off-topic or padded completions

Run: python train_rm.py   (requires sft_model.pt from train_sft.py)
Saves: reward_model.pt
"""

import random
import torch
import torch.nn.functional as F
from reward_model import RewardModel, bradley_terry_loss
from train_sft import make_sft_data, build_tokenizer, CharTokenizer


# ---------------------------------------------------------------------------
# Preference pair construction
# ---------------------------------------------------------------------------

def make_worse_completion(prompt, good_completion):
    """
    Create a noticeably worse completion for a given prompt.
    Three degradation strategies (randomly chosen):
      1. Reversed words  — "cat near park" -> "park near cat"
      2. Random padding  — prepend filler words
      3. Repetition      — repeat the first word several times
    """
    random.seed(hash(prompt + good_completion) % (2 ** 31))
    strategy = random.randint(0, 2)

    if strategy == 0:
        # Reverse words
        words = good_completion.split()
        return " ".join(reversed(words)) if len(words) > 1 else good_completion + " " + good_completion

    elif strategy == 1:
        # Pad with irrelevant filler
        fillers = ["things happened", "something occurred", "events unfolded", "it was noted"]
        filler = random.choice(fillers)
        return filler + " and also " + good_completion + " and more"

    else:
        # Repeat first word
        first = good_completion.split()[0] if good_completion.split() else "thing"
        return " ".join([first] * 4)


def make_preference_pairs(sft_data):
    """
    For each (prompt, ideal_completion) pair, create:
      - chosen:   the ideal completion (from SFT data)
      - rejected: a worse version (reversed words / padding / repetition)

    Returns list of (prompt, chosen, rejected) triples.
    """
    triples = []
    for prompt, completion in sft_data:
        rejected = make_worse_completion(prompt, completion)
        triples.append((prompt, completion, rejected))
    return triples


# ---------------------------------------------------------------------------
# Tokenisation helpers
# ---------------------------------------------------------------------------

def encode_pair(tokenizer, prompt, completion, max_len=128):
    """Encode (prompt + completion) as a single token sequence."""
    ids = tokenizer.encode(prompt + " | " + completion)
    ids = ids[:max_len]
    return ids


def pad_to(ids, length, pad_id=0):
    if len(ids) >= length:
        return ids[:length]
    return ids + [pad_id] * (length - len(ids))


def make_rm_batch(triples, tokenizer, device, max_len=64):
    """Return (chosen_ids, rejected_ids) tensors, both padded to max_len."""
    chosen_batch, rejected_batch = [], []
    for prompt, chosen, rejected in triples:
        chosen_batch.append(pad_to(encode_pair(tokenizer, prompt, chosen, max_len), max_len))
        rejected_batch.append(pad_to(encode_pair(tokenizer, prompt, rejected, max_len), max_len))

    chosen_t = torch.tensor(chosen_batch, dtype=torch.long, device=device)
    rejected_t = torch.tensor(rejected_batch, dtype=torch.long, device=device)
    return chosen_t, rejected_t


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_rm(n_steps=300, batch_size=16, lr=1e-3, device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Build data
    sft_data = make_sft_data(n=80)
    tokenizer = build_tokenizer(sft_data)
    triples = make_preference_pairs(sft_data)

    print(f"Training on {len(triples)} preference triples for {n_steps} steps...")

    model = RewardModel(
        vocab_size=tokenizer.vocab_size,
        d_model=64,
        num_heads=4,
        num_layers=2,
        max_len=64,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    for step in range(1, n_steps + 1):
        # Sample mini-batch
        indices = random.sample(range(len(triples)), min(batch_size, len(triples)))
        batch = [triples[i] for i in indices]
        chosen_ids, rejected_ids = make_rm_batch(batch, tokenizer, device)

        r_chosen = model(chosen_ids)    # (B,)
        r_rejected = model(rejected_ids)  # (B,)

        loss = bradley_terry_loss(r_chosen, r_rejected)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if step % 50 == 0:
            # Accuracy: fraction of pairs where r_chosen > r_rejected
            with torch.no_grad():
                acc = (r_chosen > r_rejected).float().mean().item()
            print(f"  step {step:4d} | loss {loss.item():.4f} | accuracy {acc:.2%}")

    # Final accuracy on full dataset
    model.eval()
    correct = 0
    with torch.no_grad():
        for i in range(0, len(triples), batch_size):
            batch = triples[i : i + batch_size]
            chosen_ids, rejected_ids = make_rm_batch(batch, tokenizer, device)
            r_chosen = model(chosen_ids)
            r_rejected = model(rejected_ids)
            correct += (r_chosen > r_rejected).sum().item()
    final_acc = correct / len(triples)
    print(f"\nFinal accuracy on all {len(triples)} pairs: {final_acc:.2%}")

    # Save
    torch.save(
        {
            "model_state": model.state_dict(),
            "vocab_size": tokenizer.vocab_size,
            "tokenizer_ch2id": tokenizer.ch2id,
            "d_model": 64,
            "num_heads": 4,
            "num_layers": 2,
            "max_len": 64,
        },
        "reward_model.pt",
    )
    print("Saved reward_model.pt")
    return model, tokenizer, triples


if __name__ == "__main__":
    train_rm()
