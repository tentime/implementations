import random
import torch
import torch.nn as nn
import torch.optim as optim

from encoder import BidirectionalEncoder
from decoder import Decoder

# ---------------------------------------------------------------------------
# Data generation
# ---------------------------------------------------------------------------

MONTHS = ["january", "february", "march", "april", "may", "june",
          "july", "august", "september", "october", "november", "december"]
SUFFIXES = ["st", "nd", "rd", "th"]


def generate_date_pair(seed):
    random.seed(seed)
    month_idx = random.randint(0, 11)
    day = random.randint(1, 28)
    year = random.randint(2000, 2025)
    suffix = SUFFIXES[min(day - 1, 3)]
    src = f"{MONTHS[month_idx]} {day}{suffix} {year}"
    tgt = f"{year:04d}-{month_idx + 1:02d}-{day:02d}"
    return src, tgt


def build_dataset(n=500):
    return [generate_date_pair(i) for i in range(n)]


# ---------------------------------------------------------------------------
# Character-level vocabulary
# ---------------------------------------------------------------------------

PAD, SOS, EOS = "<PAD>", "<SOS>", "<EOS>"


def build_vocab(pairs):
    chars = set()
    for src, tgt in pairs:
        chars.update(src)
        chars.update(tgt)
    vocab = [PAD, SOS, EOS] + sorted(chars)
    char2idx = {c: i for i, c in enumerate(vocab)}
    idx2char = {i: c for c, i in char2idx.items()}
    return char2idx, idx2char


def encode(text, char2idx):
    return [char2idx[SOS]] + [char2idx[c] for c in text] + [char2idx[EOS]]


def decode_tokens(indices, idx2char):
    chars = []
    for idx in indices:
        ch = idx2char.get(idx, "?")
        if ch == EOS:
            break
        if ch in (SOS, PAD):
            continue
        chars.append(ch)
    return "".join(chars)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class Seq2SeqAttention(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_size, attn_dim):
        super().__init__()
        encoder_dim = 2 * hidden_size  # bidirectional
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.encoder = BidirectionalEncoder(embed_dim, hidden_size)
        # Project encoder final state to decoder hidden size
        self.enc2dec_h = nn.Linear(encoder_dim, hidden_size)
        self.decoder = Decoder(vocab_size, embed_dim, encoder_dim, hidden_size, attn_dim)

    def forward(self, src_indices, tgt_indices, teacher_forcing_ratio=0.5):
        """
        src_indices: (src_len,) int tensor
        tgt_indices: (tgt_len,) int tensor (includes SOS at front, EOS at end)
        Returns logits: (tgt_len-1, vocab_size)
        """
        # Embed source
        src_embedded = self.embedding(src_indices)     # (src_len, embed_dim)
        encoder_outputs, final_h = self.encoder(src_embedded)  # (src_len, 2*H), (2*H,)

        # Project encoder final state to decoder initial hidden
        dec_h = torch.tanh(self.enc2dec_h(final_h))   # (hidden_size,)
        dec_c = torch.zeros_like(dec_h)

        # Decode: input is tgt[:-1], target is tgt[1:]
        tgt_in = tgt_indices[:-1]   # drop last EOS for input
        tgt_out = tgt_indices[1:]   # drop first SOS for target

        sos_token = tgt_indices[0].item()
        logits, _ = self.decoder(
            encoder_outputs, dec_h, dec_c,
            sos_token=sos_token,
            target_seq=tgt_in,
            teacher_forcing_ratio=teacher_forcing_ratio,
        )
        return logits  # (tgt_len-1, vocab_size)

    def predict(self, src_indices, sos_token, eos_token, max_len=30):
        """Greedy inference."""
        with torch.no_grad():
            src_embedded = self.embedding(src_indices)
            encoder_outputs, final_h = self.encoder(src_embedded)
            dec_h = torch.tanh(self.enc2dec_h(final_h))
            dec_c = torch.zeros_like(dec_h)
            tokens, attn = self.decoder.greedy_decode(
                encoder_outputs, dec_h, dec_c,
                sos_token, eos_token, max_len=max_len,
            )
        return tokens, attn


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train():
    # Reproducibility
    torch.manual_seed(42)
    random.seed(42)

    # Data
    all_pairs = build_dataset(500)
    train_pairs = all_pairs[:450]
    test_pairs = all_pairs[450:]

    char2idx, idx2char = build_vocab(all_pairs)
    vocab_size = len(char2idx)
    sos_idx = char2idx[SOS]
    eos_idx = char2idx[EOS]

    def to_tensor(text):
        return torch.tensor(encode(text, char2idx), dtype=torch.long)

    train_data = [(to_tensor(s), to_tensor(t)) for s, t in train_pairs]
    test_data = [(to_tensor(s), to_tensor(t), s, t) for s, t, in [(s, t) for s, t in test_pairs]]

    # Hyperparameters
    EMBED_DIM = 32
    HIDDEN_SIZE = 64
    ATTN_DIM = 32
    LR = 0.001
    STEPS = 2000
    TEACHER_FORCING = 0.5

    model = Seq2SeqAttention(vocab_size, EMBED_DIM, HIDDEN_SIZE, ATTN_DIM)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss(ignore_index=0)

    print(f"Vocab size: {vocab_size}")
    print(f"Train examples: {len(train_data)}, Test examples: {len(test_pairs)}")
    print(f"Training for {STEPS} steps...\n")

    model.train()
    for step in range(1, STEPS + 1):
        # Sample a random training example
        idx = random.randint(0, len(train_data) - 1)
        src, tgt = train_data[idx]

        optimizer.zero_grad()
        logits = model(src, tgt, teacher_forcing_ratio=TEACHER_FORCING)
        # tgt[1:] are the target tokens (after SOS)
        targets = tgt[1:]
        loss = criterion(logits, targets)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        if step % 200 == 0:
            print(f"Step {step:4d} | Loss: {loss.item():.4f}")

    # ---------------------------------------------------------------------------
    # Inference on 5 test examples
    # ---------------------------------------------------------------------------
    print("\n--- Inference on 5 test examples ---")
    model.eval()
    for i, (s, t) in enumerate(test_pairs[:5]):
        src_tensor = to_tensor(s)
        pred_tokens, _ = model.predict(src_tensor, sos_idx, eos_idx)
        predicted = decode_tokens(pred_tokens, idx2char)
        print(f"  Source:    {s}")
        print(f"  Predicted: {predicted}")
        print(f"  Target:    {t}")
        print()

    # Save model + vocab for visualize.py
    torch.save({
        "model_state": model.state_dict(),
        "char2idx": char2idx,
        "idx2char": idx2char,
        "vocab_size": vocab_size,
        "embed_dim": EMBED_DIM,
        "hidden_size": HIDDEN_SIZE,
        "attn_dim": ATTN_DIM,
        "test_pairs": test_pairs,
    }, "seq2seq_checkpoint.pt")
    print("Checkpoint saved to seq2seq_checkpoint.pt")

    return model, char2idx, idx2char, test_pairs


if __name__ == "__main__":
    train()
