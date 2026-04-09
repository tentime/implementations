"""
Train 6 GPT model sizes on the same dataset and log scaling results.
Produces results.json with (n_params, val_loss) for each model size.
"""

import json
import math
from dataclasses import dataclass, asdict

import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Mini GPT (self-contained, ~120 lines)
# ---------------------------------------------------------------------------

@dataclass
class GPTConfig:
    vocab_size: int
    d_model: int
    num_heads: int
    num_layers: int
    max_len: int


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        assert cfg.d_model % cfg.num_heads == 0
        self.num_heads = cfg.num_heads
        self.head_dim = cfg.d_model // cfg.num_heads

        self.q_proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.k_proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.v_proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.out_proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)

        # Causal mask
        self.register_buffer(
            "mask",
            torch.tril(torch.ones(cfg.max_len, cfg.max_len)).unsqueeze(0).unsqueeze(0),
        )

    def forward(self, x):
        B, T, C = x.shape
        H, D = self.num_heads, self.head_dim

        q = self.q_proj(x).view(B, T, H, D).transpose(1, 2)  # (B, H, T, D)
        k = self.k_proj(x).view(B, T, H, D).transpose(1, 2)
        v = self.v_proj(x).view(B, T, H, D).transpose(1, 2)

        scale = math.sqrt(D)
        attn = (q @ k.transpose(-2, -1)) / scale              # (B, H, T, T)
        attn = attn.masked_fill(self.mask[:, :, :T, :T] == 0, float("-inf"))
        attn = F.softmax(attn, dim=-1)

        out = (attn @ v).transpose(1, 2).contiguous().view(B, T, C)
        return self.out_proj(out)


class GPTBlock(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.d_model)
        self.attn = CausalSelfAttention(cfg)
        self.ln2 = nn.LayerNorm(cfg.d_model)
        self.mlp = nn.Sequential(
            nn.Linear(cfg.d_model, 4 * cfg.d_model),
            nn.GELU(),
            nn.Linear(4 * cfg.d_model, cfg.d_model),
        )

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class GPT(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos_emb = nn.Embedding(cfg.max_len, cfg.d_model)
        self.blocks = nn.Sequential(*[GPTBlock(cfg) for _ in range(cfg.num_layers)])
        self.ln_f = nn.LayerNorm(cfg.d_model)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)

    def forward(self, idx):
        B, T = idx.shape
        pos = torch.arange(T, device=idx.device)
        x = self.tok_emb(idx) + self.pos_emb(pos)
        x = self.blocks(x)
        x = self.ln_f(x)
        return self.lm_head(x)


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


# ---------------------------------------------------------------------------
# Model configs
# ---------------------------------------------------------------------------

CONFIGS = [
    GPTConfig(vocab_size=65, d_model=32,  num_heads=2, num_layers=1, max_len=64),   # ~12K
    GPTConfig(vocab_size=65, d_model=64,  num_heads=4, num_layers=2, max_len=64),   # ~70K
    GPTConfig(vocab_size=65, d_model=128, num_heads=4, num_layers=2, max_len=64),   # ~270K
    GPTConfig(vocab_size=65, d_model=128, num_heads=4, num_layers=4, max_len=64),   # ~530K
    GPTConfig(vocab_size=65, d_model=256, num_heads=8, num_layers=4, max_len=64),   # ~2M
    GPTConfig(vocab_size=65, d_model=256, num_heads=8, num_layers=6, max_len=64),   # ~3M
]

# ---------------------------------------------------------------------------
# Training corpus
# ---------------------------------------------------------------------------

TEXT = """First Citizen: Before we proceed any further, hear me speak.
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
speak this in hunger for bread, not in thirst for revenge.""" * 20

# ---------------------------------------------------------------------------
# Dataset helpers
# ---------------------------------------------------------------------------

def build_vocab(text: str):
    chars = sorted(set(text))
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for ch, i in stoi.items()}
    return stoi, itos


def encode(text: str, stoi: dict) -> list:
    return [stoi[ch] for ch in text]


def get_batch(data: torch.Tensor, seq_len: int, batch_size: int, device):
    ix = torch.randint(0, len(data) - seq_len, (batch_size,))
    x = torch.stack([data[i: i + seq_len] for i in ix]).to(device)
    y = torch.stack([data[i + 1: i + seq_len + 1] for i in ix]).to(device)
    return x, y


# ---------------------------------------------------------------------------
# Training loop for one config
# ---------------------------------------------------------------------------

def train_one(cfg: GPTConfig, train_data: torch.Tensor, val_data: torch.Tensor,
              steps: int = 500, batch_size: int = 32, lr: float = 3e-4,
              device: str = "cpu") -> float:
    model = GPT(cfg).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    model.train()
    for step in range(steps):
        x, y = get_batch(train_data, cfg.max_len, batch_size, device)
        logits = model(x)
        loss = F.cross_entropy(logits.view(-1, cfg.vocab_size), y.view(-1))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Evaluate on validation set
    model.eval()
    with torch.no_grad():
        val_losses = []
        for _ in range(20):
            x, y = get_batch(val_data, cfg.max_len, batch_size, device)
            logits = model(x)
            val_loss = F.cross_entropy(logits.view(-1, cfg.vocab_size), y.view(-1))
            val_losses.append(val_loss.item())
    return sum(val_losses) / len(val_losses)


# ---------------------------------------------------------------------------
# Main sweep
# ---------------------------------------------------------------------------

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    stoi, itos = build_vocab(TEXT)
    data = torch.tensor(encode(TEXT, stoi), dtype=torch.long)

    n = len(data)
    split = int(0.9 * n)
    train_data = data[:split]
    val_data = data[split:]

    results = []

    for i, cfg in enumerate(CONFIGS):
        model_for_count = GPT(cfg)
        n_params = count_params(model_for_count)
        del model_for_count

        val_loss = train_one(cfg, train_data, val_data, steps=500,
                             batch_size=32, lr=3e-4, device=device)

        entry = {
            "n_params": n_params,
            "val_loss": val_loss,
            "config": asdict(cfg),
        }
        results.append(entry)
        print(f"Model {i + 1}/6 | params={n_params:,} | val_loss={val_loss:.4f}")

    with open("results.json", "w") as f:
        json.dump(results, f, indent=2)

    print("\nResults saved to results.json")


if __name__ == "__main__":
    main()
