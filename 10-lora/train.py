"""
LoRA fine-tuning experiment.

1. Train a base GPT char-level model on Shakespeare text. Save as base_model.pt.
2. Experiment A — Full fine-tune: load base model, fine-tune ALL parameters on legal text.
3. Experiment B — LoRA fine-tune: load base model, inject LoRA into Q/V projections,
   fine-tune only LoRA parameters on the same legal text.

Both fine-tuning runs: 300 steps, print loss every 100 steps.
"""

import math
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from lora import inject_lora, count_parameters, save_lora_adapter

# ---------------------------------------------------------------------------
# Mini GPT (self-contained, ~100 lines)
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

        self.register_buffer(
            "mask",
            torch.tril(torch.ones(cfg.max_len, cfg.max_len)).unsqueeze(0).unsqueeze(0),
        )

    def forward(self, x):
        B, T, C = x.shape
        H, D = self.num_heads, self.head_dim

        q = self.q_proj(x).view(B, T, H, D).transpose(1, 2)
        k = self.k_proj(x).view(B, T, H, D).transpose(1, 2)
        v = self.v_proj(x).view(B, T, H, D).transpose(1, 2)

        scale = math.sqrt(D)
        attn = (q @ k.transpose(-2, -1)) / scale
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

    @torch.no_grad()
    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.cfg.max_len:]
            logits = self(idx_cond)
            probs = F.softmax(logits[:, -1, :], dim=-1)
            next_tok = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, next_tok], dim=1)
        return idx


# ---------------------------------------------------------------------------
# Corpus text
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

LEGAL_TEXT = """whereas the party of the first part hereinafter referred to as the licensor
hereby grants to the party of the second part hereinafter referred to as the licensee
a non-exclusive non-transferable limited license to use the software
subject to the terms and conditions set forth herein
the licensee shall not sublicense sell rent lease transfer assign
or otherwise dispose of the software
the licensor warrants that the software will perform substantially
in accordance with the documentation for a period of ninety days""" * 15

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def build_vocab(text: str):
    chars = sorted(set(text))
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for ch, i in stoi.items()}
    return stoi, itos


def encode(text: str, stoi: dict) -> list:
    return [stoi.get(ch, 0) for ch in text]


def decode(tokens: list, itos: dict) -> str:
    return "".join(itos.get(t, "?") for t in tokens)


def get_batch(data: torch.Tensor, seq_len: int, batch_size: int, device: str):
    ix = torch.randint(0, len(data) - seq_len, (batch_size,))
    x = torch.stack([data[i: i + seq_len] for i in ix]).to(device)
    y = torch.stack([data[i + 1: i + seq_len + 1] for i in ix]).to(device)
    return x, y


def train(model, data: torch.Tensor, cfg: GPTConfig, steps: int,
          lr: float = 3e-4, batch_size: int = 32, device: str = "cpu",
          label: str = ""):
    optimizer = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad], lr=lr
    )
    model.train()
    for step in range(1, steps + 1):
        x, y = get_batch(data, cfg.max_len, batch_size, device)
        logits = model(x)
        loss = F.cross_entropy(logits.view(-1, cfg.vocab_size), y.view(-1))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if step % 100 == 0:
            print(f"  [{label}] step {step}/{steps}  loss={loss.item():.4f}")


def generate_samples(model, cfg: GPTConfig, stoi: dict, itos: dict,
                     prompt: str, n_samples: int = 5, max_new: int = 60,
                     device: str = "cpu") -> list:
    model.eval()
    results = []
    prompt_ids = encode(prompt, stoi)
    for _ in range(n_samples):
        idx = torch.tensor([prompt_ids], dtype=torch.long, device=device)
        out = model.generate(idx, max_new)
        text = decode(out[0].tolist(), itos)
        results.append(text)
    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Build vocab from combined text so both corpora share the same tokenizer
    combined = TEXT + LEGAL_TEXT
    stoi, itos = build_vocab(combined)
    vocab_size = len(stoi)

    cfg = GPTConfig(
        vocab_size=vocab_size,
        d_model=128,
        num_heads=4,
        num_layers=4,
        max_len=64,
    )

    # Encode corpora
    shakespeare_data = torch.tensor(encode(TEXT, stoi), dtype=torch.long)
    legal_data = torch.tensor(encode(LEGAL_TEXT, stoi), dtype=torch.long)

    # -----------------------------------------------------------------------
    # Step 1: Train base model on Shakespeare
    # -----------------------------------------------------------------------
    print("=" * 60)
    print("Step 1: Training base GPT on Shakespeare (500 steps)")
    print("=" * 60)
    base_model = GPT(cfg).to(device)
    train(base_model, shakespeare_data, cfg, steps=500, device=device, label="base")
    torch.save(base_model.state_dict(), "base_model.pt")
    print("Base model saved to base_model.pt\n")

    # -----------------------------------------------------------------------
    # Step 2: Experiment A — Full fine-tune
    # -----------------------------------------------------------------------
    print("=" * 60)
    print("Experiment A: Full fine-tune on legal text (300 steps, all params)")
    print("=" * 60)
    full_ft_model = GPT(cfg).to(device)
    full_ft_model.load_state_dict(torch.load("base_model.pt", map_location=device))
    train(full_ft_model, legal_data, cfg, steps=300, device=device, label="full-FT")

    # -----------------------------------------------------------------------
    # Step 3: Experiment B — LoRA fine-tune
    # -----------------------------------------------------------------------
    print()
    print("=" * 60)
    print("Experiment B: LoRA fine-tune on legal text (300 steps, A+B only)")
    print("=" * 60)
    lora_model = GPT(cfg).to(device)
    lora_model.load_state_dict(torch.load("base_model.pt", map_location=device))
    lora_model = inject_lora(lora_model, rank=4, alpha=16,
                             target_modules=("q_proj", "v_proj"))
    train(lora_model, legal_data, cfg, steps=300, device=device, label="LoRA-FT")

    # -----------------------------------------------------------------------
    # Parameter summary
    # -----------------------------------------------------------------------
    print()
    print("=" * 60)
    print("Parameter counts")
    print("=" * 60)
    base_total = sum(p.numel() for p in base_model.parameters())
    full_trainable, full_total = count_parameters(full_ft_model)
    lora_trainable, lora_total = count_parameters(lora_model)

    print(f"Base model:        {base_total:>10,} params")
    print(f"Full fine-tune:    {full_trainable:>10,} / {full_total:,} trainable")
    print(f"LoRA fine-tune:    {lora_trainable:>10,} / {lora_total:,} trainable "
          f"({100.0 * lora_trainable / lora_total:.2f}%)")

    # -----------------------------------------------------------------------
    # Adapter size comparison
    # -----------------------------------------------------------------------
    print()
    print("=" * 60)
    print("Adapter vs full model size")
    print("=" * 60)
    torch.save(full_ft_model.state_dict(), "full_ft_model.pt")
    save_lora_adapter(lora_model, "lora_adapter.pt")

    full_size_kb = Path("full_ft_model.pt").stat().st_size / 1024
    adapter_size_kb = Path("lora_adapter.pt").stat().st_size / 1024
    print(f"Full fine-tuned model: {full_size_kb:.1f} KB")
    print(f"LoRA adapter:          {adapter_size_kb:.1f} KB  "
          f"({100.0 * adapter_size_kb / full_size_kb:.1f}% of full)")

    # -----------------------------------------------------------------------
    # Text generation comparison
    # -----------------------------------------------------------------------
    prompt = "the party of the first"

    print()
    print("=" * 60)
    print(f'Generated text (prompt: "{prompt}")')
    print("=" * 60)

    print("\n--- Base model (Shakespeare-trained, no fine-tune) ---")
    for s in generate_samples(base_model, cfg, stoi, itos, prompt,
                               n_samples=5, device=device):
        print(f"  {s}")

    print("\n--- Experiment A: Full fine-tune ---")
    for s in generate_samples(full_ft_model, cfg, stoi, itos, prompt,
                               n_samples=5, device=device):
        print(f"  {s}")

    print("\n--- Experiment B: LoRA fine-tune ---")
    for s in generate_samples(lora_model, cfg, stoi, itos, prompt,
                               n_samples=5, device=device):
        print(f"  {s}")


if __name__ == "__main__":
    main()
