"""
Tests for lora.py

Run with:
    pytest test_lora.py -v
"""

import math
import sys
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytest

sys.path.insert(0, str(Path(__file__).parent))
from lora import LoRALinear, inject_lora, count_parameters, save_lora_adapter, load_lora_adapter


# ---------------------------------------------------------------------------
# Minimal GPT for testing inject_lora
# ---------------------------------------------------------------------------

@dataclass
class GPTConfig:
    vocab_size: int = 32
    d_model: int = 64
    num_heads: int = 4
    num_layers: int = 2
    max_len: int = 16


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
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
        attn = (q @ k.transpose(-2, -1)) / math.sqrt(D)
        attn = attn.masked_fill(self.mask[:, :, :T, :T] == 0, float("-inf"))
        attn = F.softmax(attn, dim=-1)
        out = (attn @ v).transpose(1, 2).contiguous().view(B, T, C)
        return self.out_proj(out)


class GPTBlock(nn.Module):
    def __init__(self, cfg):
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
    def __init__(self, cfg):
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


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_lora_output_matches_base_at_init():
    """
    At initialization (B=zeros), LoRALinear output equals the base nn.Linear output.
    Use the same weight matrix for both.
    """
    torch.manual_seed(42)
    in_features, out_features = 32, 64
    rank = 4

    base_linear = nn.Linear(in_features, out_features, bias=False)
    weight = base_linear.weight.data  # (out_features, in_features)

    lora_linear = LoRALinear(
        in_features=in_features,
        out_features=out_features,
        rank=rank,
        alpha=16,
        base_weight=weight,
    )

    x = torch.randn(2, 8, in_features)

    with torch.no_grad():
        base_out = base_linear(x)
        lora_out = lora_linear(x)

    # B is zeros at init, so LoRA contribution is zero
    assert torch.allclose(base_out, lora_out, atol=1e-6), (
        f"LoRA output at init should equal base output. "
        f"Max diff: {(base_out - lora_out).abs().max().item():.2e}"
    )


def test_base_weights_frozen():
    """After inject_lora, base weight tensors have requires_grad=False."""
    cfg = GPTConfig()
    model = GPT(cfg)
    inject_lora(model, rank=4, alpha=16, target_modules=("q_proj", "v_proj"))

    for name, module in model.named_modules():
        if isinstance(module, LoRALinear):
            assert not module.W.requires_grad, (
                f"LoRALinear.W at '{name}' should be frozen (requires_grad=False)"
            )


def test_trainable_param_ratio():
    """
    After injecting LoRA (rank=4) into Q and V projections of a small GPT,
    trainable params < 10% of total params.
    """
    cfg = GPTConfig(d_model=128, num_heads=4, num_layers=4)
    model = GPT(cfg)
    inject_lora(model, rank=4, alpha=16, target_modules=("q_proj", "v_proj"))

    trainable, total = count_parameters(model)
    ratio = trainable / total

    assert ratio < 0.10, (
        f"Expected trainable params < 10% of total, got {100 * ratio:.2f}% "
        f"({trainable:,} / {total:,})"
    )


def test_adapter_save_load(tmp_path):
    """
    Save adapter, load it back, verify the A and B matrices are identical.
    Also verify the saved file is smaller than the full model checkpoint.
    """
    cfg = GPTConfig()
    model = GPT(cfg)
    inject_lora(model, rank=4, alpha=16, target_modules=("q_proj", "v_proj"))

    # Modify A and B so they are not the default values (makes comparison meaningful)
    with torch.no_grad():
        for name, param in model.named_parameters():
            if param.requires_grad:
                param.data.fill_(0.123)

    adapter_path = tmp_path / "adapter.pt"
    full_path = tmp_path / "full_model.pt"

    save_lora_adapter(model, adapter_path)
    torch.save(model.state_dict(), full_path)

    # Load into a fresh model with LoRA injected
    model2 = GPT(cfg)
    inject_lora(model2, rank=4, alpha=16, target_modules=("q_proj", "v_proj"))
    load_lora_adapter(model2, adapter_path)

    # Verify A and B matrices match
    original_params = {n: p for n, p in model.named_parameters() if p.requires_grad}
    loaded_params = {n: p for n, p in model2.named_parameters() if p.requires_grad}

    for name, orig in original_params.items():
        assert name in loaded_params, f"Parameter '{name}' not found after load"
        assert torch.allclose(orig, loaded_params[name], atol=1e-6), (
            f"Mismatch in '{name}' after save/load"
        )

    # Verify adapter file is smaller than full model checkpoint
    adapter_size = adapter_path.stat().st_size
    full_size = full_path.stat().st_size
    assert adapter_size < full_size, (
        f"Adapter ({adapter_size} bytes) should be smaller than full checkpoint "
        f"({full_size} bytes)"
    )
