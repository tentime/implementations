import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# ---------------------------------------------------------------------------
# Mini GPT-style transformer (self-contained)
# ---------------------------------------------------------------------------

class CausalSelfAttention(nn.Module):
    def __init__(self, d_model, num_heads, max_len):
        super().__init__()
        assert d_model % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.proj = nn.Linear(d_model, d_model, bias=False)
        # Causal mask: lower-triangular
        mask = torch.tril(torch.ones(max_len, max_len)).unsqueeze(0).unsqueeze(0)
        self.register_buffer("mask", mask)

    def forward(self, x):
        B, T, C = x.shape
        qkv = self.qkv(x)  # (B, T, 3C)
        q, k, v = qkv.split(C, dim=-1)

        def split_heads(t):
            return t.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        q, k, v = split_heads(q), split_heads(k), split_heads(v)
        scale = self.head_dim ** -0.5
        attn = (q @ k.transpose(-2, -1)) * scale  # (B, H, T, T)
        attn = attn.masked_fill(self.mask[:, :, :T, :T] == 0, float("-inf"))
        attn = F.softmax(attn, dim=-1)
        out = attn @ v  # (B, H, T, head_dim)
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.proj(out)


class TransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads, max_len):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, num_heads, max_len)
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model),
        )

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x


class GPTBase(nn.Module):
    """Shared backbone: token embedding + positional embedding + transformer blocks."""

    def __init__(self, vocab_size, d_model, num_heads, num_layers, max_len):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_len, d_model)
        self.blocks = nn.ModuleList(
            [TransformerBlock(d_model, num_heads, max_len) for _ in range(num_layers)]
        )
        self.ln_f = nn.LayerNorm(d_model)
        self.max_len = max_len

    def forward(self, token_ids):
        B, T = token_ids.shape
        assert T <= self.max_len, f"Sequence length {T} exceeds max_len {self.max_len}"
        positions = torch.arange(T, device=token_ids.device).unsqueeze(0)  # (1, T)
        x = self.tok_emb(token_ids) + self.pos_emb(positions)
        for block in self.blocks:
            x = block(x)
        return self.ln_f(x)  # (B, T, d_model)


# ---------------------------------------------------------------------------
# Reward Model
# ---------------------------------------------------------------------------

class RewardModel(nn.Module):
    """
    Takes a (prompt + completion) token sequence, runs it through a transformer,
    pools the final hidden state, and outputs a scalar reward.

    Architecture:
    - GPT-style transformer (shared with policy, but separate instance)
    - Linear head: d_model -> 1 (scalar reward)

    The reward represents "how good is this completion for this prompt?"
    """

    def __init__(self, vocab_size, d_model=64, num_heads=4, num_layers=2, max_len=128):
        super().__init__()
        self.backbone = GPTBase(vocab_size, d_model, num_heads, num_layers, max_len)
        self.reward_head = nn.Linear(d_model, 1, bias=False)

    def forward(self, token_ids):
        """
        Returns reward scalar per sequence. Shape: (batch,)

        We take the hidden state at the *last* position of each sequence.
        In a causal transformer this position has attended to all prior tokens,
        making it a natural summary of the full (prompt + completion).
        """
        hidden = self.backbone(token_ids)  # (B, T, d_model)
        last_hidden = hidden[:, -1, :]    # (B, d_model)
        reward = self.reward_head(last_hidden).squeeze(-1)  # (B,)
        return reward


# ---------------------------------------------------------------------------
# Bradley-Terry loss
# ---------------------------------------------------------------------------

def bradley_terry_loss(reward_chosen, reward_rejected):
    """
    Bradley-Terry preference model loss:
        L = -log(sigmoid(r_chosen - r_rejected))

    This is the standard loss for training reward models from pairwise preferences.
    Lower loss = model correctly predicts chosen > rejected.

    Derivation:
        P(chosen > rejected) = sigmoid(r_chosen - r_rejected)
        MLE => maximize log P => minimize -log P
    """
    return -F.logsigmoid(reward_chosen - reward_rejected).mean()
