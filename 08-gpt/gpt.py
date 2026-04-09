import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def scaled_dot_product_attention(Q, K, V, mask=None):
    """
    Compute Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) V

    Written out again here rather than imported — this folder is meant to be
    self-contained, and the function is short enough to warrant it.

    Args:
        Q: (batch, heads, seq_q, d_k)
        K: (batch, heads, seq_k, d_k)
        V: (batch, heads, seq_k, d_k)
        mask: optional tensor broadcastable to (batch, heads, seq_q, seq_k)
              Positions where mask == 0 are set to -inf before softmax.
    Returns:
        output:  (batch, heads, seq_q, d_k)
        weights: (batch, heads, seq_q, seq_k)
    """
    d_k = Q.shape[-1]
    scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float('-inf'))
    weights = F.softmax(scores, dim=-1)
    output = torch.matmul(weights, V)
    return output, weights


class CausalSelfAttention(nn.Module):
    """
    Multi-head self-attention with a causal (lower-triangular) mask.

    This is the ONLY architectural difference between BERT's TransformerBlock
    and GPT's GPTBlock at the attention level: BERT passes no mask (attends
    everywhere); GPT passes a lower-triangular mask so each position can only
    attend to positions at or before itself. That single change converts a
    bidirectional encoder into an autoregressive decoder.

    The mask is created once in __init__ and registered as a buffer so it:
      - is not a learnable parameter
      - moves to the correct device automatically with .to(device)
      - is included in state_dict (so checkpoints are self-contained)
    """

    def __init__(self, d_model, num_heads, max_len):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        # Per-head projections (explicit loop, same style as folders 06/07)
        self.W_q = nn.ModuleList([nn.Linear(d_model, self.d_k, bias=False) for _ in range(num_heads)])
        self.W_k = nn.ModuleList([nn.Linear(d_model, self.d_k, bias=False) for _ in range(num_heads)])
        self.W_v = nn.ModuleList([nn.Linear(d_model, self.d_k, bias=False) for _ in range(num_heads)])
        self.W_o = nn.Linear(d_model, d_model, bias=False)

        # Causal mask: lower-triangular ones matrix, shape (max_len, max_len).
        # Registered as a buffer — created once, reused every forward pass.
        self.register_buffer('mask', torch.tril(torch.ones(max_len, max_len)))

    def forward(self, x):
        """
        x: (batch, seq_len, d_model)
        Returns: (batch, seq_len, d_model)
        """
        batch, seq_len, _ = x.shape

        # Crop the pre-built mask to the current sequence length
        causal_mask = self.mask[:seq_len, :seq_len]  # (seq_len, seq_len)
        # Unsqueeze for (batch=1, heads=1) broadcasting
        causal_mask = causal_mask.unsqueeze(0).unsqueeze(0)  # (1, 1, seq_len, seq_len)

        head_outputs = []
        for h in range(self.num_heads):
            Q = self.W_q[h](x).unsqueeze(1)  # (batch, 1, seq_len, d_k)
            K = self.W_k[h](x).unsqueeze(1)
            V = self.W_v[h](x).unsqueeze(1)
            out, _ = scaled_dot_product_attention(Q, K, V, mask=causal_mask)
            head_outputs.append(out.squeeze(1))  # (batch, seq_len, d_k)

        concat = torch.cat(head_outputs, dim=-1)  # (batch, seq_len, d_model)
        return self.W_o(concat)


class GPTBlock(nn.Module):
    """
    PRE-LayerNorm transformer block — GPT-2 style.

    Structure:
        x = x + CausalSelfAttention(LayerNorm(x))   # pre-LN attention
        x = x + FFN(LayerNorm(x))                    # pre-LN FFN

    The original 2017 transformer (Vaswani et al.) used POST-LN:
        x = LayerNorm(x + sublayer(x))

    GPT-2 switched to PRE-LN for training stability with deeper models: the
    residual stream stays on a stable scale regardless of depth because the
    norm is applied inside the branch, not across the skip connection. This
    is now the universal standard (GPT-2, GPT-3, PaLM, LLaMA, Mistral, …).
    """

    def __init__(self, d_model, num_heads, d_ff, max_len, dropout=0.0):
        super().__init__()
        self.attn = CausalSelfAttention(d_model, num_heads, max_len)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # Pre-LN: norm before attention, then residual
        x = x + self.dropout(self.attn(self.norm1(x)))
        # Pre-LN: norm before FFN, then residual
        x = x + self.dropout(self.ffn(self.norm2(x)))
        return x


class GPT(nn.Module):
    """
    GPT language model (decoder-only transformer).

    Key design choices:
    - LEARNED positional embeddings: GPT-2 deviated from the original paper's
      sinusoidal PE and used a learned embedding table instead. Empirically the
      difference is small, but learned PE is now more common in practice.
    - TIED input/output embeddings: lm_head.weight IS token_embedding.weight.
      Same reasoning as BERT's MLMHead — input and output live in the same
      semantic space, and tying halves the vocabulary parameter count.
    - Stack of GPTBlock layers with causal masking throughout.

    Args:
        vocab_size: size of the character/token vocabulary
        d_model:    embedding dimension
        num_heads:  number of attention heads (must divide d_model)
        num_layers: depth of the transformer stack
        max_len:    maximum sequence length (for positional embedding table)
        dropout:    dropout rate (0.0 for small models / inference)
    """

    def __init__(self, vocab_size, d_model, num_heads, num_layers, max_len, dropout=0.0):
        super().__init__()
        self.max_len = max_len

        # Token embedding (input)
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        # Learned positional embedding — one vector per position up to max_len
        self.position_embedding = nn.Embedding(max_len, d_model)

        self.dropout = nn.Dropout(dropout)
        self.blocks = nn.ModuleList(
            [GPTBlock(d_model, num_heads, d_ff=d_model * 4, max_len=max_len, dropout=dropout)
             for _ in range(num_layers)]
        )
        # Final LayerNorm (PRE-LN stacks need a norm after the last block)
        self.norm = nn.LayerNorm(d_model)

        # Output projection: hidden → vocab logits
        # Weight tying: lm_head shares the token embedding matrix
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.lm_head.weight = self.token_embedding.weight  # tied — same object

        self._init_weights()

        # Sanity-check weight tying at construction time
        assert self.lm_head.weight is self.token_embedding.weight, (
            "Weight tying failed: lm_head.weight is not the same object as token_embedding.weight"
        )

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, idx):
        """
        idx: (batch, seq_len) — integer token ids
        Returns: logits (batch, seq_len, vocab_size)
        """
        batch, seq_len = idx.shape
        assert seq_len <= self.max_len, (
            f"Sequence length {seq_len} exceeds max_len {self.max_len}"
        )
        device = idx.device

        positions = torch.arange(seq_len, device=device).unsqueeze(0)  # (1, seq_len)
        x = self.dropout(self.token_embedding(idx) + self.position_embedding(positions))

        for block in self.blocks:
            x = block(x)

        x = self.norm(x)
        logits = self.lm_head(x)  # (batch, seq_len, vocab_size)
        return logits

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """
        Autoregressive generation — append max_new_tokens one at a time.

        For each new token:
        1. Crop idx to max_len if needed (the model has a fixed positional table)
        2. Forward pass to get logits at the last position only
        3. Apply temperature scaling (divide logits by temperature)
        4. Optionally zero out all but the top-k logits before sampling
        5. Sample from the resulting distribution (or argmax if temperature=0)
        6. Append the sampled token to idx and repeat

        Args:
            idx:            (batch, seq_len) — prompt token ids
            max_new_tokens: number of tokens to generate
            temperature:    controls randomness; 0 → greedy argmax
            top_k:          if set, only sample from the k highest-probability tokens

        Returns:
            (batch, seq_len + max_new_tokens) — prompt + generated tokens
        """
        self.eval()
        for _ in range(max_new_tokens):
            # Crop to max_len so the positional embedding table is not exceeded
            idx_cond = idx if idx.shape[1] <= self.max_len else idx[:, -self.max_len:]

            logits = self.forward(idx_cond)        # (batch, seq_len, vocab_size)
            logits = logits[:, -1, :]              # last position: (batch, vocab_size)

            if temperature == 0:
                # Greedy: deterministic argmax
                next_token = logits.argmax(dim=-1, keepdim=True)  # (batch, 1)
            else:
                logits = logits / temperature

                if top_k is not None:
                    # Zero out all logits outside the top-k
                    # torch.topk returns (values, indices); we want a threshold
                    top_values, _ = torch.topk(logits, min(top_k, logits.shape[-1]))
                    # The k-th largest value per row
                    threshold = top_values[:, -1].unsqueeze(-1)
                    logits = logits.masked_fill(logits < threshold, float('-inf'))

                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)  # (batch, 1)

            idx = torch.cat([idx, next_token], dim=1)

        return idx
