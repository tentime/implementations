import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def scaled_dot_product_attention(Q, K, V, mask=None):
    """
    Compute Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) V

    Args:
        Q: (batch, heads, seq_q, d_k)
        K: (batch, heads, seq_k, d_k)
        V: (batch, heads, seq_k, d_v)
        mask: optional boolean tensor broadcastable to (batch, heads, seq_q, seq_k)
              True  = keep this position
              False = mask it out (set to -inf before softmax)
    Returns:
        output:  (batch, heads, seq_q, d_v)
        weights: (batch, heads, seq_q, seq_k)
    """
    d_k = Q.shape[-1]

    # Scaled dot product: (batch, heads, seq_q, seq_k)
    scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)

    # Additive masking: set masked positions to -inf before softmax
    # so they become 0 after softmax (exp(-inf) = 0)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float('-inf'))

    weights = F.softmax(scores, dim=-1)  # (batch, heads, seq_q, seq_k)
    output = torch.matmul(weights, V)    # (batch, heads, seq_q, d_v)
    return output, weights


class MultiHeadAttention(nn.Module):
    """
    Multi-head attention with an explicit per-head loop (clarity > speed).
    Supports both self-attention and cross-attention via the context parameter.

    For self-attention:   call forward(x)
    For cross-attention:  call forward(x, context=encoder_output)

    Each head has its own Q, K, V projection matrices.
    After all heads are computed their outputs are concatenated and projected
    through a final Wo matrix.

    Args:
        d_model:   total embedding dimension
        num_heads: number of attention heads
                   (d_model must be divisible by num_heads)
    """

    def __init__(self, d_model, num_heads):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.num_heads = num_heads
        self.d_k = d_model // num_heads  # dimension per head

        # Separate Q, K, V projection for each head
        self.W_q = nn.ModuleList([nn.Linear(d_model, self.d_k, bias=False) for _ in range(num_heads)])
        self.W_k = nn.ModuleList([nn.Linear(d_model, self.d_k, bias=False) for _ in range(num_heads)])
        self.W_v = nn.ModuleList([nn.Linear(d_model, self.d_k, bias=False) for _ in range(num_heads)])

        # Output projection: concat of all heads → d_model
        self.W_o = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x, context=None, mask=None):
        """
        x:       (batch, seq_q, d_model)  — queries always come from here
        context: (batch, seq_k, d_model)  — keys and values come from here;
                  if None, this is self-attention (context = x)
        mask:    (batch, 1, seq_q, seq_k) or broadcastable boolean tensor

        Returns:
            output:       (batch, seq_q, d_model)
            attn_weights: (batch, num_heads, seq_q, seq_k)  — for inspection/visualization
        """
        if context is None:
            context = x  # self-attention: Q, K, V all from x

        batch_size = x.shape[0]
        head_outputs = []
        head_weights = []

        for h in range(self.num_heads):
            Q = self.W_q[h](x)        # (batch, seq_q, d_k)
            K = self.W_k[h](context)  # (batch, seq_k, d_k)
            V = self.W_v[h](context)  # (batch, seq_k, d_k)

            # Add heads dimension for scaled_dot_product_attention
            Q = Q.unsqueeze(1)  # (batch, 1, seq_q, d_k)
            K = K.unsqueeze(1)  # (batch, 1, seq_k, d_k)
            V = V.unsqueeze(1)  # (batch, 1, seq_k, d_k)

            out, weights = scaled_dot_product_attention(Q, K, V, mask=mask)
            # out: (batch, 1, seq_q, d_k)
            head_outputs.append(out.squeeze(1))   # (batch, seq_q, d_k)
            head_weights.append(weights.squeeze(1))  # (batch, seq_q, seq_k)

        # Concatenate all head outputs along the feature dimension
        concat = torch.cat(head_outputs, dim=-1)  # (batch, seq_q, d_model)
        output = self.W_o(concat)                 # (batch, seq_q, d_model)

        # Stack weights for inspection: (batch, num_heads, seq_q, seq_k)
        attn_weights = torch.stack(head_weights, dim=1)

        return output, attn_weights


class SinusoidalPositionalEncoding(nn.Module):
    """
    Fixed (non-learned) positional encoding from Vaswani et al. 2017.

    PE(pos, 2i)   = sin(pos / 10000^(2i / d_model))
    PE(pos, 2i+1) = cos(pos / 10000^(2i / d_model))

    Precomputed for max_len positions and stored as a buffer (not a parameter).
    Added to token embeddings before the first encoder/decoder layer.

    Args:
        d_model: embedding dimension
        max_len: maximum sequence length to precompute
        dropout: applied after adding positional encoding
    """

    def __init__(self, d_model, max_len=512, dropout=0.0):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Build the PE table: shape (max_len, d_model)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)  # (max_len, 1)
        # Denominator: 10000^(2i / d_model) — computed in log space for numerical stability
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float) * (-math.log(10000.0) / d_model)
        )  # (d_model/2,)

        pe[:, 0::2] = torch.sin(position * div_term)   # even indices
        pe[:, 1::2] = torch.cos(position * div_term)   # odd indices

        # Shape (1, max_len, d_model) so it broadcasts over batch
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        """
        x: (batch, seq_len, d_model)
        Returns: (batch, seq_len, d_model) with positional encoding added
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)
