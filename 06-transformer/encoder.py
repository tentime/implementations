import torch
import torch.nn as nn

from attention import MultiHeadAttention


class FeedForward(nn.Module):
    """
    Position-wise feed-forward network: Linear -> ReLU -> Linear.

    Applied identically to each position in the sequence (hence "position-wise").
    The paper uses ReLU; modern models (GPT-3 and beyond) typically use SwiGLU —
    see folder 12 for that variant.

    Args:
        d_model: input/output dimension
        d_ff:    inner hidden dimension (paper uses d_ff = 4 * d_model)
        dropout: applied after the first linear
    """

    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        # x: (batch, seq_len, d_model)
        return self.linear2(self.dropout(torch.relu(self.linear1(x))))


class EncoderLayer(nn.Module):
    """
    Single transformer encoder layer.

    Structure (POST-LayerNorm, as in the original Vaswani et al. 2017 paper):
        x = LayerNorm(x + MultiHeadSelfAttention(x))
        x = LayerNorm(x + FFN(x))

    Note on PRE-LayerNorm: modern models (GPT-2 and later) apply LayerNorm *before*
    the sublayer (PRE-LN). PRE-LN trains more stably without a learning rate warmup
    schedule; POST-LN matches the paper but can be sensitive to initialization.
    See README for more detail.

    Args:
        d_model:   embedding dimension
        num_heads: number of attention heads
        d_ff:      feed-forward hidden dimension
        dropout:   dropout rate
    """

    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.ffn = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, src_mask=None):
        """
        x:        (batch, seq_len, d_model)
        src_mask: optional padding mask for encoder input
        Returns:  (batch, seq_len, d_model)
        """
        # Sub-layer 1: multi-head self-attention + residual + LayerNorm (POST-LN)
        attn_out, _ = self.self_attn(x, context=None, mask=src_mask)
        x = self.norm1(x + self.dropout(attn_out))

        # Sub-layer 2: feed-forward + residual + LayerNorm (POST-LN)
        ffn_out = self.ffn(x)
        x = self.norm2(x + self.dropout(ffn_out))

        return x


class TransformerEncoder(nn.Module):
    """
    Stack of N EncoderLayer blocks.

    Args:
        num_layers: number of encoder layers (N)
        d_model:    embedding dimension
        num_heads:  attention heads per layer
        d_ff:       feed-forward hidden dimension
        dropout:    dropout rate
    """

    def __init__(self, num_layers, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            EncoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(d_model)  # final layer norm

    def forward(self, x, src_mask=None):
        """
        x:        (batch, seq_len, d_model)  — already includes positional encoding
        src_mask: optional padding mask
        Returns:  (batch, seq_len, d_model)
        """
        for layer in self.layers:
            x = layer(x, src_mask=src_mask)
        return self.norm(x)
