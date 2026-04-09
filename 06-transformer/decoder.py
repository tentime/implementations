import torch
import torch.nn as nn

from attention import MultiHeadAttention
from encoder import FeedForward


class DecoderLayer(nn.Module):
    """
    Single transformer decoder layer. Three sub-layers:

    1. Masked multi-head self-attention — the decoder can only attend to
       positions up to and including the current one (causal mask). This
       preserves the autoregressive property: when generating token t, the
       model cannot see tokens t+1, t+2, ...
    2. Add & Norm
    3. Cross-attention (encoder-decoder attention) — Q comes from the decoder,
       K and V come from the encoder output. This is how the decoder reads
       the source sequence.
    4. Add & Norm
    5. Position-wise feed-forward network
    6. Add & Norm

    POST-LayerNorm ordering matches the original Vaswani et al. 2017 paper.

    Args:
        d_model:   embedding dimension
        num_heads: number of attention heads
        d_ff:      feed-forward hidden dimension
        dropout:   dropout rate
    """

    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn  = MultiHeadAttention(d_model, num_heads)   # masked self-attention
        self.cross_attn = MultiHeadAttention(d_model, num_heads)   # cross-attention
        self.ffn        = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, encoder_output, tgt_mask=None, src_mask=None):
        """
        x:             (batch, tgt_len, d_model)  — decoder input (target sequence so far)
        encoder_output:(batch, src_len, d_model)  — encoder hidden states
        tgt_mask:      causal mask (batch, 1, tgt_len, tgt_len) or broadcastable
        src_mask:      padding mask for the encoder output (optional)

        Returns: (batch, tgt_len, d_model)
        """
        # 1. Masked self-attention: Q=K=V=x, apply causal mask
        self_attn_out, _ = self.self_attn(x, context=None, mask=tgt_mask)
        x = self.norm1(x + self.dropout(self_attn_out))

        # 2. Cross-attention: Q from decoder, K and V from encoder
        cross_attn_out, _ = self.cross_attn(x, context=encoder_output, mask=src_mask)
        x = self.norm2(x + self.dropout(cross_attn_out))

        # 3. Feed-forward
        ffn_out = self.ffn(x)
        x = self.norm3(x + self.dropout(ffn_out))

        return x


class TransformerDecoder(nn.Module):
    """
    Stack of N DecoderLayer blocks.

    Args:
        num_layers: number of decoder layers (N)
        d_model:    embedding dimension
        num_heads:  attention heads per layer
        d_ff:       feed-forward hidden dimension
        dropout:    dropout rate
    """

    def __init__(self, num_layers, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            DecoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(d_model)  # final layer norm

    def forward(self, x, encoder_output, tgt_mask=None, src_mask=None):
        """
        x:             (batch, tgt_len, d_model)  — already includes positional encoding
        encoder_output:(batch, src_len, d_model)
        tgt_mask:      causal mask
        src_mask:      encoder padding mask

        Returns: (batch, tgt_len, d_model)
        """
        for layer in self.layers:
            x = layer(x, encoder_output, tgt_mask=tgt_mask, src_mask=src_mask)
        return self.norm(x)
