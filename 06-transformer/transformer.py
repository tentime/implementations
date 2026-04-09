import torch
import torch.nn as nn

from attention import SinusoidalPositionalEncoding
from encoder import TransformerEncoder
from decoder import TransformerDecoder


def make_causal_mask(seq_len, device):
    """
    Lower-triangular boolean mask: position i can attend to positions 0..i.
    Positions above the diagonal are masked out (False → -inf in attention).

    Shape: (seq_len, seq_len)
    Broadcasting in MultiHeadAttention handles the (batch, heads) dimensions.
    """
    return torch.tril(torch.ones(seq_len, seq_len, device=device)).bool()


class Transformer(nn.Module):
    """
    Full encoder-decoder transformer (Vaswani et al. 2017).

    Architecture:
        Encoder: embed + positional encoding → N × EncoderLayer
        Decoder: embed + positional encoding → N × DecoderLayer (with cross-attention)
        Output:  linear projection to target vocabulary logits

    Args:
        src_vocab_size: source vocabulary size
        tgt_vocab_size: target vocabulary size
        d_model:   embedding dimension (e.g. 64)
        num_heads: attention heads (must divide d_model evenly, e.g. 4)
        num_layers:encoder and decoder depth (e.g. 2)
        d_ff:      feed-forward hidden dimension (e.g. 128)
        max_len:   maximum sequence length for positional encoding
        dropout:   dropout rate applied throughout
    """

    def __init__(
        self,
        src_vocab_size,
        tgt_vocab_size,
        d_model=64,
        num_heads=4,
        num_layers=2,
        d_ff=128,
        max_len=512,
        dropout=0.1,
    ):
        super().__init__()
        self.d_model = d_model

        # Source and target embeddings (separate, even if vocabs overlap)
        self.src_embedding = nn.Embedding(src_vocab_size, d_model)
        self.tgt_embedding = nn.Embedding(tgt_vocab_size, d_model)

        # Shared positional encoding (same formula for both sides)
        self.pos_encoding = SinusoidalPositionalEncoding(d_model, max_len, dropout)

        # Encoder and decoder stacks
        self.encoder = TransformerEncoder(num_layers, d_model, num_heads, d_ff, dropout)
        self.decoder = TransformerDecoder(num_layers, d_model, num_heads, d_ff, dropout)

        # Final linear to project decoder output to target vocab logits
        # (no softmax — use CrossEntropyLoss which includes log-softmax)
        self.output_proj = nn.Linear(d_model, tgt_vocab_size)

        self._init_weights()

    def _init_weights(self):
        """Xavier uniform initialization for all linear and embedding weights."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def encode(self, src, src_mask=None):
        """
        Encode a source sequence.

        src:      (batch, src_len) integer token ids
        src_mask: optional padding mask

        Returns: (batch, src_len, d_model)
        """
        import math
        x = self.src_embedding(src) * math.sqrt(self.d_model)  # scale as in paper
        x = self.pos_encoding(x)
        return self.encoder(x, src_mask=src_mask)

    def decode(self, tgt, encoder_output, tgt_mask=None, src_mask=None):
        """
        Decode one step (or a full sequence in parallel during training).

        tgt:           (batch, tgt_len) integer token ids
        encoder_output:(batch, src_len, d_model)

        Returns: (batch, tgt_len, d_model)
        """
        import math
        x = self.tgt_embedding(tgt) * math.sqrt(self.d_model)
        x = self.pos_encoding(x)
        return self.decoder(x, encoder_output, tgt_mask=tgt_mask, src_mask=src_mask)

    def forward(self, src, tgt, src_mask=None):
        """
        Full forward pass: encode source, then decode target.

        src: (batch, src_len) — source token ids
        tgt: (batch, tgt_len) — target token ids (SOS + ground truth, without EOS)

        Returns: (batch, tgt_len, tgt_vocab_size) — logits (un-normalized)
        """
        tgt_len = tgt.shape[1]
        device = src.device

        # Causal mask for target self-attention: shape (tgt_len, tgt_len)
        # Unsqueeze to (1, 1, tgt_len, tgt_len) for broadcasting over batch and heads
        causal_mask = make_causal_mask(tgt_len, device).unsqueeze(0).unsqueeze(0)

        encoder_output = self.encode(src, src_mask=src_mask)
        decoder_output = self.decode(tgt, encoder_output, tgt_mask=causal_mask, src_mask=src_mask)

        logits = self.output_proj(decoder_output)  # (batch, tgt_len, tgt_vocab_size)
        return logits

    def greedy_decode(self, src, sos_token, eos_token, max_len=50, src_mask=None):
        """
        Autoregressive greedy decoding for inference.

        src: (1, src_len) — single source sequence (batch size 1)
        Returns: list of integer token ids (without SOS, up to and including EOS)
        """
        device = src.device
        encoder_output = self.encode(src, src_mask=src_mask)

        # Start with just the SOS token
        tgt = torch.tensor([[sos_token]], dtype=torch.long, device=device)

        generated = []
        for _ in range(max_len):
            tgt_len = tgt.shape[1]
            causal_mask = make_causal_mask(tgt_len, device).unsqueeze(0).unsqueeze(0)
            dec_out = self.decode(tgt, encoder_output, tgt_mask=causal_mask)
            logits = self.output_proj(dec_out)          # (1, tgt_len, vocab)
            next_token = logits[0, -1, :].argmax().item()  # greedy pick from last position
            generated.append(next_token)
            if next_token == eos_token:
                break
            tgt = torch.cat(
                [tgt, torch.tensor([[next_token]], dtype=torch.long, device=device)], dim=1
            )

        return generated
