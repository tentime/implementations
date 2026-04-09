import torch
import torch.nn as nn
import torch.nn.functional as F


class BahdanauAttention(nn.Module):
    """
    Bahdanau (additive) attention: score(h_t, s_{t-1}) = v^T tanh(W1 h_t + W2 s_{t-1})

    Args:
        encoder_dim: size of encoder hidden states (2*hidden for bidirectional)
        decoder_dim: size of decoder hidden state
        attn_dim: size of the intermediate attention space
    """

    def __init__(self, encoder_dim, decoder_dim, attn_dim=32):
        super().__init__()
        self.W1 = nn.Linear(encoder_dim, attn_dim, bias=False)
        self.W2 = nn.Linear(decoder_dim, attn_dim, bias=False)
        self.v = nn.Linear(attn_dim, 1, bias=False)

    def forward(self, encoder_outputs, decoder_hidden):
        """
        encoder_outputs: (seq_len, encoder_dim)
        decoder_hidden: (decoder_dim,)
        Returns: context_vector (encoder_dim,), attention_weights (seq_len,)
        """
        # encoder_outputs: (seq_len, encoder_dim)
        # decoder_hidden: (decoder_dim,) -> expand to (seq_len, decoder_dim) for broadcasting
        seq_len = encoder_outputs.shape[0]

        # score = v^T tanh(W1 @ encoder_outputs + W2 @ decoder_hidden)
        energy_enc = self.W1(encoder_outputs)                          # (seq_len, attn_dim)
        energy_dec = self.W2(decoder_hidden).unsqueeze(0).expand(seq_len, -1)  # (seq_len, attn_dim)
        scores = self.v(torch.tanh(energy_enc + energy_dec)).squeeze(-1)       # (seq_len,)

        # attention_weights = softmax(scores)
        attention_weights = F.softmax(scores, dim=0)                   # (seq_len,)

        # context = sum(attention_weights * encoder_outputs)
        context_vector = (attention_weights.unsqueeze(-1) * encoder_outputs).sum(dim=0)  # (encoder_dim,)

        return context_vector, attention_weights
