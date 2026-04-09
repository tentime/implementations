import random
import torch
import torch.nn as nn

from attention import BahdanauAttention
from encoder import LSTMCell


class Decoder(nn.Module):
    """
    Autoregressive decoder with Bahdanau attention.

    At each step:
      1. Embed the previous output token
      2. Concatenate with context vector from attention
      3. Run one LSTMCell step
      4. Project to vocabulary

    Teacher forcing: at training time, feed the true target token as input
    instead of the model's own prediction (with 50% probability).
    """

    def __init__(self, vocab_size, embed_dim, encoder_dim, hidden_size, attn_dim=32):
        super().__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.attention = BahdanauAttention(encoder_dim, hidden_size, attn_dim)
        # Input to cell: embedded token + context vector
        self.cell = LSTMCell(embed_dim + encoder_dim, hidden_size)
        self.output_proj = nn.Linear(hidden_size, vocab_size)

    def forward_step(self, prev_token, h, c, encoder_outputs):
        """
        Single decoder step.

        Args:
            prev_token: (,) integer token index
            h: (hidden_size,) current hidden state
            c: (hidden_size,) current cell state
            encoder_outputs: (src_len, encoder_dim)

        Returns:
            logits: (vocab_size,)
            h_new:  (hidden_size,)
            c_new:  (hidden_size,)
            attn_weights: (src_len,)
        """
        embedded = self.embedding(prev_token)                    # (embed_dim,)
        context, attn_weights = self.attention(encoder_outputs, h)  # (encoder_dim,), (src_len,)
        cell_input = torch.cat([embedded, context], dim=-1)     # (embed_dim + encoder_dim,)
        h_new, c_new = self.cell(cell_input, h, c)
        logits = self.output_proj(h_new)                         # (vocab_size,)
        return logits, h_new, c_new, attn_weights

    def forward(self, encoder_outputs, initial_h, initial_c,
                sos_token, target_seq=None, teacher_forcing_ratio=0.5):
        """
        Decode a full sequence.

        Args:
            encoder_outputs:    (src_len, encoder_dim)
            initial_h:          (hidden_size,) — projected from encoder final state
            initial_c:          (hidden_size,) — zeros or projected
            sos_token:          integer, the <SOS> token id
            target_seq:         (tgt_len,) integer tensor, used for teacher forcing (training only)
            teacher_forcing_ratio: probability of using ground-truth input at each step

        Returns:
            all_logits:    (tgt_len, vocab_size)
            all_attn:      (tgt_len, src_len)
        """
        device = encoder_outputs.device
        tgt_len = target_seq.shape[0] if target_seq is not None else 20

        h, c = initial_h, initial_c
        prev_token = torch.tensor(sos_token, device=device)

        all_logits = []
        all_attn = []

        for t in range(tgt_len):
            logits, h, c, attn_weights = self.forward_step(prev_token, h, c, encoder_outputs)
            all_logits.append(logits)
            all_attn.append(attn_weights)

            # Decide next input token
            if target_seq is not None and random.random() < teacher_forcing_ratio:
                prev_token = target_seq[t]   # teacher forcing: use ground truth
            else:
                prev_token = logits.argmax(dim=-1).detach()  # use own prediction

        all_logits = torch.stack(all_logits, dim=0)   # (tgt_len, vocab_size)
        all_attn = torch.stack(all_attn, dim=0)       # (tgt_len, src_len)
        return all_logits, all_attn

    def greedy_decode(self, encoder_outputs, initial_h, initial_c,
                      sos_token, eos_token, max_len=30):
        """
        Greedy inference (no teacher forcing). Stops at EOS or max_len.

        Returns:
            predicted_tokens: list of integer token ids
            all_attn:         (steps, src_len)
        """
        device = encoder_outputs.device
        h, c = initial_h, initial_c
        prev_token = torch.tensor(sos_token, device=device)

        predicted_tokens = []
        all_attn = []

        for _ in range(max_len):
            logits, h, c, attn_weights = self.forward_step(prev_token, h, c, encoder_outputs)
            all_attn.append(attn_weights)
            prev_token = logits.argmax(dim=-1)
            predicted_tokens.append(prev_token.item())
            if prev_token.item() == eos_token:
                break

        all_attn = torch.stack(all_attn, dim=0)  # (steps, src_len)
        return predicted_tokens, all_attn
