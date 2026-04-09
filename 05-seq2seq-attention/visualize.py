"""
Attention heatmap visualization.

Usage:
    python visualize.py

Loads the checkpoint saved by train.py (or runs training inline if absent),
picks one test example, and saves an attention heatmap to attention_heatmap.png.
"""

import os
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from encoder import BidirectionalEncoder
from decoder import Decoder
from train import (
    Seq2SeqAttention, build_dataset, build_vocab,
    encode, decode_tokens, SOS, EOS,
)


def load_or_train():
    checkpoint_path = "seq2seq_checkpoint.pt"
    if os.path.exists(checkpoint_path):
        print(f"Loading checkpoint from {checkpoint_path}")
        ckpt = torch.load(checkpoint_path, map_location="cpu")
    else:
        print("No checkpoint found — running training first.")
        from train import train
        train()
        ckpt = torch.load(checkpoint_path, map_location="cpu")

    model = Seq2SeqAttention(
        vocab_size=ckpt["vocab_size"],
        embed_dim=ckpt["embed_dim"],
        hidden_size=ckpt["hidden_size"],
        attn_dim=ckpt["attn_dim"],
    )
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, ckpt["char2idx"], ckpt["idx2char"], ckpt["test_pairs"]


def visualize():
    model, char2idx, idx2char, test_pairs = load_or_train()

    sos_idx = char2idx[SOS]
    eos_idx = char2idx[EOS]

    # Pick first test example
    src_str, tgt_str = test_pairs[0]
    src_tensor = torch.tensor(encode(src_str, char2idx), dtype=torch.long)

    with torch.no_grad():
        src_embedded = model.embedding(src_tensor)
        encoder_outputs, final_h = model.encoder(src_embedded)
        dec_h = torch.tanh(model.enc2dec_h(final_h))
        dec_c = torch.zeros_like(dec_h)
        pred_tokens, attn_weights = model.decoder.greedy_decode(
            encoder_outputs, dec_h, dec_c,
            sos_token=sos_idx, eos_token=eos_idx, max_len=30,
        )

    predicted = decode_tokens(pred_tokens, idx2char)
    print(f"Source:    {src_str}")
    print(f"Predicted: {predicted}")
    print(f"Target:    {tgt_str}")

    # attn_weights: (tgt_steps, src_len)
    # src characters: the actual chars in src_str (skip SOS/EOS for display)
    src_chars = list(src_str)
    # tgt characters: characters produced by the decoder
    tgt_chars = list(predicted)

    attn_np = attn_weights[:len(tgt_chars), 1:len(src_chars) + 1].numpy()
    # attn_weights columns: 0=SOS, 1..len(src_str)=chars, last=EOS
    # Trim to match displayed characters
    attn_display = attn_weights[:len(tgt_chars)].numpy()

    # Use source characters including SOS/EOS padding positions
    # More intuitive: just use the raw source string chars as y-labels
    # and take columns 1..len(src_str)+1 of attn
    src_labels = src_chars  # one per position in src_str

    fig_h = max(4, len(src_labels) * 0.35)
    fig_w = max(4, len(tgt_chars) * 0.4)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    im = ax.imshow(attn_np.T, aspect="auto", cmap="viridis", vmin=0, vmax=1)
    ax.set_xticks(range(len(tgt_chars)))
    ax.set_xticklabels(tgt_chars, fontsize=9)
    ax.set_yticks(range(len(src_labels)))
    ax.set_yticklabels(src_labels, fontsize=9)
    ax.set_xlabel("Output characters")
    ax.set_ylabel("Input characters")
    ax.set_title(f'Attention: "{src_str}" → "{predicted}"')
    plt.colorbar(im, ax=ax, shrink=0.8)
    plt.tight_layout()
    plt.savefig("attention_heatmap.png", dpi=150)
    print("Saved attention_heatmap.png")


if __name__ == "__main__":
    visualize()
