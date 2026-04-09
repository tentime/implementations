# 05 — Seq2Seq with Bahdanau Attention

## Context

**Seq2seq (Sutskever et al., 2014)** introduced the encoder-decoder architecture for sequence transduction: a recurrent encoder compresses an input sequence into a fixed-size vector, and a recurrent decoder generates the output one token at a time. It worked surprisingly well for machine translation, but had a fundamental limitation — everything the encoder knows must be squeezed into a single vector regardless of input length. This is the **bottleneck problem**: as sequences grow longer, the fixed-size context vector becomes an increasingly lossy summary.

**Bahdanau et al. (2015)** — "Neural Machine Translation by Jointly Learning to Align and Translate" — solved this by letting the decoder dynamically attend to *all* encoder hidden states at each decoding step. Instead of one context vector, the decoder computes a weighted average over the full encoder sequence, where the weights reflect how relevant each input position is to the current output position. These weights are the attention mechanism, and they can be visualized as an alignment matrix between input and output.

The alignment (attention) score used here is additive:

```
score(h_t, s_{t-1}) = v^T * tanh(W1 * h_t + W2 * s_{t-1})
```

where `h_t` is an encoder hidden state, `s_{t-1}` is the previous decoder hidden state, and `W1`, `W2`, `v` are learned parameters.

## What this code does

The task is **date format conversion**: turning natural-language dates like `"january 3rd 2025"` into ISO format `"2025-01-03"`. It is a useful toy task because:
- The output structure is rigid and predictable (the model should learn a clear alignment)
- The attention heatmap is easy to interpret (year digits should attend to the year in the source)

500 synthetic date pairs are generated inline. 450 are used for training, 50 for testing. Tokenization is character-level (each character is its own token). The model trains for 2000 steps with Adam and 50% teacher forcing.

Files:
- `attention.py` — `BahdanauAttention` module
- `encoder.py` — `LSTMCell` (manual gates) + `BidirectionalEncoder`
- `decoder.py` — `Decoder` with attention, teacher forcing, and greedy inference
- `train.py` — data generation, training loop, inference printout
- `visualize.py` — attention heatmap → `attention_heatmap.png`
- `test_attention.py` — unit tests (pytest)

## Key implementation details

**Alignment score formula.** The score is computed for every encoder position simultaneously:

```python
energy_enc = W1(encoder_outputs)             # (seq_len, attn_dim)
energy_dec = W2(decoder_hidden).unsqueeze(0) # (1,       attn_dim) → broadcast
scores     = v(tanh(energy_enc + energy_dec)).squeeze(-1)  # (seq_len,)
weights    = softmax(scores)                 # (seq_len,) sums to 1
context    = (weights.unsqueeze(-1) * encoder_outputs).sum(0)  # (encoder_dim,)
```

**Teacher forcing.** At each decoder step, the input token is either the ground-truth previous token (teacher forcing, 50% of the time) or the model's own previous prediction. Teacher forcing accelerates early training but too-high a ratio can cause the model to be fragile at inference time when it must feed its own outputs back in.

**Reading the attention heatmap.** The x-axis is output characters, the y-axis is input characters. A bright cell at `(out_i, in_j)` means the decoder paid strong attention to input position `j` when generating output character `i`. For a well-trained model on this task, you should see diagonal-ish bands: the year digits in the output will attend to the year digits in the input, and the month will attend to the month name.

**Bidirectional encoder.** Two `LSTMCell` instances run in opposite directions over the input and their hidden states are concatenated at each position, giving each encoder output access to both left and right context. The final hidden state (used to initialize the decoder) is the concatenation of the last forward and last backward states.

## What's deliberately omitted

- **Beam search.** Greedy decoding is used at inference time. Beam search would keep the top-k partial sequences at each step and typically improves output quality, especially for longer sequences.
- **BPE / subword tokenization.** Character-level tokenization is used for simplicity. Real NMT systems use byte-pair encoding or SentencePiece.
- **Multi-head attention.** Bahdanau attention uses a single attention head. Splitting into multiple heads (each attending to different aspects of the input) is the key innovation in folder 06.
- **Luong attention.** Luong et al. (2015) introduced a simpler multiplicative score `h_t^T W s_{t-1}` and input-feeding; neither is implemented here.
