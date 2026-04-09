# 06 — Transformer (Attention Is All You Need)

## Context

**Vaswani et al. 2017** — "Attention Is All You Need" — eliminated recurrence entirely. Where seq2seq used RNNs to process sequences step by step, the transformer processes all positions in parallel using self-attention. This unlocked the ability to train on much larger datasets (no sequential bottleneck in the forward pass) and to capture long-range dependencies without the vanishing gradient problems that plagued deep RNNs.

Two things replaced recurrence:

**Positional encoding.** Because self-attention is permutation-equivariant (shuffling the input produces shuffled outputs), the model has no built-in sense of order. Positional encodings — fixed sinusoidal signals added to the input embeddings — inject position information. The specific formula (sin/cos at different frequencies) allows the model to attend by relative position, because `PE(pos+k)` can be expressed as a linear function of `PE(pos)`.

**Self-attention.** Each token attends to all other tokens in the sequence (or, for the decoder, all previous tokens) via scaled dot-product attention. This replaces the hidden state as the mechanism for carrying information across the sequence. Multiple attention heads run in parallel, each learning to focus on different aspects of the input.

## What this code does

The task is translating English digit words to French digit words: `"three two seven"` → `"trois deux sept"`. Sequences of 3–6 words are generated randomly. This is a tiny word-level task with a vocabulary of about 20 tokens total, small enough to reach high accuracy in 3000 training steps.

Files:
- `attention.py` — `scaled_dot_product_attention`, `MultiHeadAttention`, `SinusoidalPositionalEncoding` (~80 lines, self-contained)
- `encoder.py` — `FeedForward`, `EncoderLayer`, `TransformerEncoder`
- `decoder.py` — `DecoderLayer`, `TransformerDecoder`
- `transformer.py` — `make_causal_mask`, `Transformer` (full model), greedy decoding
- `train.py` — data generation, training loop (3000 steps), exact-match accuracy evaluation
- `test_transformer.py` — unit tests (pytest)

## Key implementation details

**Additive masking (`-inf` before softmax).** The causal mask is implemented by filling masked positions with `-inf` before the softmax. After `exp(-inf) = 0`, these positions contribute nothing to the weighted sum of values. This is numerically cleaner than zeroing weights post-softmax (which would require renormalization) and is the standard approach in every major framework.

**Why `sqrt(d_k)` scaling.** With random initialization, the dot products `Q @ K^T` grow in variance proportionally to `d_k`. For large `d_k`, this pushes the softmax into regions of near-zero gradient (the distribution becomes very peaked). Dividing by `sqrt(d_k)` keeps the variance of the dot products at roughly 1 regardless of head dimension, maintaining healthy gradient flow.

**POST-LayerNorm vs PRE-LayerNorm.** The original paper uses POST-LN: `LayerNorm(x + Sublayer(x))`. This matches the paper but is sensitive to initialization — without a learning rate warmup schedule the gradients at the bottom layers can be very large early in training. Modern models (GPT-2, GPT-3, LLaMA) use PRE-LN: `x + Sublayer(LayerNorm(x))`. PRE-LN trains stably with a constant learning rate. This codebase uses POST-LN to match the 2017 paper; switching to PRE-LN is a one-line change in `EncoderLayer` and `DecoderLayer`.

**`attention.py` as the self-contained core.** The entire attention mechanism — scaled dot-product attention, multi-head attention with an explicit per-head loop, and sinusoidal positional encoding — lives in `attention.py` in under 80 lines. The per-head loop (`for h in range(num_heads)`) is slower than the batched matrix formulation but makes the computation explicit: each head really does have its own `W_Q`, `W_K`, `W_V` matrices and produces its own output, which are then concatenated and projected.

## What's deliberately omitted

- **Label smoothing.** The paper uses label smoothing (ε = 0.1) to prevent the model from becoming overconfident. Plain `CrossEntropyLoss` is used here for clarity.
- **Warmup schedule.** The paper uses a specific learning rate schedule: warmup for 4000 steps then inverse square root decay. Plain Adam with a fixed learning rate works fine for this toy task.
- **Beam search.** Greedy decoding is used at inference time. Beam search (keeping the top-k partial sequences at each step) improves translation quality significantly on real tasks.
- **BPE / subword tokenization.** Word-level tokens are used here. Real translation systems use byte-pair encoding or SentencePiece to handle rare words and morphology.
