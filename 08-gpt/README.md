# 08 — GPT

## Context

Radford et al. 2018 (GPT-1) and 2019 (GPT-2) made a single architectural bet that has defined the field: train a large transformer language model — one that simply predicts the next token — at scale, and useful capabilities will emerge.

The causal mask is the architectural choice that defines "decoder-only". By forcing each position to attend only to positions at or before itself, the model can only use past context to predict the next token. This makes training efficient — every position in a sequence provides a training signal in a single forward pass — and generation natural: at inference time, you feed in a prompt and let the model extend it one token at a time.

GPT-2 (1.5B parameters, trained on 40GB of web text) demonstrated that scale alone, with no task-specific fine-tuning, could generate coherent long-form text and achieve reasonable zero-shot performance on language tasks. The lesson that sparked GPT-3 and everything after: scaling the same simple recipe works.

## What this code does

- `gpt.py` — implements `CausalSelfAttention` (with a causal mask registered as a buffer), `GPTBlock` (PRE-LN), and `GPT` with weight-tied input/output embeddings and an autoregressive `generate` method supporting temperature and top-k filtering.
- `train.py` — char-level language model trained on a Shakespeare excerpt. Runs for 2000 steps, prints loss every 200 steps and a 100-char sample every 500 steps, then saves a checkpoint.
- `generate.py` — loads the checkpoint (or trains a quick 500-step model if none exists) and prints generations at three temperatures: greedy (deterministic argmax), 0.8 (conservative), and 1.4 (high entropy).
- `test_gpt.py` — six pytest tests covering the causal mask shape, output shape, generation length, greedy determinism, loss decrease, and weight tying.

## Key implementation details

**The causal mask as a buffer.** The lower-triangular mask is created once in `CausalSelfAttention.__init__` via `self.register_buffer('mask', torch.tril(torch.ones(max_len, max_len)))`. Registering as a buffer means: it is not a learnable parameter, it moves to the correct device automatically when you call `model.to(device)`, and it is saved in `state_dict` so checkpoints are self-contained. At each forward pass the mask is simply sliced to the current sequence length — no recomputation.

**PRE-LN as the GPT-2 deviation.** The original 2017 transformer applied LayerNorm after the residual addition (POST-LN). GPT-2 moved it before the sub-layer (PRE-LN): `x = x + sublayer(norm(x))`. With POST-LN, the residual stream's scale grows with depth because the norm is applied across the skip connection, not inside it. PRE-LN keeps the residual stream stable regardless of depth, enabling training of deeper models without careful learning-rate warmup schedules. This is now the default in every major architecture.

**Learned vs. sinusoidal positional embeddings.** The original Vaswani et al. paper used fixed sinusoidal PE. GPT-2 replaced this with a learned `nn.Embedding` table of shape `(max_len, d_model)`. In practice the two approaches perform comparably on standard benchmarks, but learned PE is simpler to implement and now more common. The comment in `GPT.__init__` notes this deviation explicitly.

**Temperature and top-k filtering.** Temperature scales logits before the softmax: low temperature sharpens the distribution (more greedy), high temperature flattens it (more uniform). `temperature=0` bypasses softmax entirely and uses argmax — deterministic, reproducible. `top_k` zeroes out all logits below the k-th largest before sampling, preventing the model from ever choosing very low-probability tokens; this is a cheap but effective way to avoid degenerate outputs at high temperatures.

**Weight tying.** `lm_head.weight` is set to the same `nn.Parameter` object as `token_embedding.weight`. An `assert … is …` in `__init__` verifies this. Same motivation as in folder 07: the embedding space is shared between input lookup and output scoring.

## What's deliberately omitted

**BPE tokenizer.** Real GPT models use byte-pair encoding (tiktoken for GPT-4). Char-level tokenization keeps the vocabulary tiny and lets us focus on the architecture.

**KV cache.** At inference time, re-running the full forward pass for every new token is wasteful — the K and V tensors for all previous positions are recomputed identically each time. A KV cache stores them across steps. It is the most natural next implementation step after understanding the core model, and a satisfying exercise: the generate loop stays the same, but each forward pass only processes the latest token.

**Flash Attention.** Flash Attention (Dao et al. 2022) rewrites scaled dot-product attention to be IO-aware: it avoids materialising the full `(seq_len, seq_len)` attention matrix in GPU memory by fusing the operations into a single CUDA kernel. It is algorithmically equivalent to standard attention but ~2–4× faster and dramatically more memory-efficient for long sequences. It is an engineering optimisation, not an architectural change.
