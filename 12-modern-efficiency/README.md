# 12 — Modern Efficiency: RoPE, Flash Attention, RMSNorm, SwiGLU

## Context

Between 2019 and 2022, four engineering improvements quietly became universal in production LLMs. None of them changed the fundamental transformer architecture — they made it faster, cheaper, and more capable of handling long sequences:

- **RoPE** (Su et al. 2021) — Rotary Position Embedding. Replaces absolute learned position embeddings with a rotation applied to query and key vectors, encoding position directly into the attention dot product. Enables better length generalization and makes the attention score a function of *relative* position.
- **Flash Attention** (Dao et al. 2022) — A memory-efficient exact attention algorithm. Uses tiling and the online softmax trick to avoid materializing the full N×N attention matrix, reducing memory from O(N²) to O(N). Now the standard attention implementation in PyTorch and all major frameworks.
- **RMSNorm** (Zhang & Sennrich 2019) — A simplified layer normalization that skips mean-centering. Saves ~15% of normalization compute with negligible quality loss. Used in LLaMA, Mistral, Gemma.
- **SwiGLU** (Shazeer 2020) — A gated activation function for the FFN layer. Replaces ReLU with a smooth gate that selectively routes information. Used in PaLM, LLaMA, Mistral, and virtually every modern open LLM.

These are not paradigm shifts. They are engineering improvements that compound: together they meaningfully reduce compute per token while improving model quality at the same parameter count.

---

## What this code does

### `rope.py`
Implements `precompute_freqs_cis` (the complex rotation table) and `apply_rotary_emb` (multiplying query/key vectors by their position-dependent rotation). The `demonstrate_relative_position_property` function confirms the key mathematical property: dot products between rotated queries and keys depend only on the *difference* in position, not the absolute positions.

### `flash_attention_concept.py`
Two implementations of scaled dot-product attention:
- `naive_attention`: standard, materializes the full N×N score matrix
- `tiled_attention`: uses the online softmax algorithm to process keys/values in tiles, never holding more than a tile's worth of scores in memory

`benchmark_memory` prints a table comparing peak memory estimates across sequence lengths, showing O(N²) vs O(N) scaling. `verify_equivalence` confirms that both implementations produce identical outputs.

### `rmsnorm.py`
Implements `RMSNorm` and benchmarks it against `nn.LayerNorm` on identical inputs. Verifies that outputs differ (they normalize differently), then measures wall-clock time over 100 trials.

### `swiglu.py`
Implements `SwiGLU` and `ReLUFFN` side-by-side. The `compare_param_counts` function shows that SwiGLU with `d_ff = d_model * 4 * 2/3` has approximately the same parameter count as a standard 4× ReLU FFN — verifying the 2/3 scaling rule.

### `benchmark.py`
Runs all four demonstrations in sequence and prints a unified report.

---

## Key implementation details

### RoPE: why dot products encode relative position

For a single dimension pair, RoPE multiplies the query at position i by e^{i·m·θ} and the key at position j by e^{i·n·θ}. The attention dot product then contains:

```
Re(q̃ · k̃*) = Re(q · k* · e^{i(m-n)θ})
```

The factor `e^{i(m-n)θ}` depends only on `m-n` (the relative position), not on the absolute values. This means the attention score automatically captures relative distance without any architectural modification to the attention mechanism.

### Online softmax: the tiling trick

Standard softmax requires two passes over the data: one to find the max (for numerical stability) and one to compute the exponentials. The online softmax algorithm maintains running estimates of both the max and the denominator, updating them tile by tile with a rescaling correction:

```
When new_max > old_max:
    accumulated_output *= exp(old_max - new_max)   # rescale old contributions
    accumulate new tile's contribution
```

At the end, divide by the accumulated denominator. The output is bit-for-bit identical to standard softmax attention.

### RMSNorm: skipping mean-centering

LayerNorm computes `(x - mean(x)) / sqrt(var(x) + eps)`. RMSNorm computes `x / sqrt(mean(x²) + eps)`. The subtraction of the mean is the expensive step. The hypothesis (confirmed empirically) is that the centering is not necessary for stable training — only the scale normalization matters. RMSNorm also removes the shift (bias) parameter, giving a small additional saving.

### SwiGLU: the 2/3 parameter scaling rule

A standard ReLU FFN with 4× expansion has `2 × d_model × (4 × d_model)` parameters (two matrices). SwiGLU has three matrices: W1, W2, W3. To match the parameter count:

```
3 × d_model × d_ff_swi = 2 × d_model × (4 × d_model)
d_ff_swi = 8/3 × d_model = 4 × d_model × 2/3
```

So `d_ff = int(d_model * 4 * 2/3)` gives approximately the same total parameter count with three matrices instead of two.

---

## What's deliberately omitted

**Actual CUDA Flash Attention kernel.** The memory savings demonstrated here are real, but the speed improvement requires a CUDA kernel that fuses the tiling loop and avoids slow HBM (high-bandwidth memory) reads/writes. The PyTorch implementation shown here is algorithmically correct but does not provide the 2–4× speedup of the real CUDA implementation. Use `torch.nn.functional.scaled_dot_product_attention` or the `flash-attn` package for that.

**GQA/MQA (Grouped/Multi-Query Attention).** Introduced in the original MQA paper (Shazeer 2019) and popularized by LLaMA 2 and Mistral. Instead of one set of K/V heads per Q head, multiple query heads share a single K/V head. Dramatically reduces the KV cache size during inference. Would fit naturally as a `grouped_query_attention.py` in this folder.

**Sliding window attention (Mistral 7B).** Rather than attending to all previous tokens, each position attends only to the last W tokens. Reduces attention memory from O(N²) to O(N·W) and improves throughput for long sequences. Implemented efficiently using Flash Attention's block-sparse variant.

**ALiBi (Press et al. 2022).** An alternative to RoPE for position encoding: instead of rotating Q/K, add a fixed linear bias to attention scores based on relative distance. Simpler to implement than RoPE and also extrapolates to longer sequences, but RoPE has generally won in practice.
