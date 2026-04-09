# 10 — LoRA: Low-Rank Adaptation

## Context

Hu et al. 2022 ("LoRA: Low-Rank Adaptation of Large Language Models") identified a striking property of fine-tuning: the weight updates that actually matter have low intrinsic rank. When you fine-tune a large pretrained model, the change in each weight matrix ΔW can be well-approximated by a product of two small matrices — ΔW ≈ B · A, where A has shape (d_in × r) and B has shape (r × d_out), with r ≪ min(d_in, d_out).

The practical consequence: instead of updating all W's parameters during fine-tuning, freeze W entirely and train only the low-rank matrices A and B. The forward pass becomes:

```
output = W·x + (α/r) · B·A·x
```

For a model with d = 4096, the Q projection matrix has 16M parameters. With rank r = 4, A and B together have 4096 × 4 + 4 × 4096 = 32,768 parameters — a 500× reduction. Across a full transformer with adapters on Q and V, typical trainable parameter ratios fall below 1% of the full model.

This matters practically for three reasons: you can fine-tune on a single GPU that couldn't hold the optimizer states for full fine-tuning; you can maintain one copy of the base model and swap tiny adapter files for different tasks; and the adapter file size (kilobytes, not gigabytes) makes deployment and versioning tractable.

## What this code does

`train.py` runs three sequential experiments on a small GPT:

1. **Base training**: trains char-level GPT on Shakespeare for 500 steps, saves `base_model.pt`.

2. **Experiment A (full fine-tune)**: loads the base model, unfreezes all parameters, fine-tunes on legal-style text for 300 steps. This is the baseline: every parameter participates in gradient updates.

3. **Experiment B (LoRA fine-tune)**: loads the same base model, injects LoRA into Q and V projections via `inject_lora`, then fine-tunes only the A and B matrices for 300 steps.

After training, the script prints:

- The trainable parameter ratio for the LoRA model (typically 1–3% of total parameters for this small architecture)
- The size of `lora_adapter.pt` vs `full_ft_model.pt`, showing the file size advantage directly
- Five generated continuations from each model given the same legal-domain prompt

The generation comparison shows whether LoRA successfully shifts the model's output distribution toward the target domain while using a fraction of the compute.

## Key implementation details

**B=zeros initialization.** The LoRA adapter is initialized with B as an all-zeros matrix and A sampled from N(0, 0.02). Because the adapter output is `scaling * B·A·x`, zeroing B means the adapter contributes exactly zero at the start of fine-tuning — the LoRA model is numerically identical to the base model on step 0. This is load-bearing: if B were random, fine-tuning would immediately diverge from the base model before any gradient signal has been received. The zero-init ensures the first steps of fine-tuning are stable and the base model's representations are preserved as a starting point.

**The scaling factor alpha/rank.** The adapter output is multiplied by `alpha/rank` before being added to the base output. This decouples the effective learning rate of the LoRA adapter from the rank hyperparameter. If you increase rank from 4 to 16, the scaling factor compensates, keeping the adapter's contribution to the output in the same range. In practice, alpha is often set to the same value as rank (giving scaling = 1.0) or to 2×rank. The paper found results were not particularly sensitive to alpha.

**Why Q and V, not K?** The paper's experiments found that adapting Q and V alone achieves most of the fine-tuning benefit, while adapting K adds little. The intuition: K and Q together form the attention pattern, and adapting Q already changes the dot-product scores; adapting V changes what information is actually retrieved. K adapting is somewhat redundant with Q. This is an empirical finding rather than a theoretical guarantee, and some practitioners adapt all three or four projection matrices.

**inject_lora mechanics.** The function walks `model.named_modules()`, finds `nn.Linear` layers whose dotted name ends with any of the target strings, then replaces them in-place using `setattr` on the parent module. The base weight is copied into the frozen `W` parameter of `LoRALinear`. After injection, a call to `model.parameters()` returns both the frozen base parameters and the trainable A/B matrices — the optimizer receives only the trainable ones by filtering `requires_grad`.

## What's deliberately omitted

**QLoRA (4-bit quantization).** Dettmers et al. 2023 extended LoRA by quantizing the frozen base weights to 4-bit NormalFloat format, reducing the memory footprint of the frozen W matrices by another 8×. This enables fine-tuning 65B-parameter models on a single A100. Implementing QLoRA requires custom CUDA kernels (bitsandbytes library) and is outside scope here.

**Merging the adapter back into weights for inference.** Because the LoRA update is linear, you can compute W_merged = W + (alpha/rank) * B·A once after fine-tuning and discard the separate A and B matrices. The merged model is identical to the base + adapter model at inference time, but requires no special LoRA infrastructure — it's just a standard `nn.Linear`. This is how LoRA models are typically deployed. The merge operation is omitted here to keep the adapter concept explicit.

**DoRA (Weight-Decomposed Low-Rank Adaptation, Liu et al. 2024).** DoRA decomposes the pretrained weight into magnitude and direction components, then applies LoRA only to the directional component. This tends to improve fine-tuning quality, especially for smaller ranks, at the cost of a slightly more complex implementation. Omitted here in favor of keeping the original LoRA formulation clean.
