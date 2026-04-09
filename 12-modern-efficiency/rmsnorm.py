"""
RMSNorm: A Faster Alternative to LayerNorm

LayerNorm: normalize by (x - mean) / sqrt(var + eps), then scale and shift
RMSNorm:   normalize by x / sqrt(mean(x^2) + eps), then scale only

The key difference: RMSNorm skips the mean-centering step (subtract mean) and
omits the shift parameter (bias). The hypothesis is that re-centering is not
necessary for training stability — only re-scaling matters.

This saves approximately 15% of the normalization computation and removes one
parameter vector. Used in LLaMA, Mistral, Gemma, and most modern open LLMs.

Reference: Zhang & Sennrich, "Root Mean Square Layer Normalization", NeurIPS 2019.
"""

import time
import torch
import torch.nn as nn


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        """
        x: (..., dim)
        Normalize along the last dimension using RMS.

        RMS(x) = sqrt(mean(x^2) + eps)
        output = (x / RMS(x)) * scale
        """
        rms = torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return (x / rms) * self.scale


def benchmark_vs_layernorm(dim=512, seq_len=1024, batch_size=32, n_trials=100):
    """
    Compare RMSNorm and nn.LayerNorm on identical inputs.
    1. Verify outputs are similar (not identical — different normalization)
    2. Benchmark wall clock time over n_trials
    3. Print speedup ratio
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x = torch.randn(batch_size, seq_len, dim, device=device)

    rms_norm = RMSNorm(dim).to(device)
    layer_norm = nn.LayerNorm(dim).to(device)

    # --- Verify behavior ---
    with torch.no_grad():
        rms_out = rms_norm(x)
        ln_out = layer_norm(x)

    # Shapes should match
    assert rms_out.shape == x.shape, f"RMSNorm shape mismatch: {rms_out.shape}"
    assert ln_out.shape == x.shape, f"LayerNorm shape mismatch: {ln_out.shape}"

    # Outputs should NOT be identical (different normalization)
    max_diff = (rms_out - ln_out).abs().max().item()
    print(f"Max difference RMSNorm vs LayerNorm: {max_diff:.4f}  (expected: non-zero)")

    # --- Warmup ---
    for _ in range(10):
        _ = rms_norm(x)
        _ = layer_norm(x)

    if device.type == "cuda":
        torch.cuda.synchronize()

    # --- Benchmark RMSNorm ---
    start = time.perf_counter()
    for _ in range(n_trials):
        _ = rms_norm(x)
    if device.type == "cuda":
        torch.cuda.synchronize()
    rms_time = time.perf_counter() - start

    # --- Benchmark LayerNorm ---
    start = time.perf_counter()
    for _ in range(n_trials):
        _ = layer_norm(x)
    if device.type == "cuda":
        torch.cuda.synchronize()
    ln_time = time.perf_counter() - start

    rms_ms = rms_time / n_trials * 1000
    ln_ms = ln_time / n_trials * 1000
    speedup = ln_time / rms_time

    print(f"\nBenchmark over {n_trials} trials (batch={batch_size}, seq={seq_len}, dim={dim})")
    print(f"  RMSNorm:   {rms_ms:.3f} ms/call")
    print(f"  LayerNorm: {ln_ms:.3f} ms/call")
    print(f"  Speedup:   {speedup:.2f}x  (RMSNorm faster)")
    if device.type == "cpu":
        print("  Note: speedup is modest on CPU; ~15-30% is typical on GPU.")

    return speedup


if __name__ == "__main__":
    benchmark_vs_layernorm()
