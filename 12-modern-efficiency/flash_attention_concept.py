"""
Flash Attention: The Tiling Trick

The insight: standard attention materializes an (N x N) matrix.
This costs O(N^2) memory. For N=4096, that is 64M floats = 256MB per layer per head.

Flash Attention (Dao et al. 2022) avoids materializing the full matrix by computing
the output in tiles, using the online softmax algorithm to maintain numerical stability.

This file implements the concept in pure PyTorch (no CUDA kernels).
The memory difference is real; the speed difference requires CUDA because the
actual speedup comes from eliminating slow HBM (off-chip memory) reads/writes.

Reference: Dao et al., "FlashAttention: Fast and Memory-Efficient Exact Attention
with IO-Awareness", NeurIPS 2022.
"""

import torch
import torch.nn.functional as F
import time


def naive_attention(Q, K, V):
    """
    Standard attention. Materializes the full (N x N) score matrix.
    Memory: O(N^2) — the full score matrix lives in memory during the forward pass.

    Args:
        Q, K, V: (batch, heads, seq_len, head_dim)

    Returns:
        output: same shape as Q
        peak_memory_bytes: estimated bytes for the score matrix
    """
    N = Q.shape[-2]
    scale = Q.shape[-1] ** -0.5

    # This line allocates (batch, heads, N, N) — the expensive part
    scores = torch.matmul(Q, K.transpose(-2, -1)) * scale  # (B, H, N, N)
    weights = F.softmax(scores, dim=-1)                    # (B, H, N, N)
    output = torch.matmul(weights, V)                      # (B, H, N, head_dim)

    # Peak memory: the score matrix + weights (both N x N, float32 = 4 bytes)
    B, H = Q.shape[0], Q.shape[1]
    peak_memory = B * H * N * N * 4 * 2  # scores + weights
    return output, peak_memory


def tiled_attention(Q, K, V, tile_size=32):
    """
    Tiled attention using the online softmax trick.
    Memory: O(N * tile_size) — never materializes the full N x N matrix.

    The online softmax algorithm (Milakov & Gimelshein 2018):
    For each output row q_i, process K/V in tiles of size T:
      1. Compute raw scores for this tile: s = q_i @ K_tile^T
      2. Update running max:   new_max = max(running_max, max(s))
      3. Rescale accumulated output: out *= exp(running_max - new_max)
                                     denom *= exp(running_max - new_max)
      4. Accumulate this tile:  out += exp(s - new_max) @ V_tile
                                denom += sum(exp(s - new_max))
      5. Update running_max = new_max
    Final:  output = out / denom

    This is numerically equivalent to standard softmax attention.

    Args:
        Q, K, V: (batch, heads, seq_len, head_dim)
        tile_size: number of key/value positions processed at once

    Returns:
        output: same shape as Q
        peak_memory_bytes: estimated bytes (proportional to N * tile_size, not N^2)
    """
    B, H, N, d = Q.shape
    scale = d ** -0.5
    output = torch.zeros_like(Q)

    for b in range(B):
        for h in range(H):
            q = Q[b, h]  # (N, d)
            k = K[b, h]  # (N, d)
            v = V[b, h]  # (N, d)

            # Accumulators for each query position
            out_acc = torch.zeros(N, d, dtype=Q.dtype, device=Q.device)
            denom = torch.zeros(N, 1, dtype=Q.dtype, device=Q.device)
            running_max = torch.full((N, 1), float("-inf"), dtype=Q.dtype, device=Q.device)

            # Iterate over tiles of key/value
            for tile_start in range(0, N, tile_size):
                tile_end = min(tile_start + tile_size, N)
                k_tile = k[tile_start:tile_end]  # (T, d)
                v_tile = v[tile_start:tile_end]  # (T, d)

                # Scores for this tile: (N, T)
                scores = (q @ k_tile.transpose(0, 1)) * scale

                # Online softmax update
                tile_max = scores.max(dim=-1, keepdim=True).values  # (N, 1)
                new_max = torch.maximum(running_max, tile_max)       # (N, 1)

                # Rescale previous accumulation to new max
                rescale = torch.exp(running_max - new_max)  # (N, 1)
                out_acc = out_acc * rescale
                denom = denom * rescale

                # Add this tile's contribution
                exp_scores = torch.exp(scores - new_max)   # (N, T)
                out_acc = out_acc + exp_scores @ v_tile    # (N, d)
                denom = denom + exp_scores.sum(dim=-1, keepdim=True)  # (N, 1)

                running_max = new_max

            output[b, h] = out_acc / denom

    # Peak memory: we only hold one tile of scores + accumulators in memory
    # Dominant term: tile scores (N x tile_size) rather than N x N
    peak_memory = B * H * (N * tile_size * 4 + N * d * 4 * 3)  # scores tile + 3 accumulators
    return output, peak_memory


def benchmark_memory(seq_lengths=None):
    """
    For each sequence length, compare peak memory estimates of naive vs tiled attention.
    Print a table:

        seq_len | naive_MB | tiled_MB | ratio
        --------|----------|----------|------

    Note: these are estimates based on the dominant allocation (the score matrix for
    naive, the tile buffer for tiled). Actual PyTorch memory includes activation
    storage for backprop; these numbers show the forward-pass bottleneck.
    """
    if seq_lengths is None:
        seq_lengths = [64, 128, 256, 512]

    batch, heads, head_dim, tile_size = 1, 1, 32, 32

    print(f"{'seq_len':>8} | {'naive_MB':>9} | {'tiled_MB':>9} | {'ratio':>7}")
    print("-" * 45)

    for N in seq_lengths:
        Q = torch.randn(batch, heads, N, head_dim)
        K = torch.randn(batch, heads, N, head_dim)
        V = torch.randn(batch, heads, N, head_dim)

        _, naive_bytes = naive_attention(Q, K, V)
        _, tiled_bytes = tiled_attention(Q, K, V, tile_size=tile_size)

        naive_mb = naive_bytes / 1e6
        tiled_mb = tiled_bytes / 1e6
        ratio = naive_bytes / tiled_bytes

        print(f"{N:>8} | {naive_mb:>9.3f} | {tiled_mb:>9.3f} | {ratio:>7.2f}x")


def verify_equivalence(seq_len=64):
    """
    Verify that tiled_attention produces the same output as naive_attention.
    Returns max absolute difference.
    """
    torch.manual_seed(0)
    B, H, N, d = 2, 2, seq_len, 16
    Q = torch.randn(B, H, N, d)
    K = torch.randn(B, H, N, d)
    V = torch.randn(B, H, N, d)

    naive_out, _ = naive_attention(Q, K, V)
    tiled_out, _ = tiled_attention(Q, K, V, tile_size=16)

    max_diff = (naive_out - tiled_out).abs().max().item()
    return max_diff


if __name__ == "__main__":
    print("Verifying tiled == naive:", verify_equivalence())
    print()
    benchmark_memory()
