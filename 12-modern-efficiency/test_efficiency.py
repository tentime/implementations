"""
Tests for the modern efficiency implementations.
Run: python -m pytest test_efficiency.py -v
"""

import torch
import torch.nn as nn
import pytest
from rope import precompute_freqs_cis, apply_rotary_emb
from flash_attention_concept import naive_attention, tiled_attention
from rmsnorm import RMSNorm
from swiglu import SwiGLU, ReLUFFN


# ---------------------------------------------------------------------------
# RoPE tests
# ---------------------------------------------------------------------------

def test_rope_relative_position_property():
    """
    dot(RoPE(q, i), RoPE(k, j)) should depend only on (i-j).
    Test 3 pairs with delta=3: (5,2), (10,7), (20,17).
    All dot products should be equal within 1e-4.
    """
    torch.manual_seed(42)
    head_dim = 16
    n_heads = 1
    max_pos = 32

    freqs_cis = precompute_freqs_cis(head_dim, max_pos)
    q_vec = torch.randn(1, n_heads, head_dim)
    k_vec = torch.randn(1, n_heads, head_dim)

    test_pairs = [(5, 2), (10, 7), (20, 17)]
    dot_products = []

    for pos_i, pos_j in test_pairs:
        q_rot, _ = apply_rotary_emb(q_vec, k_vec, freqs_cis[pos_i : pos_i + 1])
        _, k_rot = apply_rotary_emb(q_vec, k_vec, freqs_cis[pos_j : pos_j + 1])
        dot = (q_rot[0, 0] * k_rot[0, 0]).sum().item()
        dot_products.append(dot)

    # All three pairs have the same delta=3, so dot products should match
    for dp in dot_products[1:]:
        diff = abs(dp - dot_products[0])
        assert diff < 1e-4, (
            f"RoPE relative position property violated: "
            f"dot products {dot_products} differ by {diff}"
        )


def test_rope_different_deltas_give_different_dots():
    """
    Pairs with different relative distances should give different dot products.
    (Sanity check that the property is non-trivial.)
    """
    torch.manual_seed(7)
    head_dim = 16
    freqs_cis = precompute_freqs_cis(head_dim, 32)
    q = torch.randn(1, 1, head_dim)
    k = torch.randn(1, 1, head_dim)

    # delta=1
    q1, _ = apply_rotary_emb(q, k, freqs_cis[5:6])
    _, k1 = apply_rotary_emb(q, k, freqs_cis[4:5])
    dot1 = (q1[0, 0] * k1[0, 0]).sum().item()

    # delta=5
    q5, _ = apply_rotary_emb(q, k, freqs_cis[5:6])
    _, k5 = apply_rotary_emb(q, k, freqs_cis[0:1])
    dot5 = (q5[0, 0] * k5[0, 0]).sum().item()

    assert abs(dot1 - dot5) > 1e-3, "Different deltas should give different dot products"


def test_rope_output_shape():
    """apply_rotary_emb preserves shape."""
    head_dim = 16
    seq_len = 8
    n_heads = 4
    freqs_cis = precompute_freqs_cis(head_dim, seq_len)
    xq = torch.randn(seq_len, n_heads, head_dim)
    xk = torch.randn(seq_len, n_heads, head_dim)
    xq_out, xk_out = apply_rotary_emb(xq, xk, freqs_cis)
    assert xq_out.shape == xq.shape
    assert xk_out.shape == xk.shape


# ---------------------------------------------------------------------------
# Flash Attention / tiled attention tests
# ---------------------------------------------------------------------------

def test_tiled_attention_matches_naive():
    """Tiled and naive attention outputs agree to within 1e-5."""
    torch.manual_seed(0)
    B, H, N, d = 2, 2, 64, 16
    Q = torch.randn(B, H, N, d)
    K = torch.randn(B, H, N, d)
    V = torch.randn(B, H, N, d)

    naive_out, _ = naive_attention(Q, K, V)
    tiled_out, _ = tiled_attention(Q, K, V, tile_size=16)

    max_diff = (naive_out - tiled_out).abs().max().item()
    assert max_diff < 1e-5, f"Tiled attention differs from naive by {max_diff}"


def test_tiled_attention_matches_naive_various_tile_sizes():
    """Agreement holds for multiple tile sizes."""
    torch.manual_seed(1)
    B, H, N, d = 1, 1, 64, 8
    Q = torch.randn(B, H, N, d)
    K = torch.randn(B, H, N, d)
    V = torch.randn(B, H, N, d)

    naive_out, _ = naive_attention(Q, K, V)
    for tile_size in [8, 16, 32, 64]:
        tiled_out, _ = tiled_attention(Q, K, V, tile_size=tile_size)
        max_diff = (naive_out - tiled_out).abs().max().item()
        assert max_diff < 1e-5, f"tile_size={tile_size}: diff={max_diff}"


def test_naive_attention_output_shape():
    """naive_attention returns (B, H, N, d) and a positive memory estimate."""
    B, H, N, d = 2, 4, 32, 16
    Q = torch.randn(B, H, N, d)
    K = torch.randn(B, H, N, d)
    V = torch.randn(B, H, N, d)
    out, mem = naive_attention(Q, K, V)
    assert out.shape == (B, H, N, d)
    assert mem > 0


def test_tiled_uses_less_memory_than_naive():
    """Peak memory estimate for tiled should be less than naive for large N."""
    B, H, N, d = 1, 1, 256, 32
    Q = torch.randn(B, H, N, d)
    K = torch.randn(B, H, N, d)
    V = torch.randn(B, H, N, d)
    _, naive_mem = naive_attention(Q, K, V)
    _, tiled_mem = tiled_attention(Q, K, V, tile_size=32)
    assert tiled_mem < naive_mem, (
        f"Tiled memory ({tiled_mem}) should be < naive ({naive_mem})"
    )


# ---------------------------------------------------------------------------
# RMSNorm tests
# ---------------------------------------------------------------------------

def test_rmsnorm_output_shape():
    """RMSNorm output shape matches input shape."""
    norm = RMSNorm(64)
    x = torch.randn(4, 16, 64)
    out = norm(x)
    assert out.shape == x.shape, f"Expected {x.shape}, got {out.shape}"


def test_rmsnorm_normalizes():
    """
    After normalization (before scale), RMS of output should be ~1.0.
    Set scale=1 and check sqrt(mean(out^2)) ≈ 1.0 for each vector.
    """
    dim = 128
    norm = RMSNorm(dim)
    # Reset scale to 1 so we see raw normalization behavior
    with torch.no_grad():
        norm.scale.fill_(1.0)

    x = torch.randn(8, 32, dim) * 5  # large values to test normalization
    out = norm(x)

    # RMS of each output vector should be ≈ 1.0
    rms_per_vec = torch.sqrt(out.pow(2).mean(dim=-1))
    max_deviation = (rms_per_vec - 1.0).abs().max().item()
    assert max_deviation < 1e-5, f"RMS deviation from 1.0: {max_deviation}"


def test_rmsnorm_scale_parameter_applied():
    """Scaling by a constant gamma multiplies output by gamma."""
    dim = 32
    gamma = 3.0
    norm = RMSNorm(dim)
    with torch.no_grad():
        norm.scale.fill_(gamma)

    x = torch.randn(2, 8, dim)
    out = norm(x)

    # Output should be gamma times the normalized input
    norm_unit = RMSNorm(dim)
    with torch.no_grad():
        norm_unit.scale.fill_(1.0)
    out_unit = norm_unit(x)

    max_diff = (out - gamma * out_unit).abs().max().item()
    assert max_diff < 1e-5, f"Scale parameter not applied correctly: diff={max_diff}"


def test_rmsnorm_vs_layernorm_different_outputs():
    """RMSNorm and LayerNorm produce different outputs (different normalization)."""
    dim = 64
    rms = RMSNorm(dim)
    ln = nn.LayerNorm(dim)
    x = torch.randn(4, 16, dim)
    with torch.no_grad():
        rms_out = rms(x)
        ln_out = ln(x)
    # They should differ
    max_diff = (rms_out - ln_out).abs().max().item()
    assert max_diff > 0.01, "RMSNorm and LayerNorm should produce different outputs"


# ---------------------------------------------------------------------------
# SwiGLU tests
# ---------------------------------------------------------------------------

def test_swiglu_output_shape():
    """SwiGLU output shape matches (batch, seq, d_model)."""
    d_model = 64
    swiglu = SwiGLU(d_model)
    x = torch.randn(4, 16, d_model)
    out = swiglu(x)
    assert out.shape == x.shape, f"Expected {x.shape}, got {out.shape}"


def test_swiglu_param_count():
    """
    SwiGLU (2/3 scaling) has approximately same params as ReLU FFN (4x).
    Allow 5% tolerance for rounding.
    """
    d_model = 256
    swiglu = SwiGLU(d_model)
    relu_ffn = ReLUFFN(d_model)
    swiglu_params = sum(p.numel() for p in swiglu.parameters())
    relu_params = sum(p.numel() for p in relu_ffn.parameters())
    ratio = swiglu_params / relu_params
    assert abs(ratio - 1.0) < 0.05, (
        f"SwiGLU/ReLU param ratio {ratio:.3f} should be ~1.0 (within 5%)"
    )


def test_swiglu_gate_effect():
    """
    SwiGLU with zero gate weights should produce zero output.
    (When W2 weights are 0, SiLU(0) = 0, so gate is 0, output is 0.)
    """
    d_model = 32
    swiglu = SwiGLU(d_model)
    with torch.no_grad():
        swiglu.W2.weight.fill_(0.0)

    x = torch.randn(2, 8, d_model)
    out = swiglu(x)
    max_val = out.abs().max().item()
    assert max_val < 1e-6, f"Zero gate should produce zero output, got {max_val}"


def test_relu_ffn_output_shape():
    """ReLUFFN output shape matches input."""
    d_model = 64
    ffn = ReLUFFN(d_model)
    x = torch.randn(4, 16, d_model)
    out = ffn(x)
    assert out.shape == x.shape
