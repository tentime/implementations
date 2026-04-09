import torch
import torch.nn as nn
import math


def precompute_freqs_cis(dim, max_seq_len, theta=10000.0):
    """
    Precompute the complex-valued rotation matrix for RoPE.

    For each dimension pair (2i, 2i+1), the frequency is:
        theta_i = 1 / (theta^(2i/dim))

    Returns freqs_cis: (max_seq_len, dim//2) complex tensor
    where freqs_cis[pos] = exp(i * pos * theta_i) for each i.

    The theta=10000 base follows the original RoPE paper (Su et al. 2021).
    Larger theta → slower rotation → encodes longer-range positions distinctly.
    """
    # One frequency per dimension pair
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
    positions = torch.arange(max_seq_len).float()
    # Outer product: (max_seq_len, dim//2)
    freqs = torch.outer(positions, freqs)
    # Convert to complex: e^{i*freq} = cos(freq) + i*sin(freq)
    return torch.polar(torch.ones_like(freqs), freqs)  # complex64


def apply_rotary_emb(xq, xk, freqs_cis):
    """
    Apply rotary embeddings to query and key tensors.

    Args:
        xq: (seq_len, n_heads, head_dim)
        xk: (seq_len, n_heads, head_dim)
        freqs_cis: (seq_len, head_dim//2)

    The key property: dot(RoPE(q, pos_i), RoPE(k, pos_j)) depends only on (i-j),
    not on i and j individually. This enables relative position awareness without
    modifying the attention mechanism itself.

    Implementation:
        1. Reshape last dim from head_dim to (head_dim//2, 2)
        2. View as complex numbers (each pair becomes one complex number)
        3. Multiply by the precomputed complex rotation e^{i*pos*theta}
        4. View the result back as real and flatten
    """
    # Reshape to complex: (..., head_dim) -> (..., head_dim//2) complex
    xq_r = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_r = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))

    # freqs_cis: (seq_len, head_dim//2) -> (seq_len, 1, head_dim//2) to broadcast over heads
    freqs_cis = freqs_cis.unsqueeze(1)

    # Rotate: complex multiplication encodes position
    xq_out = torch.view_as_real(xq_r * freqs_cis).flatten(2)
    xk_out = torch.view_as_real(xk_r * freqs_cis).flatten(2)

    return xq_out.type_as(xq), xk_out.type_as(xk)


def demonstrate_relative_position_property():
    """
    Show that dot(RoPE(q, i), RoPE(k, j)) depends only on (i-j).

    Test: for 3 pairs with the same relative distance (delta=3):
      - (pos_i=5,  pos_j=2)
      - (pos_i=10, pos_j=7)
      - (pos_i=20, pos_j=17)

    The dot products should all be equal (within 1e-5).
    Print the values to show this.
    """
    torch.manual_seed(42)

    head_dim = 16
    n_heads = 1
    max_pos = 32

    freqs_cis = precompute_freqs_cis(head_dim, max_pos)

    # Fixed query and key vectors (same for all tests — we only change position)
    q_vec = torch.randn(1, n_heads, head_dim)
    k_vec = torch.randn(1, n_heads, head_dim)

    test_pairs = [(5, 2), (10, 7), (20, 17)]
    delta = 3
    dot_products = []

    print(f"Relative position delta = {delta}")
    print(f"{'(pos_i, pos_j)':<18} | dot product")
    print("-" * 36)

    for pos_i, pos_j in test_pairs:
        # Apply RoPE at their respective positions
        q_rot, _ = apply_rotary_emb(q_vec, k_vec, freqs_cis[pos_i : pos_i + 1])
        _, k_rot = apply_rotary_emb(q_vec, k_vec, freqs_cis[pos_j : pos_j + 1])

        # Dot product: sum over head_dim (squeeze out batch and head dims)
        dot = (q_rot[0, 0] * k_rot[0, 0]).sum().item()
        dot_products.append(dot)
        print(f"  ({pos_i:2d}, {pos_j:2d})             | {dot:.6f}")

    # Verify all are equal within tolerance
    max_diff = max(abs(dp - dot_products[0]) for dp in dot_products)
    print(f"\nMax difference across pairs: {max_diff:.2e}  (should be < 1e-4)")
    assert max_diff < 1e-4, f"RoPE relative position property violated: max_diff={max_diff}"
    print("Property confirmed: dot product depends only on relative distance.")


if __name__ == "__main__":
    demonstrate_relative_position_property()
