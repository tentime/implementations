"""
Run all four efficiency comparisons and print a summary table.

Usage: python benchmark.py
"""

from rope import demonstrate_relative_position_property
from flash_attention_concept import benchmark_memory
from rmsnorm import benchmark_vs_layernorm
from swiglu import SwiGLU, ReLUFFN


def main():
    print("=" * 60)
    print("1. RoPE: Relative Position Property")
    print("=" * 60)
    demonstrate_relative_position_property()

    print("\n" + "=" * 60)
    print("2. Flash Attention: Memory Scaling")
    print("=" * 60)
    benchmark_memory()

    print("\n" + "=" * 60)
    print("3. RMSNorm vs LayerNorm: Speed")
    print("=" * 60)
    benchmark_vs_layernorm()

    print("\n" + "=" * 60)
    print("4. SwiGLU vs ReLU: Parameter Count")
    print("=" * 60)
    # Show that SwiGLU with d_ff=2/3*4*d_model has same params as ReLU with d_ff=4*d_model
    d_model = 256
    swiglu = SwiGLU(d_model)
    relu_ffn = ReLUFFN(d_model)
    swiglu_params = sum(p.numel() for p in swiglu.parameters())
    relu_params = sum(p.numel() for p in relu_ffn.parameters())
    print(f"SwiGLU params: {swiglu_params:,}")
    print(f"ReLU FFN params: {relu_params:,}")
    print(f"Ratio: {swiglu_params / relu_params:.3f} (target: ~1.0)")


if __name__ == "__main__":
    main()
