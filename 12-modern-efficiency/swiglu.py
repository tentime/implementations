"""
SwiGLU: A Gated Activation Function

Standard FFN (Transformer 2017):
    FFN(x) = max(0, x W1 + b1) W2 + b2   (ReLU)

SwiGLU (Noam Shazeer 2020):
    FFN(x) = (x W1 * SiLU(x W2)) W3
    where SiLU(x) = x * sigmoid(x)

The gate (SiLU(x W2)) controls which parts of the linear projection are active.
Unlike ReLU which hard-zeros negative values, SiLU is smooth and allows small
negative values through, enabling more selective information routing.

Used in PaLM, LLaMA, Mistral, Gemma, and virtually all modern large LLMs.

Parameter count note: SwiGLU uses 3 weight matrices (W1, W2, W3) instead of 2.
To keep total parameter count equal to a standard 4x FFN, the intermediate
dimension is scaled by 2/3:
    SwiGLU params: 3 * d_model * d_ff_swi  where d_ff_swi = 4 * d_model * 2/3
                 = 3 * d_model * (8/3 * d_model) = 8 * d_model^2
    ReLU params:   2 * d_model * d_ff_relu  where d_ff_relu = 4 * d_model
                 = 8 * d_model^2

Reference: Shazeer, "GLU Variants Improve Transformer", 2020.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SwiGLU(nn.Module):
    """
    SwiGLU feed-forward layer.

    FFN(x) = W3( W1(x) * SiLU(W2(x)) )

    W1: linear projection (the "value" branch)
    W2: gating projection (the "gate" branch)
    W3: output projection
    """

    def __init__(self, d_model, d_ff=None):
        super().__init__()
        if d_ff is None:
            # Scale by 2/3 to match parameter count of standard 4x FFN with 2 matrices
            d_ff = int(d_model * 4 * 2 / 3)
        self.W1 = nn.Linear(d_model, d_ff, bias=False)  # value branch
        self.W2 = nn.Linear(d_model, d_ff, bias=False)  # gate branch
        self.W3 = nn.Linear(d_ff, d_model, bias=False)  # output projection
        self.d_ff = d_ff

    def forward(self, x):
        # Gate controls which parts of the value projection are active
        return self.W3(self.W1(x) * F.silu(self.W2(x)))


class ReLUFFN(nn.Module):
    """Standard ReLU FFN for comparison."""

    def __init__(self, d_model, d_ff=None):
        super().__init__()
        if d_ff is None:
            d_ff = d_model * 4
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff, bias=False),
            nn.ReLU(),
            nn.Linear(d_ff, d_model, bias=False),
        )
        self.d_ff = d_ff

    def forward(self, x):
        return self.net(x)


def compare_param_counts(d_model=256):
    """
    Show that SwiGLU with 2/3 scaling has approximately the same parameter
    count as a standard ReLU FFN with 4x expansion.
    """
    swiglu = SwiGLU(d_model)
    relu_ffn = ReLUFFN(d_model)

    swiglu_params = sum(p.numel() for p in swiglu.parameters())
    relu_params = sum(p.numel() for p in relu_ffn.parameters())
    ratio = swiglu_params / relu_params

    print(f"d_model = {d_model}")
    print(f"SwiGLU  d_ff = {swiglu.d_ff}  (= {d_model} * 4 * 2/3)")
    print(f"ReLU    d_ff = {relu_ffn.d_ff}  (= {d_model} * 4)")
    print(f"SwiGLU  params: {swiglu_params:,}")
    print(f"ReLU    params: {relu_params:,}")
    print(f"Ratio:  {ratio:.3f}  (target: ~1.0)")
    return swiglu_params, relu_params


if __name__ == "__main__":
    compare_param_counts()
