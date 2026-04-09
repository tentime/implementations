"""
LoRA: Low-Rank Adaptation of Large Language Models.
Implements LoRALinear and utilities to inject, save, and load adapters.
"""

from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# LoRALinear
# ---------------------------------------------------------------------------

class LoRALinear(nn.Module):
    """
    Drop-in replacement for nn.Linear that adds a low-rank adapter.

    The forward pass computes:
        W @ x + (alpha/rank) * B @ A @ x

    where:
        W is the frozen pretrained weight (requires_grad=False)
        A is (in_features × rank), initialized from N(0, 0.02)
        B is (rank × out_features), initialized to ZEROS

    B is initialized to zero so that at the start of fine-tuning,
    the LoRA adapter contributes nothing (identical to the base model).
    This is the key initialization detail from the paper.

    Args:
        in_features: input dimension
        out_features: output dimension
        rank: rank of the low-rank decomposition (r in the paper)
        alpha: scaling factor (lora_alpha in the paper)
        base_weight: optional pretrained weight tensor to load into W
    """

    def __init__(self, in_features, out_features, rank=4, alpha=16, base_weight=None):
        super().__init__()
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank

        # Frozen base weight  (out_features, in_features) — same layout as nn.Linear
        self.W = nn.Parameter(
            torch.zeros(out_features, in_features) if base_weight is None
            else base_weight.clone(),
            requires_grad=False,
        )

        # Trainable low-rank matrices
        self.A = nn.Parameter(torch.empty(in_features, rank))
        self.B = nn.Parameter(torch.zeros(rank, out_features))

        nn.init.normal_(self.A, std=0.02)
        # B stays zeros — LoRA adapter starts as identity (contributes 0)

    def forward(self, x):
        # Base: x @ W.T  (standard linear, frozen)
        base_out = F.linear(x, self.W)
        # LoRA: x @ A @ B * scaling
        lora_out = (x @ self.A @ self.B) * self.scaling
        return base_out + lora_out

    def extra_repr(self):
        return f"rank={self.rank}, alpha={self.alpha}, scaling={self.scaling:.3f}"


# ---------------------------------------------------------------------------
# inject_lora
# ---------------------------------------------------------------------------

def inject_lora(model, rank=4, alpha=16, target_modules=("q_proj", "v_proj")):
    """
    Walk the model tree and replace nn.Linear layers whose name ends with
    any string in target_modules with LoRALinear.

    All model parameters are first frozen. Only the LoRA A and B matrices
    remain trainable. This reflects the real use-case: the pretrained base
    model is fixed; only the adapter is updated during fine-tuning.

    Prints a summary:
        Injected LoRA into N layers
        Trainable parameters: X / Y (Z%)

    Returns the modified model.
    """
    # Freeze everything first — LoRA trains only A and B
    for param in model.parameters():
        param.requires_grad_(False)

    replaced = 0

    # Collect replacements first to avoid mutating while iterating
    replacements = []
    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        if not any(name.endswith(target) for target in target_modules):
            continue

        # Find the parent module and the attribute name on it
        parts = name.split(".")
        parent = model
        for part in parts[:-1]:
            parent = getattr(parent, part)
        attr_name = parts[-1]

        lora_layer = LoRALinear(
            in_features=module.in_features,
            out_features=module.out_features,
            rank=rank,
            alpha=alpha,
            base_weight=module.weight,
        )
        replacements.append((parent, attr_name, lora_layer))

    for parent, attr_name, lora_layer in replacements:
        setattr(parent, attr_name, lora_layer)
        replaced += 1

    trainable, total = count_parameters(model)
    pct = 100.0 * trainable / total if total > 0 else 0.0
    print(f"Injected LoRA into {replaced} layers")
    print(f"Trainable parameters: {trainable:,} / {total:,} ({pct:.2f}%)")

    return model


# ---------------------------------------------------------------------------
# count_parameters
# ---------------------------------------------------------------------------

def count_parameters(model):
    """Return (trainable_params, total_params)."""
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total


# ---------------------------------------------------------------------------
# save_lora_adapter
# ---------------------------------------------------------------------------

def save_lora_adapter(model, path):
    """
    Save only the LoRA A and B matrices (not the frozen base weights).

    This produces a tiny file — typically <1% of the full model size.
    Prints the file size in KB.
    """
    adapter_state = {
        name: param
        for name, param in model.named_parameters()
        if param.requires_grad  # only A and B matrices
    }
    torch.save(adapter_state, path)
    size_kb = Path(path).stat().st_size / 1024
    print(f"Adapter saved: {size_kb:.1f} KB")


# ---------------------------------------------------------------------------
# load_lora_adapter
# ---------------------------------------------------------------------------

def load_lora_adapter(model, path):
    """Load adapter weights back into a model that has had inject_lora applied."""
    adapter_state = torch.load(path, map_location="cpu", weights_only=True)
    missing, unexpected = [], []

    model_params = dict(model.named_parameters())
    for name, param in adapter_state.items():
        if name in model_params:
            model_params[name].data.copy_(param.data)
        else:
            unexpected.append(name)

    # Report any params that were in the model but not in the checkpoint
    for name in model_params:
        if model_params[name].requires_grad and name not in adapter_state:
            missing.append(name)

    if missing:
        print(f"Warning: missing adapter keys: {missing}")
    if unexpected:
        print(f"Warning: unexpected adapter keys: {unexpected}")

    return model
