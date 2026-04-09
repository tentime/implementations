"""
Train a 2-layer MLP to solve XOR.

XOR is the canonical demonstration that hidden layers are necessary:
no linear model can separate the XOR truth table, but a single hidden
layer with sigmoid activations can.

Truth table:
    [0, 0] -> 0
    [0, 1] -> 1
    [1, 0] -> 1
    [1, 1] -> 0

We train with full-batch gradient descent (all 4 examples every step)
because the dataset is tiny.
"""

import numpy as np
from mlp import MLP, SGDOptimizer, mse_loss

# Reproducibility
np.random.seed(42)

# ---------------------------------------------------------------------------
# XOR dataset
# ---------------------------------------------------------------------------

X = np.array([[0, 0],
              [0, 1],
              [1, 0],
              [1, 1]], dtype=float)

y = np.array([[0],
              [1],
              [1],
              [0]], dtype=float)

# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

model = MLP(input_size=2, hidden_size=4, output_size=1)
optimizer = SGDOptimizer()

LR = 0.5
N_STEPS = 10_000
PRINT_EVERY = 1_000

print("Training MLP on XOR (10,000 steps, lr=0.5)\n")
print(f"{'Step':>8}  {'Loss':>12}")
print("-" * 24)

for step in range(1, N_STEPS + 1):
    # Forward pass: compute predictions
    preds = model.forward(X)

    # Compute loss
    loss = mse_loss(preds, y)

    # Backward pass: compute gradients
    grads = model.backward(X, y)

    # Parameter update
    optimizer.step(model, grads, lr=LR)

    if step % PRINT_EVERY == 0:
        print(f"{step:>8}  {loss:>12.6f}")

# ---------------------------------------------------------------------------
# Final results
# ---------------------------------------------------------------------------

final_preds = model.forward(X)
final_loss = mse_loss(final_preds, y)

print("\n" + "=" * 40)
print("Final predictions vs. targets:")
print(f"{'Input':<12}  {'Target':>8}  {'Predicted':>10}  {'Rounded':>8}")
print("-" * 44)
for i in range(len(X)):
    inp = f"[{int(X[i,0])}, {int(X[i,1])}]"
    tgt = int(y[i, 0])
    pred = final_preds[i, 0]
    rounded = int(round(pred))
    print(f"{inp:<12}  {tgt:>8}  {pred:>10.4f}  {rounded:>8}")

print(f"\nFinal MSE loss: {final_loss:.6f}")

# ---------------------------------------------------------------------------
# Sanity check
# ---------------------------------------------------------------------------

assert final_loss < 0.01, (
    f"Training did not converge: final loss {final_loss:.6f} >= 0.01. "
    "Try more steps or a different learning rate."
)
print("\nAssertion passed: final loss < 0.01")
