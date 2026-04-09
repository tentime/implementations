"""
Pytest tests for mlp.py.
"""

import numpy as np
import pytest
from mlp import MLP, SGDOptimizer, mse_loss, mse_loss_deriv


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def xor_data():
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
    y = np.array([[0], [1], [1], [0]], dtype=float)
    return X, y


# ---------------------------------------------------------------------------
# Test 1: forward output shape
# ---------------------------------------------------------------------------

def test_forward_output_shape(xor_data):
    """
    MLP.forward() on a (4, 2) input must return a (4, 1) array.

    Verifies that matrix dimensions are wired correctly through both layers.
    """
    np.random.seed(42)
    X, _ = xor_data
    model = MLP(input_size=2, hidden_size=4, output_size=1)
    out = model.forward(X)
    assert out.shape == (4, 1), (
        f"Expected output shape (4, 1), got {out.shape}"
    )


# ---------------------------------------------------------------------------
# Test 2: all gradients are finite after one backward pass
# ---------------------------------------------------------------------------

def test_backward_gradients_finite(xor_data):
    """
    After a single forward + backward pass, no gradient should be NaN or Inf.

    NaN gradients usually indicate a numerical problem in the sigmoid or loss
    derivatives; Inf gradients indicate exploding values.
    """
    np.random.seed(42)
    X, y = xor_data
    model = MLP(input_size=2, hidden_size=4, output_size=1)
    model.forward(X)
    grads = model.backward(X, y)

    for name, g in grads.items():
        assert np.all(np.isfinite(g)), (
            f"Gradient '{name}' contains non-finite values: {g}"
        )


# ---------------------------------------------------------------------------
# Test 3: XOR convergence after 5000 steps
# ---------------------------------------------------------------------------

def test_xor_convergence(xor_data):
    """
    After 5,000 training steps with lr=0.5, MSE loss should be below 0.05.

    This confirms that the gradient direction and optimizer are correctly
    implemented — a random-walk model would not converge.
    """
    np.random.seed(42)
    X, y = xor_data
    model = MLP(input_size=2, hidden_size=4, output_size=1)
    optimizer = SGDOptimizer()

    for _ in range(5000):
        model.forward(X)
        grads = model.backward(X, y)
        optimizer.step(model, grads, lr=0.5)

    final_preds = model.forward(X)
    loss = mse_loss(final_preds, y)

    assert loss < 0.05, (
        f"Expected loss < 0.05 after 5000 steps, got {loss:.4f}"
    )


# ---------------------------------------------------------------------------
# Test 4: numerical gradient check for dW1
# ---------------------------------------------------------------------------

def test_numerical_gradient_check(xor_data):
    """
    Compare the analytical dW1 from backward() against a finite-difference
    approximation.

    Finite difference for element (i, j) of W1:
        grad_numerical[i,j] = (L(W1 + eps*e_{ij}) - L(W1 - eps*e_{ij})) / (2*eps)

    Tolerance is 1e-4; this is looser than machine precision because finite
    differences themselves have numerical error proportional to eps.

    A mismatch here indicates a bug in how chain-rule terms are combined in
    backward() — typically a missing transpose or wrong axis in a sum.
    """
    np.random.seed(42)
    X, y = xor_data
    model = MLP(input_size=2, hidden_size=4, output_size=1)

    # Analytical gradient
    model.forward(X)
    grads = model.backward(X, y)
    dW1_analytical = grads["dW1"]

    # Numerical gradient via central difference
    eps = 1e-5
    dW1_numerical = np.zeros_like(model.W1)

    for i in range(model.W1.shape[0]):
        for j in range(model.W1.shape[1]):
            # Perturb W1[i,j] positively
            model.W1[i, j] += eps
            loss_plus = mse_loss(model.forward(X), y)

            # Perturb W1[i,j] negatively
            model.W1[i, j] -= 2 * eps
            loss_minus = mse_loss(model.forward(X), y)

            # Restore original value
            model.W1[i, j] += eps

            dW1_numerical[i, j] = (loss_plus - loss_minus) / (2 * eps)

    max_error = np.max(np.abs(dW1_analytical - dW1_numerical))
    assert max_error < 1e-4, (
        f"Gradient check failed: max |analytical - numerical| = {max_error:.2e} "
        f"(tolerance 1e-4).\n"
        f"Analytical dW1:\n{dW1_analytical}\n"
        f"Numerical  dW1:\n{dW1_numerical}"
    )
