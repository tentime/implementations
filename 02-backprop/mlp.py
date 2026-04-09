"""
Two-layer MLP with fully manual backpropagation.

This is pedagogical code. Every gradient computation is written out
explicitly with a comment tying it to the chain rule term it represents.
No autograd. No PyTorch. Just numpy and algebra.

Network architecture:

    Input X  (N, input_size)
        |
       W1, b1   (input_size, hidden_size)
        |
       z1 = X @ W1 + b1
        |
       a1 = sigmoid(z1)
        |
       W2, b2   (hidden_size, output_size)
        |
       z2 = a1 @ W2 + b2
        |
       a2 = sigmoid(z2)   <- predictions
        |
    Loss = MSE(a2, y)
"""

import numpy as np


# ---------------------------------------------------------------------------
# Activation function
# ---------------------------------------------------------------------------

def sigmoid(x):
    """
    Logistic sigmoid: sigma(x) = 1 / (1 + exp(-x)).

    Clips x to [-500, 500] to avoid overflow in exp().
    Maps any real number to (0, 1), which is what we want for binary outputs.
    """
    return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))


def sigmoid_deriv(x):
    """
    Derivative of sigmoid with respect to its *pre-activation* input x.

    d/dx sigmoid(x) = sigmoid(x) * (1 - sigmoid(x))

    This is the term that appears in the chain rule whenever we
    backpropagate through a sigmoid activation.
    """
    s = sigmoid(x)
    return s * (1.0 - s)


# ---------------------------------------------------------------------------
# Loss function
# ---------------------------------------------------------------------------

def mse_loss(pred, target):
    """
    Mean Squared Error averaged over all examples and output units.

    L = (1/N) * sum_i (pred_i - target_i)^2
    """
    return float(np.mean((pred - target) ** 2))


def mse_loss_deriv(pred, target):
    """
    Gradient of MSE loss with respect to *pred*.

    dL/d(pred) = (2/N) * (pred - target)

    Shape matches pred: (N, output_size).
    """
    N = pred.shape[0]
    return (2.0 / N) * (pred - target)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class MLP:
    """
    Two-layer Multi-Layer Perceptron.

    Weights are initialised with small random values (scaled by 0.5) so that
    sigmoids start in their responsive region — not so small that gradients
    vanish immediately, not so large that they saturate.  Biases start at zero.

    All intermediate values computed during forward() are stored as instance
    attributes so backward() can reuse them without recomputation — this is the
    standard pattern used by every autograd engine.
    """

    def __init__(self, input_size=2, hidden_size=4, output_size=1):
        # Layer 1 parameters
        # Scale of 0.5 keeps initial activations in the sigmoid's responsive
        # region (away from saturation at 0 and 1) while still being "small".
        self.W1 = np.random.randn(input_size, hidden_size) * 0.5   # (2, 4)
        self.b1 = np.zeros((1, hidden_size))                        # (1, 4)

        # Layer 2 parameters
        self.W2 = np.random.randn(hidden_size, output_size) * 0.5  # (4, 1)
        self.b2 = np.zeros((1, output_size))                        # (1, 1)

        # Placeholders — filled by forward().
        self.z1 = None
        self.a1 = None
        self.z2 = None
        self.a2 = None

    def forward(self, X):
        """
        Compute the network's prediction for input X.

        Stores all intermediate values (z1, a1, z2, a2) on self so that
        backward() can reference them.

        Parameters
        ----------
        X : ndarray of shape (N, input_size)

        Returns
        -------
        a2 : ndarray of shape (N, output_size)
            The network's output after the second sigmoid.
        """
        self.z1 = X @ self.W1 + self.b1        # Pre-activation, hidden layer
        self.a1 = sigmoid(self.z1)              # Hidden activations
        self.z2 = self.a1 @ self.W2 + self.b2  # Pre-activation, output layer
        self.a2 = sigmoid(self.z2)              # Output activations (predictions)
        return self.a2

    def backward(self, X, y):
        """
        Compute gradients of MSE loss w.r.t. all parameters.

        Every line below annotates which term in the chain rule it represents.
        The chain rule for this network (reading bottom-up):

            dL/dW2 = dL/da2 * da2/dz2 * dz2/dW2
            dL/dW1 = dL/da2 * da2/dz2 * dz2/da1 * da1/dz1 * dz1/dW1

        Parameters
        ----------
        X : ndarray of shape (N, input_size)
        y : ndarray of shape (N, output_size)

        Returns
        -------
        dict with keys 'dW1', 'db1', 'dW2', 'db2'
        """
        N = X.shape[0]

        # --- Output layer gradients ---

        # dL/da2 : gradient of loss w.r.t. network output
        dL_da2 = mse_loss_deriv(self.a2, y)           # (N, output_size)

        # da2/dz2 : sigmoid derivative at the output pre-activation
        da2_dz2 = sigmoid_deriv(self.z2)              # (N, output_size)

        # dL/dz2 = dL/da2 * da2/dz2  (element-wise, chain rule)
        dL_dz2 = dL_da2 * da2_dz2                     # (N, output_size)

        # dz2/dW2 = a1  (because z2 = a1 @ W2 + b2)
        # dL/dW2  = a1^T @ dL/dz2  (accumulated over the N examples)
        dW2 = self.a1.T @ dL_dz2                      # (hidden_size, output_size)

        # dz2/db2 = 1  ->  dL/db2 = sum of dL/dz2 over examples
        db2 = np.sum(dL_dz2, axis=0, keepdims=True)   # (1, output_size)

        # --- Hidden layer gradients ---

        # dz2/da1 = W2^T  (because z2 = a1 @ W2 + b2)
        # dL/da1  = dL/dz2 @ W2^T  (chain rule: pass error back through W2)
        dL_da1 = dL_dz2 @ self.W2.T                   # (N, hidden_size)

        # da1/dz1 : sigmoid derivative at the hidden pre-activation
        da1_dz1 = sigmoid_deriv(self.z1)              # (N, hidden_size)

        # dL/dz1 = dL/da1 * da1/dz1  (element-wise, chain rule through sigmoid)
        dL_dz1 = dL_da1 * da1_dz1                     # (N, hidden_size)

        # dz1/dW1 = X  (because z1 = X @ W1 + b1)
        # dL/dW1  = X^T @ dL/dz1
        dW1 = X.T @ dL_dz1                            # (input_size, hidden_size)

        # dz1/db1 = 1  ->  dL/db1 = sum of dL/dz1 over examples
        db1 = np.sum(dL_dz1, axis=0, keepdims=True)   # (1, hidden_size)

        return {"dW1": dW1, "db1": db1, "dW2": dW2, "db2": db2}


# ---------------------------------------------------------------------------
# Optimizer
# ---------------------------------------------------------------------------

class SGDOptimizer:
    """
    Vanilla Stochastic Gradient Descent.

    Applies the update rule:
        theta <- theta - lr * grad_theta

    for each parameter theta.  No momentum, no weight decay, no learning
    rate schedule — those come later.
    """

    def step(self, model, grads, lr):
        """
        Update model parameters in-place.

        Parameters
        ----------
        model : MLP
        grads : dict returned by MLP.backward()
        lr : float  — learning rate
        """
        model.W1 -= lr * grads["dW1"]
        model.b1 -= lr * grads["db1"]
        model.W2 -= lr * grads["dW2"]
        model.b2 -= lr * grads["db2"]
