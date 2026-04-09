# Manual Backpropagation — 2-Layer MLP

## Context

Before backpropagation, training multi-layer networks was an open problem. The perceptron (Rosenblatt, 1958) could learn, but only for networks without hidden layers. Hidden layers are necessary to learn non-linear functions, but without a principled way to assign credit to hidden-layer weights, researchers could not train them. Various heuristics were tried; none scaled.

Rumelhart, Hinton, and Williams' 1986 paper *Learning Representations by Back-propagating Errors* changed this. The key insight was that the chain rule of calculus, applied recursively from the output layer backward through each weight matrix, yields the exact gradient of the loss with respect to every parameter in the network. Credit assignment — the question of how much each hidden weight contributed to the final error — reduces to a sequence of matrix multiplications.

The paper demonstrated that networks trained with backprop could learn internal representations (hidden-layer activations that encode useful features) that neither the programmer nor the network designer specified. This was the conceptual breakthrough: the network discovers its own intermediate representations by gradient descent.

The XOR problem was a symbolic target. Minsky and Papert's 1969 book *Perceptrons* had proven that single-layer networks cannot solve XOR, and this was widely interpreted as a fundamental limitation of neural networks. Demonstrating that a two-layer network trivially solves XOR via backprop was partly a rhetorical rebuttal to that pessimism.

---

## What this code does

- `mlp.py` — Core library:
  - `sigmoid` and `sigmoid_deriv`: the activation function and its derivative w.r.t. the pre-activation input.
  - `mse_loss` and `mse_loss_deriv`: mean-squared error and its gradient w.r.t. predictions.
  - `MLP`: stores weights `W1, b1, W2, b2`; `forward()` computes and caches all intermediates; `backward()` implements the chain rule step by step with inline comments.
  - `SGDOptimizer`: applies `theta -= lr * grad` for each parameter.

- `train.py`: defines the XOR dataset, trains for 10,000 steps, prints loss every 1,000 steps, prints final predictions alongside targets, and asserts final MSE < 0.01.

- `test_mlp.py`: four pytest tests — output shape, gradient finiteness, convergence, and a numerical gradient check against finite differences.

During training you observe the loss start around 0.25 and decrease steadily, reaching below 0.01 by step 7,000–8,000. The final predictions are close to 0 for [0,0] and [1,1], and close to 1 for [0,1] and [1,0].

Run training:
```
python train.py
```

Run tests:
```
pytest test_mlp.py -v
```

---

## Key implementation details

**Stored intermediates.** `forward()` saves `z1, a1, z2, a2` as instance attributes on the model object. This is the pattern every autograd library uses: you record the computation graph during the forward pass so the backward pass can look up the values it needs without recomputing them. In PyTorch, this is done inside the `Function.forward()` method via `ctx.save_for_backward()`.

**Chain rule, line by line.** `backward()` is written so each line corresponds to exactly one chain rule term. The sequence is: `dL/da2` (loss gradient) → `dL/dz2` (through output sigmoid) → `dL/dW2`, `dL/db2` (through the linear layer) → `dL/da1` (through `W2`) → `dL/dz1` (through hidden sigmoid) → `dL/dW1`, `dL/db1`. Reading the comments in order gives you the full derivation.

**Why XOR requires a hidden layer.** XOR is not linearly separable: no single line in 2D input space can separate the positive class `{[0,1],[1,0]}` from the negative class `{[0,0],[1,1]}`. A network with no hidden layer is just logistic regression — a linear classifier — and cannot solve it. The hidden layer learns a change of basis: the four XOR inputs are mapped to a 4-dimensional hidden space where the two classes become linearly separable, and the output layer draws the boundary there.

**Numerical gradient check.** `test_numerical_gradient_check` verifies `dW1` by perturbing each element of `W1` by ±ε and measuring the resulting change in loss. This central-difference approximation has error O(ε²), so with ε=1e-5 we expect agreement to about 5 decimal places. If the analytical gradient were wrong by even a sign flip in one term, the max error would be orders of magnitude larger.

---

## What's deliberately omitted

**Batching.** This code does full-batch gradient descent: all 4 examples are used on every step. Real training uses mini-batches (typically 32–512 examples), which introduces noise that helps escape local minima and dramatically improves hardware utilisation. Adding batching would require a data loader and shuffle logic that would distract from the backprop story.

**Momentum and adaptive learning rates.** SGD with a fixed learning rate is the simplest optimizer. In practice, momentum (which accumulates a moving average of past gradients) and adaptive methods like Adam (which scales each parameter's update by the square root of its gradient history) are almost always used. They are omitted here because they wrap around the same gradient computation — understanding the raw gradient first makes the optimizers easier to reason about.

**Regularisation.** L2 weight decay and dropout are omitted. On a 4-example dataset they would do nothing useful.

**Autograd.** PyTorch, JAX, and TensorFlow all implement automatic differentiation — they trace the forward computation and automatically construct the backward pass. Using an autograd library here would hide the only thing this code is trying to show.
