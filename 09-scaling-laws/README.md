# 09 — Scaling Laws

## Context

Kaplan et al. 2020 ("Scaling Laws for Neural Language Models", OpenAI) established that language model loss follows smooth, predictable power laws as a function of three independent variables: model size (number of parameters), dataset size (number of tokens), and compute budget. The relationship takes the form:

```
L(N) ≈ a · N^b
```

where N is parameter count and b is a negative exponent — larger models reliably achieve lower loss. The key finding was that these relationships hold across many orders of magnitude with striking regularity, with an exponent of approximately −0.076 for loss vs. parameters when dataset size is not the bottleneck.

The practical implication was immediate: if you have more compute, spend it on a bigger model. This drove the GPT-3 era of scaling: if the loss curve is a predictable power law on a log-log plot, you can extrapolate to model sizes you haven't trained yet. Bigger was reliably better — a clean, actionable insight that shaped two years of LLM development.

This framing was later revised by Hoffmann et al. 2022 (Chinchilla), who showed the Kaplan scaling runs were undertrained. The optimal frontier — given a fixed compute budget — requires scaling model size and token count proportionally, not preferentially growing the model. But the scaling law methodology itself remains foundational.

## What this code does

`train_sweep.py` trains six GPT variants ranging from ~12K to ~3M parameters on the same Shakespeare corpus (char-level, 500 steps each) and writes `results.json` with each model's parameter count and validation loss.

`plot_scaling.py` loads those results and:

- Plots validation loss against parameter count on a log-log scale
- Fits a power law `loss = a · N^b` using `scipy.optimize.curve_fit`
- Reports the fitted exponent `b` and R² for the fit quality
- Saves `scaling_laws.png`

On the log-log plot, a power law appears as a straight line. If the fitted line is visually clean and R² is high (above 0.95), the data follows the expected scaling behavior. The slope of that line is the exponent `b`. Typical values from this small experiment land around −0.05 to −0.10, consistent with Kaplan et al.'s reported −0.076.

## Key implementation details

**Why a log-log plot?** A power law `y = a · x^b` becomes `log(y) = log(a) + b · log(x)` — a linear equation in log space. Plotting on log-log axes turns an exponential curve into a straight line, making it easy to visually confirm the functional form and read off the slope as the exponent.

**What a power law looks like on that scale.** If the points fall on a straight line in log-log space, the power law hypothesis holds. Curvature would indicate the relationship is more complex (or the model is in a data-limited regime for some sizes). A well-behaved scaling experiment produces near-collinear points across several orders of magnitude in N.

**The R² metric.** R² (coefficient of determination) measures what fraction of variance in the log-loss values is explained by the power-law fit. R² = 1.0 is a perfect fit; values above 0.95 indicate the power law is a good description of the data. Low R² suggests the relationship isn't cleanly power-law — often because the models aren't all trained to the same "compute-optimal" point, or because the range of N is too narrow.

**curve_fit mechanics.** `scipy.optimize.curve_fit` uses nonlinear least squares (Levenberg-Marquardt). The fit is done in linear (not log) space, so larger loss values have proportionally more influence. Fitting in log space (i.e., fitting `log(loss) = log(a) + b * log(N)`) gives equal weight to each decade and is more robust — a natural extension if results look noisy.

## What's deliberately omitted

**Dataset size scaling.** Kaplan et al. also characterize `L(D)` — loss as a function of dataset tokens. Reproducing that requires training the same model on different data volumes, which multiplies training cost. This experiment holds data fixed and varies model size only.

**Compute-optimal frontier (Chinchilla).** The Kaplan paper recommended allocating most of a compute budget to model size. Hoffmann et al. 2022 showed this was wrong — the optimal ratio of model parameters to training tokens is roughly 1:20. Mapping out that frontier requires a grid of (model size, token count) experiments. See `articles/llm-10-chinchilla` for the conceptual treatment.

**Proper hyperparameter tuning per model size.** All six models here use the same learning rate (3e-4) and batch size (32). In rigorous scaling studies, each model size is tuned independently — larger models often benefit from different learning rate schedules. Using a fixed LR introduces a confound: some models may be under- or over-tuned, noisying the power law. This is acceptable for illustration but would not be acceptable for a published scaling study.
