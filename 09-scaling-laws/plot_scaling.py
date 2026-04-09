"""
Load results.json and produce a scaling law plot.
- X axis: log(n_params)
- Y axis: val_loss (or log(val_loss))
- Fit a power law: loss = a * params^b using scipy.optimize.curve_fit
- Print the fitted exponent b
- Print: "Kaplan et al. 2020 report ~-0.076 for loss vs params"
- Save plot to scaling_laws.png
"""

import json
import math

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


def power_law(n, a, b):
    """loss = a * n^b"""
    return a * np.power(n, b)


def r_squared(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1.0 - ss_res / ss_tot


def main():
    with open("results.json") as f:
        results = json.load(f)

    n_params = np.array([r["n_params"] for r in results], dtype=float)
    val_loss = np.array([r["val_loss"] for r in results], dtype=float)

    # Fit power law: loss = a * params^b
    # Initial guess: a=10, b=-0.05
    popt, _ = curve_fit(power_law, n_params, val_loss, p0=[10.0, -0.05], maxfev=10000)
    a_fit, b_fit = popt

    y_pred = power_law(n_params, a_fit, b_fit)
    r2 = r_squared(val_loss, y_pred)

    print(f"Fitted power law: loss = {a_fit:.4f} * N^{b_fit:.4f}")
    print(f"Fitted exponent b = {b_fit:.4f}")
    print(f"R² = {r2:.4f}")
    print()
    print("Kaplan et al. 2020 report ~-0.076 for loss vs params")

    # --- Plot ---
    log_n = np.log10(n_params)
    log_loss = np.log10(val_loss)

    n_smooth = np.logspace(np.log10(n_params.min()), np.log10(n_params.max()), 200)
    loss_smooth = power_law(n_smooth, a_fit, b_fit)

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.scatter(log_n, log_loss, color="steelblue", s=80, zorder=5, label="Trained models")

    ax.plot(
        np.log10(n_smooth),
        np.log10(loss_smooth),
        color="tomato",
        linewidth=2,
        label=f"Power-law fit  (b={b_fit:.3f},  R²={r2:.3f})",
    )

    # Annotate points
    for n, l in zip(n_params, val_loss):
        ax.annotate(
            f"{int(n):,}",
            xy=(math.log10(n), math.log10(l)),
            xytext=(4, 4),
            textcoords="offset points",
            fontsize=7,
            color="steelblue",
        )

    ax.set_xlabel("log₁₀(Parameters)", fontsize=12)
    ax.set_ylabel("log₁₀(Validation Loss)", fontsize=12)
    ax.set_title("Scaling Laws: Validation Loss vs Model Size", fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("scaling_laws.png", dpi=150)
    print("\nPlot saved to scaling_laws.png")


if __name__ == "__main__":
    main()
