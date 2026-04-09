import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from scipy.optimize import curve_fit

# Import the module under test
sys.path.insert(0, str(Path(__file__).parent))
from train_sweep import CONFIGS, GPT, count_params


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_param_counts_monotonic():
    """The 6 model configs produce strictly increasing param counts."""
    counts = [count_params(GPT(cfg)) for cfg in CONFIGS]
    for i in range(len(counts) - 1):
        assert counts[i] < counts[i + 1], (
            f"Config {i} has {counts[i]:,} params but config {i+1} has "
            f"{counts[i+1]:,} — expected strictly increasing."
        )


@pytest.mark.slow
def test_larger_model_lower_loss():
    """Run the sweep and verify larger models achieve lower val loss (at least 4/5 consecutive pairs)."""
    result = subprocess.run(
        [sys.executable, "train_sweep.py"],
        capture_output=True,
        text=True,
        cwd=Path(__file__).parent,
    )
    assert result.returncode == 0, f"train_sweep.py failed:\n{result.stderr}"

    results_path = Path(__file__).parent / "results.json"
    with open(results_path) as f:
        results = json.load(f)

    losses = [r["val_loss"] for r in results]

    # Check that at least 4 out of 5 consecutive pairs are decreasing
    decreasing_pairs = sum(
        1 for i in range(len(losses) - 1) if losses[i] > losses[i + 1]
    )
    assert decreasing_pairs >= 4, (
        f"Expected at least 4/5 consecutive model pairs to decrease in loss. "
        f"Got {decreasing_pairs}. Losses: {losses}"
    )


def test_power_law_fit(tmp_path):
    """
    Given synthetic (params, loss) data that follows a power law,
    verify curve_fit returns an exponent within 10% of the true value.
    """
    true_a = 5.0
    true_b = -0.08

    # Generate clean synthetic data
    n_params = np.array([12_000, 70_000, 270_000, 530_000, 2_000_000, 3_000_000], dtype=float)
    val_loss = true_a * np.power(n_params, true_b)

    def power_law(n, a, b):
        return a * np.power(n, b)

    popt, _ = curve_fit(power_law, n_params, val_loss, p0=[10.0, -0.05], maxfev=10000)
    _, b_fit = popt

    tolerance = 0.10  # 10%
    assert abs(b_fit - true_b) / abs(true_b) < tolerance, (
        f"Fitted exponent {b_fit:.4f} deviates more than 10% from true value {true_b}"
    )
