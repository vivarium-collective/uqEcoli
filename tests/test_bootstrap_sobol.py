"""Tests for `uq.workflow._bootstrap_sobol_cis`.

Triage 4 / Gap 4b — non-parametric bootstrap CIs on Sobol indices, so
users can tell whether `S_Ti = 0.42` and `S_Ti = 0.38` are meaningfully
different.
"""

from __future__ import annotations

import numpy as np
import pytest

from uq.workflow import _bootstrap_sobol_cis


def _make_data(seed: int = 0, n: int = 80, d: int = 3):
    """Linear-in-germ data with a strong x0 effect — clear Sobol ranking."""
    rng = np.random.default_rng(seed)
    germ = rng.uniform(-1.0, 1.0, size=(n, d))
    # x0 dominates; x1 weak; x2 negligible.
    Y = (3.0 * germ[:, 0] + 1.0 * germ[:, 1] + 0.05 * germ[:, 2]).reshape(-1, 1)
    Y += rng.normal(0.0, 0.05, size=Y.shape)
    return germ, Y


def test_bootstrap_disabled_returns_none() -> None:
    germ, Y = _make_data()
    fo_ci, to_ci = _bootstrap_sobol_cis(
        germ, Y, polynomial_order=1,
        parameter_names=["a", "b", "c"], n_bootstrap=0,
    )
    assert fo_ci is None and to_ci is None


def test_bootstrap_ci_shape_is_n_params_by_two() -> None:
    germ, Y = _make_data()
    fo_ci, to_ci = _bootstrap_sobol_cis(
        germ, Y, polynomial_order=1,
        parameter_names=["a", "b", "c"], n_bootstrap=50,
    )
    assert fo_ci is not None and to_ci is not None
    assert fo_ci.shape == (3, 2)
    assert to_ci.shape == (3, 2)
    # Low < high column-wise
    assert (fo_ci[:, 0] <= fo_ci[:, 1]).all()
    assert (to_ci[:, 0] <= to_ci[:, 1]).all()


def test_bootstrap_ci_brackets_point_estimate() -> None:
    """The CI should contain the single-fit Sobol estimate for each param."""
    from uq.workflow import PCRV, _compute_sobol, _fit_surrogate, get_mi  # type: ignore[attr-defined]

    germ, Y = _make_data(seed=0)
    # Single fit → point Sobol estimate
    pcrv, _ = _fit_surrogate(germ, Y, polynomial_order=1, regression="lsq")
    point = _compute_sobol(pcrv, ["a", "b", "c"], Y)

    fo_ci, to_ci = _bootstrap_sobol_cis(
        germ, Y, polynomial_order=1,
        parameter_names=["a", "b", "c"], n_bootstrap=200, seed=0,
    )
    assert fo_ci is not None and to_ci is not None
    # The bracket should contain the point estimate (with small tolerance
    # for the case where the point lands right at a percentile boundary).
    for i in range(3):
        assert fo_ci[i, 0] - 1e-9 <= point.first_order[i] <= fo_ci[i, 1] + 1e-9, (
            f"first-order CI [{fo_ci[i, 0]}, {fo_ci[i, 1]}] does not bracket "
            f"point estimate {point.first_order[i]} for param {i}"
        )
        assert to_ci[i, 0] - 1e-9 <= point.total_order[i] <= to_ci[i, 1] + 1e-9, (
            f"total-order CI [{to_ci[i, 0]}, {to_ci[i, 1]}] does not bracket "
            f"point estimate {point.total_order[i]} for param {i}"
        )


def test_bootstrap_ranks_dominant_param_above_weak() -> None:
    """For data where x0 dominates, x0's CI should sit clearly above x2's."""
    germ, Y = _make_data(seed=0, n=100)
    fo_ci, to_ci = _bootstrap_sobol_cis(
        germ, Y, polynomial_order=1,
        parameter_names=["x0", "x1", "x2"], n_bootstrap=200, seed=0,
    )
    assert to_ci is not None
    # x0's lower bound should exceed x2's upper bound — non-overlapping CIs.
    assert to_ci[0, 0] > to_ci[2, 1], (
        f"x0 CI {to_ci[0]} should be entirely above x2 CI {to_ci[2]}"
    )


def test_bootstrap_reproducible_with_seed() -> None:
    germ, Y = _make_data()
    fo_a, to_a = _bootstrap_sobol_cis(
        germ, Y, polynomial_order=1, parameter_names=["a", "b", "c"],
        n_bootstrap=100, seed=42,
    )
    fo_b, to_b = _bootstrap_sobol_cis(
        germ, Y, polynomial_order=1, parameter_names=["a", "b", "c"],
        n_bootstrap=100, seed=42,
    )
    np.testing.assert_array_equal(fo_a, fo_b)
    np.testing.assert_array_equal(to_a, to_b)


def test_bootstrap_handles_multi_output() -> None:
    rng = np.random.default_rng(0)
    germ = rng.uniform(-1.0, 1.0, size=(80, 2))
    # Two outputs with different dominant parameters
    Y = np.column_stack([
        3.0 * germ[:, 0] + 0.1 * germ[:, 1],
        0.1 * germ[:, 0] + 2.0 * germ[:, 1],
    ])
    fo_ci, to_ci = _bootstrap_sobol_cis(
        germ, Y, polynomial_order=1, parameter_names=["a", "b"],
        n_bootstrap=100, seed=0,
    )
    assert fo_ci is not None and to_ci is not None
    assert fo_ci.shape == (2, 2)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
