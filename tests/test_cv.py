"""Tests for k-fold cross-validation in `uq.workflow._k_fold_cv_error`.

Triage 3 / Gap 3 — generalization estimate when no independent held-out
test set is available (BYO mode, or default with --n-test 0).
"""

from __future__ import annotations

import numpy as np
import pytest

from uq.workflow import _k_fold_cv_error


def _legendre_eval(germ: np.ndarray, coeffs: np.ndarray) -> np.ndarray:
    """Evaluate the linear Legendre polynomial Σ c_i x_i + c_0 on each row."""
    # Y = c_0 + c_1 * x_1 + c_2 * x_2 + ...   (separable, exact at order ≥ 1)
    return coeffs[0] + germ @ coeffs[1:]


def test_cv_error_is_tiny_for_truly_linear_data() -> None:
    """When Y is exactly Σ c_i x_i, an order-1 PCE fits perfectly so CV ≈ 0."""
    rng = np.random.default_rng(0)
    germ = rng.uniform(-1.0, 1.0, size=(60, 3))
    # Output is a pure linear function of germ — within the order-1 basis exactly
    coeffs = np.array([0.5, 2.0, -1.0, 0.7])
    Y = _legendre_eval(germ, coeffs).reshape(-1, 1)

    relerr_cv, n_folds = _k_fold_cv_error(germ, Y, polynomial_order=1, n_folds=5)
    assert n_folds == 5
    # Linear data fitted by linear basis → CV error in numerical noise range
    assert relerr_cv[0] < 1e-10, f"got CV error {relerr_cv[0]}"


def test_cv_error_is_finite_for_noisy_data() -> None:
    """Noisy linear data should give a small-but-positive CV error."""
    rng = np.random.default_rng(0)
    germ = rng.uniform(-1.0, 1.0, size=(80, 3))
    coeffs = np.array([0.5, 2.0, -1.0, 0.7])
    Y = _legendre_eval(germ, coeffs).reshape(-1, 1)
    Y += rng.normal(0.0, 0.05, size=Y.shape)

    relerr_cv, n_folds = _k_fold_cv_error(germ, Y, polynomial_order=1, n_folds=5)
    assert n_folds == 5
    # Noise in Y → nonzero but bounded CV error (noise std / |Y| ratio)
    assert 0.0 < relerr_cv[0] < 0.2, f"got CV error {relerr_cv[0]}"


def test_cv_error_grows_with_polynomial_order_overfit() -> None:
    """Higher order on the same N → overfit → larger CV error than train."""
    rng = np.random.default_rng(0)
    # N=70 is comfortably above the skip threshold at order=3, d=3
    # (basis_size=20 → min_samples=50).
    germ = rng.uniform(-1.0, 1.0, size=(70, 3))
    # Truly linear signal — order-1 captures it; higher orders just fit noise.
    coeffs = np.array([0.5, 2.0, -1.0, 0.7])
    Y = _legendre_eval(germ, coeffs).reshape(-1, 1)
    Y += rng.normal(0.0, 0.02, size=Y.shape)

    err_p1, n_folds_p1 = _k_fold_cv_error(germ, Y, polynomial_order=1, n_folds=5)
    err_p3, n_folds_p3 = _k_fold_cv_error(germ, Y, polynomial_order=3, n_folds=5)
    assert n_folds_p1 == 5 and n_folds_p3 == 5, "both CV runs should execute"
    # Order-3 has 5× the basis terms and fits the noise; CV error rises.
    assert err_p3[0] >= err_p1[0], (
        f"expected p=3 CV error ({err_p3[0]}) >= p=1 ({err_p1[0]})"
    )


def test_cv_skipped_when_too_few_samples() -> None:
    """N < 2.5 * basis_size triggers the skip; returns (empty, 0)."""
    rng = np.random.default_rng(0)
    germ = rng.uniform(-1.0, 1.0, size=(10, 6))  # order=2 basis=28, need ≥ 70
    Y = rng.uniform(0.0, 1.0, size=(10, 1))

    relerr_cv, n_folds = _k_fold_cv_error(germ, Y, polynomial_order=2, n_folds=5)
    assert n_folds == 0
    assert relerr_cv.size == 0


def test_cv_handles_multi_output() -> None:
    """Multi-output Y returns one CV error per output column."""
    rng = np.random.default_rng(0)
    germ = rng.uniform(-1.0, 1.0, size=(60, 2))
    Y = np.column_stack([
        2.0 * germ[:, 0] + 0.1,             # linear in x_0
        -1.5 * germ[:, 1] + 0.05 * germ[:, 0] ** 2,  # mostly linear in x_1 + tiny x_0^2
    ])

    relerr_cv, n_folds = _k_fold_cv_error(germ, Y, polynomial_order=2, n_folds=5)
    assert n_folds == 5
    assert relerr_cv.shape == (2,)
    # Both fit by an order-2 basis with comfortable N → tiny CV error.
    assert (relerr_cv < 1e-8).all(), f"got {relerr_cv}"


def test_cv_reproducible_with_seed() -> None:
    """Same seed → same shuffle → same CV error."""
    rng = np.random.default_rng(0)
    germ = rng.uniform(-1.0, 1.0, size=(60, 2))
    Y = (2.0 * germ[:, 0] + rng.normal(0, 0.1, 60)).reshape(-1, 1)

    err_a, _ = _k_fold_cv_error(germ, Y, polynomial_order=1, seed=42)
    err_b, _ = _k_fold_cv_error(germ, Y, polynomial_order=1, seed=42)
    np.testing.assert_array_equal(err_a, err_b)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
