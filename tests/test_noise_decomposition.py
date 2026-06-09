"""Tests for `uq.workflow._aleatoric_noise_decomposition`.

Triage 5 / Gap 2 — separate the aleatoric (stochastic noise) component
of output variance from the epistemic (parameter-driven) component, so
Sobol indices can be interpreted against the actual explainable share.
"""

from __future__ import annotations

import numpy as np
import pytest

from uq.workflow import _aleatoric_noise_decomposition


def _build_cache_inputs(
    n_variants: int,
    n_seeds: int,
    n_timesteps_per_seed: int,
    variant_means: np.ndarray,
    noise_std: float,
    seed: int = 0,
) -> tuple[list[np.ndarray], list[dict[str, np.ndarray]]]:
    """Synthesize Y_timeseries + meta matching the cache contract.

    Each variant has `n_seeds` lineages, each lineage has `n_timesteps_per_seed`
    timesteps drawn from N(variant_means[i], noise_std²) for that variant.
    """
    rng = np.random.default_rng(seed)
    Y_ts: list[np.ndarray] = []
    Y_meta: list[dict[str, np.ndarray]] = []
    for i in range(n_variants):
        rows: list[np.ndarray] = []
        seeds: list[np.ndarray] = []
        for s in range(n_seeds):
            data = rng.normal(variant_means[i], noise_std, size=(n_timesteps_per_seed, 1))
            rows.append(data)
            seeds.append(np.full(n_timesteps_per_seed, s, dtype=np.int64))
        Y_ts.append(np.vstack(rows))
        Y_meta.append({"lineage_seed": np.concatenate(seeds)})
    return Y_ts, Y_meta


def test_pure_signal_no_noise() -> None:
    """Different X variants, identical Y per seed → signal=1, noise=0."""
    # Each variant has a distinct mean, and zero noise per seed.
    variant_means = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    Y_ts, Y_meta = _build_cache_inputs(
        n_variants=5, n_seeds=4, n_timesteps_per_seed=10,
        variant_means=variant_means, noise_std=0.0,
    )
    decomp = _aleatoric_noise_decomposition(Y_ts, Y_meta)
    assert decomp is not None
    assert decomp.status == "estimated"
    assert decomp.per_output_signal_fraction[0] == pytest.approx(1.0, abs=1e-12)
    assert decomp.per_output_noise_fraction[0] == pytest.approx(0.0, abs=1e-12)


def test_pure_noise_no_signal() -> None:
    """All variants have the same mean → noise dominates."""
    variant_means = np.full(5, 3.0)
    Y_ts, Y_meta = _build_cache_inputs(
        n_variants=5, n_seeds=4, n_timesteps_per_seed=20,
        variant_means=variant_means, noise_std=1.0, seed=0,
    )
    decomp = _aleatoric_noise_decomposition(Y_ts, Y_meta)
    assert decomp is not None
    assert decomp.status == "estimated"
    # Signal should be tiny — small finite-sample variance between grand means.
    assert decomp.per_output_signal_fraction[0] < 0.2, (
        f"got signal={decomp.per_output_signal_fraction[0]}"
    )
    assert decomp.per_output_noise_fraction[0] > 0.8, (
        f"got noise={decomp.per_output_noise_fraction[0]}"
    )


def test_mixed_signal_and_noise_recovered() -> None:
    """50/50 mix.

    Within-variant variance is computed over per-seed *means*, each a mean
    over n_timesteps_per_seed rows. So Var_within scales as
    noise_std² / n_timesteps_per_seed. For variant_means linspace(0, 4)
    (Var_between ≈ 1.33), and n_timesteps_per_seed = 50, we need
    noise_std ≈ sqrt(1.33 × 50) ≈ 8.2 to hit 50/50.
    """
    variant_means = np.linspace(0.0, 4.0, 30)  # variance ≈ 16/12 ≈ 1.33
    Y_ts, Y_meta = _build_cache_inputs(
        n_variants=30, n_seeds=8, n_timesteps_per_seed=50,
        variant_means=variant_means, noise_std=8.2, seed=0,
    )
    decomp = _aleatoric_noise_decomposition(Y_ts, Y_meta)
    assert decomp is not None
    s = decomp.per_output_signal_fraction[0]
    assert 0.35 < s < 0.65, f"expected signal ≈ 0.5, got {s}"


def test_single_seed_returns_not_estimable() -> None:
    """n_init_sims = 1 → no within-variant replicate → can't decompose."""
    variant_means = np.array([1.0, 2.0, 3.0])
    Y_ts, Y_meta = _build_cache_inputs(
        n_variants=3, n_seeds=1, n_timesteps_per_seed=10,
        variant_means=variant_means, noise_std=0.0,
    )
    decomp = _aleatoric_noise_decomposition(Y_ts, Y_meta)
    assert decomp is not None
    assert decomp.status == "not_estimable_single_seed"
    # Per-output entries are placeholder ones/zeros to satisfy shape contract.
    assert (decomp.per_output_signal_fraction == 1.0).all()
    assert (decomp.per_output_noise_fraction == 0.0).all()


def test_returns_none_when_no_lineage_seed_in_meta() -> None:
    """Cache without lineage_seed metadata → can't group → None."""
    rng = np.random.default_rng(0)
    Y_ts = [rng.normal(0.0, 1.0, size=(10, 1)) for _ in range(3)]
    Y_meta = [{"generation": np.zeros(10, dtype=np.int64)} for _ in range(3)]
    assert _aleatoric_noise_decomposition(Y_ts, Y_meta) is None


def test_returns_none_when_timeseries_empty() -> None:
    assert _aleatoric_noise_decomposition([], []) is None
    assert _aleatoric_noise_decomposition([], None) is None


def test_unbalanced_design_status() -> None:
    """Variants with different seed counts get status='unbalanced'."""
    rng = np.random.default_rng(0)
    Y_ts = [
        np.vstack([rng.normal(1.0, 0.1, (5, 1)), rng.normal(1.0, 0.1, (5, 1))]),  # 2 seeds
        rng.normal(2.0, 0.1, (10, 1)),                                             # 1 seed
        np.vstack([
            rng.normal(3.0, 0.1, (5, 1)),
            rng.normal(3.0, 0.1, (5, 1)),
            rng.normal(3.0, 0.1, (5, 1)),
        ]),                                                                         # 3 seeds
    ]
    Y_meta = [
        {"lineage_seed": np.array([0]*5 + [1]*5, dtype=np.int64)},
        {"lineage_seed": np.zeros(10, dtype=np.int64)},
        {"lineage_seed": np.array([0]*5 + [1]*5 + [2]*5, dtype=np.int64)},
    ]
    decomp = _aleatoric_noise_decomposition(Y_ts, Y_meta)
    assert decomp is not None
    assert decomp.status == "unbalanced"
    # Should still return per-output fractions
    assert decomp.per_output_signal_fraction.shape == (1,)


def test_multi_output_handled() -> None:
    """Decomposition runs independently per output column."""
    rng = np.random.default_rng(0)
    n_variants = 8
    Y_ts: list[np.ndarray] = []
    Y_meta: list[dict[str, np.ndarray]] = []
    for i in range(n_variants):
        # Output 0: pure signal (variant-dependent mean, zero noise)
        # Output 1: pure noise (mean = 5 for all, but per-tick noise)
        out0 = np.full((20, 1), float(i))
        out1 = rng.normal(5.0, 2.0, (20, 1))
        # Two seeds per variant
        ts_seed0 = np.hstack([out0[:10], out1[:10]])
        ts_seed1 = np.hstack([out0[10:], out1[10:]])
        Y_ts.append(np.vstack([ts_seed0, ts_seed1]))
        Y_meta.append({"lineage_seed": np.array([0]*10 + [1]*10, dtype=np.int64)})

    decomp = _aleatoric_noise_decomposition(Y_ts, Y_meta)
    assert decomp is not None
    # Output 0: all signal
    assert decomp.per_output_signal_fraction[0] > 0.95
    # Output 1: mostly noise
    assert decomp.per_output_noise_fraction[1] > 0.6


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
