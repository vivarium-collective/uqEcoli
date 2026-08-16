"""Tests for Q1 baseline variance budget computation.

Covers:
  - compute_variance_budget() with synthetic data (known partition)
  - VarianceBudget.export() JSON structure
  - generate_baseline_html_report() produces valid HTML
"""

from pathlib import Path

import numpy as np
import pytest


def _make_synthetic_timeseries(
    n_gens: int = 4,
    n_seeds: int = 3,
    n_steps_per_cell: int = 50,
    n_obs: int = 3,
    gen_effect: float = 5.0,
    seed_effect: float = 2.0,
    noise_std: float = 0.5,
    rng_seed: int = 42,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Build a synthetic timeseries with known variance structure.

    The first observable has strong generation effect,
    the second has strong seed effect,
    the third has strong growth-stage (θ) effect.
    """
    rng = np.random.RandomState(rng_seed)
    rows = []
    gen_labels = []
    seed_labels = []

    for g in range(n_gens):
        for s in range(n_seeds):
            for t in range(n_steps_per_cell):
                theta = t / max(n_steps_per_cell - 1, 1)  # 0→1
                obs = np.zeros(n_obs)
                # Obs 0: generation-dominated
                obs[0] = g * gen_effect + rng.randn() * noise_std
                # Obs 1: seed-dominated
                obs[1] = s * seed_effect + rng.randn() * noise_std
                # Obs 2: growth-stage-dominated (θ effect)
                obs[2] = 10.0 * theta + rng.randn() * noise_std
                rows.append(obs)
                gen_labels.append(g)
                seed_labels.append(s)

    ts = np.array(rows)
    meta = {
        "generation": np.array(gen_labels, dtype=np.int64),
        "lineage_seed": np.array(seed_labels, dtype=np.int64),
    }
    return ts, meta


class TestComputeVarianceBudget:
    def test_basic_structure(self):
        from uq.workflow import compute_variance_budget

        ts, meta = _make_synthetic_timeseries()
        obs_names = ["gen_dominated", "seed_dominated", "theta_dominated"]
        budget = compute_variance_budget(ts, meta, obs_names, n_bins=10)

        assert budget.observable_names == obs_names
        assert budget.total_variance.shape == (3,)
        assert budget.generation_variance.shape == (3,)
        assert budget.seed_variance.shape == (3,)
        assert budget.growth_stage_variance.shape == (3,)
        assert budget.residual_variance.shape == (3,)
        assert budget.n_generations == 4
        assert budget.n_seeds == 3
        assert budget.n_stages == 10

    def test_fractions_sum_to_one(self):
        from uq.workflow import compute_variance_budget

        ts, meta = _make_synthetic_timeseries()
        obs_names = ["a", "b", "c"]
        budget = compute_variance_budget(ts, meta, obs_names, n_bins=10)

        total_fracs = (
            budget.generation_fraction
            + budget.seed_fraction
            + budget.growth_stage_fraction
            + budget.residual_fraction
        )
        np.testing.assert_allclose(total_fracs, 1.0, atol=0.01)

    def test_generation_dominated_observable(self):
        from uq.workflow import compute_variance_budget

        ts, meta = _make_synthetic_timeseries(gen_effect=10.0, seed_effect=0.1, noise_std=0.1)
        obs_names = ["gen_dom", "seed_dom", "theta_dom"]
        budget = compute_variance_budget(ts, meta, obs_names, n_bins=10)

        # Observable 0 should have generation as dominant source
        assert budget.generation_fraction[0] > 0.3, (
            f"Expected gen fraction > 0.3 for gen-dominated obs, got {budget.generation_fraction[0]:.3f}"
        )

    def test_seed_dominated_observable(self):
        from uq.workflow import compute_variance_budget

        ts, meta = _make_synthetic_timeseries(gen_effect=0.1, seed_effect=10.0, noise_std=0.1)
        obs_names = ["gen_dom", "seed_dom", "theta_dom"]
        budget = compute_variance_budget(ts, meta, obs_names, n_bins=10)

        # Observable 1 should have seed as significant source
        assert budget.seed_fraction[1] > 0.3, (
            f"Expected seed fraction > 0.3 for seed-dominated obs, got {budget.seed_fraction[1]:.3f}"
        )

    def test_all_variances_non_negative(self):
        from uq.workflow import compute_variance_budget

        ts, meta = _make_synthetic_timeseries()
        obs_names = ["a", "b", "c"]
        budget = compute_variance_budget(ts, meta, obs_names, n_bins=10)

        assert np.all(budget.total_variance >= 0)
        assert np.all(budget.generation_variance >= 0)
        assert np.all(budget.seed_variance >= 0)
        assert np.all(budget.growth_stage_variance >= 0)
        assert np.all(budget.residual_variance >= 0)

    def test_no_metadata(self):
        """Budget still works with missing generation/seed metadata."""
        from uq.workflow import compute_variance_budget

        rng = np.random.RandomState(42)
        ts = rng.randn(100, 2)
        # First col must be positive for growth fraction (log mass)
        ts[:, 0] = np.linspace(1, 2, 100)
        budget = compute_variance_budget(ts, {}, ["mass", "growth"], n_bins=5)

        assert budget.n_generations == 0
        assert budget.n_seeds == 0
        assert np.all(budget.generation_variance == 0)
        assert np.all(budget.seed_variance == 0)


class TestVarianceBudgetExport:
    def test_export_json(self, tmp_path: Path):
        from uq.workflow import compute_variance_budget

        ts, meta = _make_synthetic_timeseries(n_gens=2, n_seeds=2, n_steps_per_cell=20)
        obs_names = ["obs_a", "obs_b", "obs_c"]
        budget = compute_variance_budget(ts, meta, obs_names, n_bins=5)
        path = budget.export(tmp_path)

        assert path.exists()
        import json
        data = json.loads(path.read_text())
        assert data["mode"] == "baseline"
        assert data["n_generations"] == 2
        assert data["n_seeds"] == 2
        assert data["n_stages"] == 5
        assert set(data["per_observable"].keys()) == {"obs_a", "obs_b", "obs_c"}
        for obs_data in data["per_observable"].values():
            assert "total_variance" in obs_data
            assert "generation_fraction" in obs_data
            assert "seed_fraction" in obs_data
            assert "growth_stage_fraction" in obs_data
            assert "residual_fraction" in obs_data


class TestBaselineHtmlReport:
    def test_generates_html(self, tmp_path: Path):
        from uq.report import generate_baseline_html_report
        from uq.workflow import compute_variance_budget

        ts, meta = _make_synthetic_timeseries(n_gens=3, n_seeds=2, n_steps_per_cell=30)
        obs_names = ["dry_mass", "cell_mass", "growth"]
        budget = compute_variance_budget(ts, meta, obs_names, n_bins=5)
        budget.export(tmp_path)

        rpt = generate_baseline_html_report(tmp_path)
        assert rpt.exists()
        html_content = rpt.read_text()
        assert "Baseline Variance Report" in html_content
        assert "Variance Budget" in html_content
        assert "σ²_gen" in html_content or "Generation" in html_content
        assert "dry_mass" in html_content or "mass" in html_content

    def test_missing_budget_file(self, tmp_path: Path):
        from uq.report import generate_baseline_html_report

        with pytest.raises(FileNotFoundError):
            generate_baseline_html_report(tmp_path)
