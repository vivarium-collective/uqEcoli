"""Tests for output-side PCA reduction in the UQ workflow.

Covers:
  - apply_output_pca() with synthetic high-dim Y
  - PCAReduction dataclass (top_loadings, export)
  - Integration: PCA → PCE → Sobol on synthetic data
"""

from pathlib import Path

import numpy as np
import pytest


def _make_synthetic_high_dim_data(
    n_samples: int = 50,
    n_obs: int = 100,
    n_latent: int = 3,
    rng_seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Generate Y = X @ W + noise with known latent structure.

    Returns (X, Y, obs_names) where X is (n_samples, 5) and
    Y is (n_samples, n_obs).
    """
    rng = np.random.RandomState(rng_seed)
    n_params = 5
    X = rng.randn(n_samples, n_params)

    # Latent factors driven by first 3 params
    Z = X[:, :n_latent]  # (n_samples, 3)

    # Random mixing matrix: latent → observables
    W = rng.randn(n_latent, n_obs) * 2
    Y = Z @ W + rng.randn(n_samples, n_obs) * 0.5

    obs_names = [f"gene_{i}" for i in range(n_obs)]
    return X, Y, obs_names


class TestApplyOutputPCA:
    def test_basic_reduction(self):
        from uq.workflow import apply_output_pca

        _, Y, names = _make_synthetic_high_dim_data(n_samples=50, n_obs=100)
        Y_pca, pca = apply_output_pca(Y, n_components=5, observable_names=names)

        assert Y_pca.shape == (50, 5)
        assert pca.n_components == 5
        assert pca.mean.shape == (100,)
        assert pca.components.shape == (5, 100)
        assert pca.explained_variance_ratio.shape == (5,)
        assert len(pca.original_names) == 100

    def test_variance_explained_sums_correctly(self):
        from uq.workflow import apply_output_pca

        _, Y, names = _make_synthetic_high_dim_data()
        _, pca = apply_output_pca(Y, n_components=10, observable_names=names)

        # Sum of explained variance should be <= 1
        assert pca.explained_variance_ratio.sum() <= 1.0 + 1e-6
        # All ratios should be non-negative
        assert np.all(pca.explained_variance_ratio >= 0)
        # Should be in descending order
        assert np.all(np.diff(pca.explained_variance_ratio) <= 1e-10)

    def test_caps_at_n_samples(self):
        from uq.workflow import apply_output_pca

        _, Y, names = _make_synthetic_high_dim_data(n_samples=20, n_obs=100)
        _, pca = apply_output_pca(Y, n_components=50, observable_names=names)

        # Can't have more components than samples
        assert pca.n_components <= 20

    def test_latent_structure_captured(self):
        """With 3 latent factors, first 3 PCs should capture most variance."""
        from uq.workflow import apply_output_pca

        _, Y, names = _make_synthetic_high_dim_data(n_latent=3, n_obs=100)
        _, pca = apply_output_pca(Y, n_components=5, observable_names=names)

        # First 3 PCs should explain > 80% of variance
        top3 = pca.explained_variance_ratio[:3].sum()
        assert top3 > 0.8, f"Expected top-3 PCs to explain >80%, got {top3:.1%}"

    def test_top_loadings(self):
        from uq.workflow import apply_output_pca

        _, Y, names = _make_synthetic_high_dim_data()
        _, pca = apply_output_pca(Y, n_components=3, observable_names=names)

        top = pca.top_loadings(0, n=5)
        assert len(top) == 5
        assert all(isinstance(name, str) for name, _ in top)
        assert all(isinstance(val, float) for _, val in top)
        # Should be sorted by absolute value (descending)
        abs_vals = [abs(v) for _, v in top]
        assert abs_vals == sorted(abs_vals, reverse=True)


class TestPCAReductionExport:
    def test_export_creates_files(self, tmp_path: Path):
        from uq.workflow import apply_output_pca

        _, Y, names = _make_synthetic_high_dim_data(n_obs=50)
        _, pca = apply_output_pca(Y, n_components=3, observable_names=names)
        pca_dir = pca.export(tmp_path)

        # Spec-compliant names (B2)
        assert (pca_dir / "pca_loadings.npy").exists()
        assert (pca_dir / "pca_explained_variance.npy").exists()
        assert (pca_dir / "pca_top_loadings.json").exists()
        # Backward-compatible aliases
        assert (pca_dir / "components.npy").exists()
        assert (pca_dir / "mean.npy").exists()
        assert (pca_dir / "explained_variance_ratio.npy").exists()
        assert (pca_dir / "pca_summary.json").exists()

        import json

        summary = json.loads((pca_dir / "pca_top_loadings.json").read_text())
        assert "PC1" in summary
        assert "explained_variance_pct" in summary["PC1"]
        assert "top_loadings" in summary["PC1"]


class TestPCAtoPCESobolIntegration:
    """Integration: PCA → PCE → Sobol on synthetic data."""

    @pytest.mark.unit
    def test_pca_feeds_into_run_uqpc(self):
        """PCA-reduced Y should work as input to run_uqpc with sensible Sobol results."""
        from libuq.inputs import XSpaceVecoli
        from libuq.pipeline.models import SimDataParameter
        from uq.workflow import apply_output_pca, run_uqpc

        X, Y, obs_names = _make_synthetic_high_dim_data(
            n_samples=80,
            n_obs=50,
            n_latent=3,
        )
        X_scaled = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))

        n_params = X.shape[1]
        params = [SimDataParameter(name=f"p{i}", attr_path=f"dummy.p{i}", bounds=(0.0, 1.0)) for i in range(n_params)]
        param_space = XSpaceVecoli(
            parameter_names=[],
            parameter_bounds=[],
            parameter_types=[],
            experiment_id="test",
            parameters=params,
        )

        Y_pca, _pca = apply_output_pca(Y, n_components=5, observable_names=obs_names)

        result = run_uqpc(
            param_space=param_space,
            Y_train=Y_pca,
            X_train=X_scaled,
            polynomial_order=2,
            regression="lsq",
            n_test=20,
            seed=42,
        )

        assert result.sobol is not None
        assert result.surrogate is not None
        assert result.sobol.first_order.shape == (5,)
        assert result.sobol.total_order.shape == (5,)

        assert np.all(result.sobol.total_order >= 0)
        assert np.all(result.sobol.total_order <= 1.0 + 1e-6)

    @pytest.mark.unit
    def test_sobol_on_pcs_identifies_driving_params(self):
        """When Y is driven by subset of params, Sobol on PCs should reflect that."""
        from libuq.inputs import XSpaceVecoli
        from libuq.pipeline.models import SimDataParameter
        from uq.workflow import apply_output_pca, run_uqpc

        rng = np.random.RandomState(123)
        n_samples = 100
        n_params = 4
        n_obs = 30

        X = rng.uniform(0, 1, (n_samples, n_params))
        Y = (
            X[:, [0]] * rng.randn(1, n_obs) * 3
            + X[:, [1]] * rng.randn(1, n_obs) * 2
            + rng.randn(n_samples, n_obs) * 0.3
        )
        obs_names = [f"obs_{i}" for i in range(n_obs)]

        params = [SimDataParameter(name=f"p{i}", attr_path=f"dummy.p{i}", bounds=(0.0, 1.0)) for i in range(n_params)]
        param_space = XSpaceVecoli(
            parameter_names=[],
            parameter_bounds=[],
            parameter_types=[],
            experiment_id="test",
            parameters=params,
        )

        Y_pca, _ = apply_output_pca(Y, n_components=3, observable_names=obs_names)

        result = run_uqpc(
            param_space=param_space,
            Y_train=Y_pca,
            X_train=X,
            polynomial_order=2,
            regression="lsq",
            seed=42,
        )

        s_tot = result.sobol.total_order
        assert s_tot[0] > s_tot[2] or s_tot[1] > s_tot[2], f"Expected params 0 or 1 to dominate PC1 Sobol, got {s_tot}"
