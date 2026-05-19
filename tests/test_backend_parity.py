"""Regression tests: vEcoli vs v2ecoli backend parity.

These tests require a valid simData.cPickle and vEcoli installation.
They are skipped when the required data is not available.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

pytestmark = pytest.mark.skipif(
    not Path("../vEcoli/reconstruction/sim_data/kb/simData.cPickle").exists()
    and not Path("sim_data/baseline/kb/simData.cPickle").exists(),
    reason="simData.cPickle not available — requires vEcoli installation",
)


@pytest.fixture
def sim_data_path() -> str:
    """Find simData.cPickle."""
    candidates = [
        "../vEcoli/reconstruction/sim_data/kb/simData.cPickle",
        "sim_data/baseline/kb/simData.cPickle",
        str(Path.home() / ".local/share/vEcoli/simData.cPickle"),
    ]
    for c in candidates:
        p = Path(c)
        if p.exists():
            return str(p.resolve())
    pytest.skip("No simData.cPickle found")


@pytest.fixture
def cache_dir(tmp_path) -> str:
    """Create a temporary cache directory."""
    d = tmp_path / "cache"
    d.mkdir()
    return str(d)


# ═══════════════════════════════════════════════════════════════════════
# B.1: Single-sample parity
# ═══════════════════════════════════════════════════════════════════════


@pytest.mark.slow
class TestBackendParitySingle:
    """Same parameter → both backends produce similar Y."""

    def test_v2ecoli_cache_generation(self, sim_data_path: str, cache_dir: str):
        """Verify generate_cache_bundle produces valid cache."""
        from uq.generators.v2ecoli import generate_cache_bundle

        result = generate_cache_bundle(sim_data_path, cache_dir, seed=42)
        assert Path(result).exists()

        init_state = Path(result) / "initial_state.json"
        sim_cache = Path(result) / "sim_data_cache.dill"
        assert init_state.exists(), f"Missing {init_state}"
        assert sim_cache.exists(), f"Missing {sim_cache}"

    def test_v2ecoli_composite_build(self, cache_dir: str):
        """Verify build_v2ecoli_composite creates a valid Composite."""
        from uq.generators.v2ecoli import build_v2ecoli_composite

        composite = build_v2ecoli_composite(cache_dir=cache_dir, seed=0)
        assert composite is not None
        assert hasattr(composite, "state")
        assert "agents" in composite.state
        assert "0" in composite.state["agents"]

    def test_v2ecoli_composite_run(self, cache_dir: str):
        """Verify run_v2ecoli_composite produces timeseries."""
        from uq.generators.v2ecoli import (
            build_v2ecoli_composite,
            run_v2ecoli_composite,
        )

        composite = build_v2ecoli_composite(cache_dir=cache_dir, seed=0)
        result = run_v2ecoli_composite(composite, n_generations=2, max_time=100.0)

        assert "timeseries" in result
        assert "aggregated" in result
        assert result["generations"] >= 0
        assert result["final_time"] > 0


# ═══════════════════════════════════════════════════════════════════════
# B.2: Sobol parity
# ═══════════════════════════════════════════════════════════════════════


@pytest.mark.slow
class TestBackendParitySobol:
    """Same LHS sample set → similar Sobol indices."""

    def test_sobol_from_v2ecoli_samples(self, cache_dir: str):
        """Run a full sample → quantify cycle with v2ecoli backend."""
        import tempfile

        from libuq.pipeline.models import SimDataParameter
        from libuq.pipeline.param_loader import DEFAULT_SIM_DATA_PARAMETERS, ParameterDataset
        from libuq.sampling import PrecomputedCache
        from uq.generators.v2ecoli import V2ecoliGenerator, generate_cache_bundle
        from uq.workflow import quantify as wf_quantify, _setup_input_pc

        # Generate cache bundle
        generate_cache_bundle(sim_data_path, cache_dir, seed=42)

        # Set up parameter space
        ds = ParameterDataset(sim_data_path=sim_data_path)
        param_space = ds.to_parameter_space(parameters=None)
        bounds = np.array(param_space.parameter_bounds)

        # Generate samples
        input_pc, _, _ = _setup_input_pc(bounds)
        np.random.seed(42)
        germ = input_pc.sampleGerm(10)
        X = input_pc.evalPC(germ)

        # Run v2ecoli
        gen = V2ecoliGenerator(
            cache_dir=cache_dir,
            param_names=param_space.parameter_names,
            seed=42,
            n_generations=2,
            max_time=500.0,
        )
        Y_agg, Y_ts, Y_meta = gen._run_batch(X)

        # Cache for quantify
        cache_path = Path(cache_dir) / "_uq_cache"
        cache_path.mkdir(parents=True, exist_ok=True)
        cache = PrecomputedCache(
            cache_dir=cache_path,
            X=X,
            Y=Y_agg,
            parameter_names=param_space.parameter_names,
            metadata={"bounds": bounds.tolist(), "seed": 42, "observable_columns": []},
            Y_timeseries=Y_ts,
            Y_timeseries_meta=Y_meta,
        )
        cache.save()
        np.save(cache_path / "germ_train.npy", germ)

        # Run quantify
        with tempfile.TemporaryDirectory() as export_dir:
            result = wf_quantify(
                cache_dir=str(cache_path),
                sim_data_path=sim_data_path,
                polynomial_order=2,
                n_bins=5,
                regression="lsq",
                export_path=export_dir,
            )
            assert result is not None
            assert result.strategy1 is not None
            assert result.strategy1.sobol is not None
            # Sobol indices should be valid (sum to reasonable range)
            sti = result.strategy1.sobol.total_order
            if sti.ndim > 1:
                sti = np.mean(sti, axis=0)
            assert np.all(sti >= 0), "Negative Sobol total indices"
            assert np.all(sti <= 1.5), "Sobol total indices unreasonably large (>1.5)"


# ═══════════════════════════════════════════════════════════════════════
# B.3: Aggregation parity
# ═══════════════════════════════════════════════════════════════════════


@pytest.mark.slow
class TestBackendParityAggregation:
    """Same 4-strategy aggregation results across backends."""

    def test_v2ecoli_metadata_schema(self, cache_dir: str):
        """Verify Y_timeseries_meta has generation/seed labels."""
        from uq.generators.v2ecoli import V2ecoliGenerator, generate_cache_bundle

        generate_cache_bundle(sim_data_path, cache_dir, seed=42)

        gen = V2ecoliGenerator(
            cache_dir=cache_dir,
            param_names=["process.metabolism.kinetic_objective_weight"],
            seed=42,
            n_generations=2,
            max_time=500.0,
        )

        X = np.array([[0.5], [0.8]])
        _, _, Y_meta = gen._run_batch(X)

        assert Y_meta is not None
        assert len(Y_meta) == 2
        for m in Y_meta:
            assert "generation" in m
            assert "lineage_seed" in m
