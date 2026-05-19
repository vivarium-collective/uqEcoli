"""Tests for the v2ecoli generator module (Phases 1-3)."""

from __future__ import annotations

import copy
import pickle
import tempfile
from pathlib import Path

import numpy as np
import pytest
import typer

from uq.generators.v2ecoli import (
    V2ecoliGenerator,
    _apply_mutation_to_configs,
    _apply_mutations_to_bundle,
    extract_timeseries,
)
from uq.v2ecoli_bridge import (
    OBSERVABLE_PATHS,
    ARRAY_OBSERVABLE_PATHS,
    get_state_path,
    get_observable_from_state,
    get_array_observable_from_state,
    is_array_preset,
    is_scalar_preset,
    observable_names_for_presets,
)


# ═══════════════════════════════════════════════════════════════════════
# Unit tests: mutation logic
# ═══════════════════════════════════════════════════════════════════════


class TestApplyMutations:
    """_apply_mutation_to_configs maps dot-paths to config keys."""

    def test_metabolism_mutation(self):
        configs = {
            "ecoli-metabolism": {
                "kinetic_objective_weight": 0.5,
                "secretion_penalty_coeff": 0.001,
            }
        }
        _apply_mutation_to_configs(
            configs, "process.metabolism.kinetic_objective_weight", 0.8
        )
        assert configs["ecoli-metabolism"]["kinetic_objective_weight"] == 0.8

    def test_transcription_mutation(self):
        configs = {
            "ecoli-transcript-initiation": {
                "fraction_active_rnap_free": 0.2,
                "fraction_active_rnap_bound": 0.8,
            }
        }
        _apply_mutation_to_configs(
            configs, "process.transcription.fraction_active_rnap_free", 0.35
        )
        assert configs["ecoli-transcript-initiation"]["fraction_active_rnap_free"] == 0.35

    def test_translation_mutation(self):
        configs = {
            "ecoli-polypeptide-elongation": {
                "basal_elongation_rate": 15,
            }
        }
        _apply_mutation_to_configs(
            configs, "process.translation.basal_elongation_rate", 22
        )
        assert configs["ecoli-polypeptide-elongation"]["basal_elongation_rate"] == 22.0

    def test_mass_mutation(self):
        configs = {
            "ecoli-mass-listener": {
                "cell_dry_mass_fraction": 0.3,
            }
        }
        _apply_mutation_to_configs(configs, "mass.cell_dry_mass_fraction", 0.35)
        assert configs["ecoli-mass-listener"]["cell_dry_mass_fraction"] == 0.35

    def test_unknown_process_skipped(self):
        configs = {"ecoli-unknown": {"some_param": 1.0}}
        _apply_mutation_to_configs(
            configs, "process.unknown.some_param", 5.0
        )
        # Unknown processes are silently skipped
        assert configs["ecoli-unknown"]["some_param"] == 1.0


class TestApplyMutationsToBundle:
    """_apply_mutations_to_bundle deep-copies and mutates."""

    def test_preserves_original(self):
        bundle = {
            "configs": {
                "ecoli-metabolism": {"kinetic_objective_weight": 0.5},
            },
            "unique_names": ["full_chromosome"],
            "dry_mass_inc_dict": {},
        }
        mutated = _apply_mutations_to_bundle(bundle, {"process.metabolism.kinetic_objective_weight": 0.9})
        assert bundle["configs"]["ecoli-metabolism"]["kinetic_objective_weight"] == 0.5
        assert mutated["configs"]["ecoli-metabolism"]["kinetic_objective_weight"] == 0.9

    def test_preserves_other_keys(self):
        bundle = {
            "configs": {
                "ecoli-metabolism": {"kinetic_objective_weight": 0.5},
                "ecoli-transcript-initiation": {"fraction_active_rnap_free": 0.2},
            },
            "unique_names": ["full_chromosome"],
            "dry_mass_inc_dict": {},
        }
        mutated = _apply_mutations_to_bundle(bundle, {"process.metabolism.kinetic_objective_weight": 0.9})
        assert mutated["configs"]["ecoli-metabolism"]["kinetic_objective_weight"] == 0.9
        assert mutated["configs"]["ecoli-transcript-initiation"]["fraction_active_rnap_free"] == 0.2


# ═══════════════════════════════════════════════════════════════════════
# Unit tests: V2ecoliGenerator construction
# ═══════════════════════════════════════════════════════════════════════


class TestV2ecoliGeneratorInit:
    """V2ecoliGenerator construction and basic properties."""

    def test_constructor(self):
        gen = V2ecoliGenerator(
            cache_dir="/tmp/cache",
            param_names=["process.metabolism.kinetic_objective_weight"],
            seed=42,
        )
        assert gen.cache_dir == "/tmp/cache"
        assert len(gen.param_names) == 1
        assert gen.seed == 42
        assert gen.n_generations == 10
        assert gen.max_time == 10000.0

    def test_obs_names_raises_before_run(self):
        gen = V2ecoliGenerator(
            cache_dir="/tmp/cache",
            param_names=["process.metabolism.kinetic_objective_weight"],
        )
        with pytest.raises(RuntimeError, match="Observable names not yet known"):
            _ = gen.obs_names

    def test_custom_constructor_params(self):
        gen = V2ecoliGenerator(
            cache_dir="/tmp/cache",
            param_names=["a", "b"],
            seed=1,
            n_generations=5,
            max_time=5000.0,
            max_workers=4,
        )
        assert gen.n_generations == 5
        assert gen.max_time == 5000.0
        assert gen.max_workers == 4


# ═══════════════════════════════════════════════════════════════════════
# Unit tests: extract_timeseries
# ═══════════════════════════════════════════════════════════════════════


class TestExtractTimeseries:
    """extract_timeseries converts run results to numpy arrays."""

    def test_basic_extraction(self):
        result = {
            "timeseries": {
                "listeners__mass__dry_mass": [100.0, 110.0, 120.0],
                "listeners__mass__cell_mass": [200.0, 210.0, 220.0],
            },
            "aggregated": {
                "listeners__mass__dry_mass": 110.0,
                "listeners__mass__cell_mass": 210.0,
            },
            "metadata": {"generation_count": 1, "agent_ids": ["0"]},
        }
        ts = extract_timeseries(result)
        assert ts.shape == (3, 2)  # (n_steps, n_obs)
        # Columns are sorted alphabetically by observable name
        # "listeners__mass__cell_mass" < "listeners__mass__dry_mass"
        assert float(ts[0, 0]) == 200.0  # cell_mass at t=0
        assert float(ts[0, 1]) == 100.0  # dry_mass at t=0

    def test_with_subset_names(self):
        result = {
            "timeseries": {
                "a": [1.0, 2.0],
                "b": [3.0, 4.0],
                "c": [5.0, 6.0],
            },
            "aggregated": {"a": 1.5, "b": 3.5, "c": 5.5},
            "metadata": {"generation_count": 1, "agent_ids": ["0"]},
        }
        ts = extract_timeseries(result, observable_names=["a", "c"])
        assert ts.shape == (2, 2)
        assert ts[0, 0] == 1.0
        assert ts[0, 1] == 5.0

    def test_empty_timeseries(self):
        result = {
            "timeseries": {},
            "aggregated": {},
            "metadata": {"generation_count": 0, "agent_ids": []},
        }
        ts = extract_timeseries(result, observable_names=["x"])
        assert ts.shape == (1, 1)
        assert ts[0, 0] == 0.0


# ═══════════════════════════════════════════════════════════════════════
# Unit tests: bridge/observable path mapping
# ═══════════════════════════════════════════════════════════════════════


class TestBridgeObservables:
    """Observable path mapping from uq.v2ecoli_bridge."""

    def test_observable_paths_present(self):
        assert "listeners__mass__dry_mass" in OBSERVABLE_PATHS
        assert "listeners__mass__cell_mass" in OBSERVABLE_PATHS
        assert "listeners__mass__volume" in OBSERVABLE_PATHS

    def test_get_state_path(self):
        path = get_state_path("listeners__mass__dry_mass")
        assert path == ["listeners", "mass", "dry_mass"]

    def test_get_state_path_unknown(self):
        path = get_state_path("nonexistent")
        assert path is None

    def test_observable_names_for_presets_scalar(self):
        names = observable_names_for_presets(["mass"])
        assert "listeners__mass__dry_mass" in names
        assert "listeners__mass__cell_mass" in names

    def test_observable_names_for_presets_array(self):
        names = observable_names_for_presets(["transcriptome"])
        assert len(names) == 1
        assert names[0] == "__array__:transcriptome__"

    def test_observable_names_for_presets_mixed(self):
        names = observable_names_for_presets(["mass", "transcriptome", "proteome"])
        assert "listeners__mass__dry_mass" in names
        assert "__array__:transcriptome__" in names
        assert "__array__:proteome__" in names

    def test_observable_names_for_presets_unknown(self):
        names = observable_names_for_presets(["unknown_preset"])
        # Falls back to all observable paths
        assert len(names) >= 10


class TestArrayObservables:
    """Array observable paths and extraction."""

    def test_array_paths_defined(self):
        assert "transcriptome" in ARRAY_OBSERVABLE_PATHS
        assert "proteome" in ARRAY_OBSERVABLE_PATHS
        assert "fluxome" in ARRAY_OBSERVABLE_PATHS
        assert "exchange_fluxes" in ARRAY_OBSERVABLE_PATHS

    def test_is_array_preset(self):
        assert is_array_preset("transcriptome")
        assert not is_array_preset("mass")

    def test_is_scalar_preset(self):
        assert is_scalar_preset("mass")
        assert is_scalar_preset("higher_order")
        assert not is_scalar_preset("transcriptome")

    def test_get_array_observable_from_state_returns_array(self):
        cell_state = {
            "listeners": {
                "rna_counts": {
                    "mRNA_cistron_counts": [1.0, 2.0, 3.0, 4.0, 5.0],
                }
            }
        }
        arr = get_array_observable_from_state(cell_state, "transcriptome")
        assert arr is not None
        assert arr.shape == (5,)
        assert float(arr[0]) == 1.0
        assert float(arr[-1]) == 5.0

    def test_get_array_observable_from_state_missing_path(self):
        cell_state = {"listeners": {}}
        arr = get_array_observable_from_state(cell_state, "transcriptome")
        assert arr is None

    def test_get_array_observable_from_state_unknown_preset(self):
        cell_state = {}
        arr = get_array_observable_from_state(cell_state, "nonexistent")
        assert arr is None

    def test_get_array_observable_from_state_2d_ravels(self):
        cell_state = {
            "listeners": {
                "fba_results": {
                    "base_reaction_fluxes": [[1.0, 2.0], [3.0, 4.0]],
                }
            }
        }
        arr = get_array_observable_from_state(cell_state, "fluxome")
        assert arr is not None
        assert arr.shape == (4,)  # ravels 2D → 1D


# ═══════════════════════════════════════════════════════════════════════
# Import chain tests
# ═══════════════════════════════════════════════════════════════════════


class TestImportChain:
    """All v2ecoli integration modules import correctly."""

    def test_generators_import(self):
        from uq.generators import v2ecoli

        assert hasattr(v2ecoli, "V2ecoliGenerator")
        assert hasattr(v2ecoli, "generate_cache_bundle")
        assert hasattr(v2ecoli, "run_v2ecoli_composite")

    def test_bridge_import(self):
        from uq import v2ecoli_bridge

        assert hasattr(v2ecoli_bridge, "OBSERVABLE_PATHS")
        assert hasattr(v2ecoli_bridge, "ARRAY_OBSERVABLE_PATHS")
        assert hasattr(v2ecoli_bridge, "get_observable_from_state")
        assert hasattr(v2ecoli_bridge, "get_array_observable_from_state")
        assert hasattr(v2ecoli_bridge, "observable_names_for_presets")
        assert hasattr(v2ecoli_bridge, "is_array_preset")
        assert hasattr(v2ecoli_bridge, "is_scalar_preset")

    def test_cli_flag(self):
        """The CLI has the --backend parameter."""
        from uq.cli import app

        # Verify the sample command has backend param
        cmd = typer.main.get_command(app)
        sample_cmd = cmd.get_command(None, "sample")  # type: ignore[union-attr]
        assert sample_cmd is not None
        params = {p.name: p for p in sample_cmd.params}
        assert "backend" in params
        assert params["backend"].default == "vecoli"

    def test_pbg_uqEcoli_import(self):
        from pbg_uqEcoli import UQPipeline, pipeline_document
        from pbg_uqEcoli.core import build_core

        assert UQPipeline is not None
        assert callable(pipeline_document)
        assert callable(build_core)
