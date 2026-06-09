"""Tests for the design-config helpers in uq.vecoli_config.

End-to-end CLI integration is tested manually with a real simData.cPickle;
these unit tests cover the pure helpers using synthetic variant modules
installed via the same pattern as test_convert_variants.py.
"""

from __future__ import annotations

import sys
import types

import numpy as np
import pytest

from uq.vecoli_config import (
    _apply_design_to_baseline,
    _design_condition_id,
    _design_layer_attr_paths,
)


# ── Test doubles ──────────────────────────────────────────────────────


class FakeMetabolism:
    def __init__(self):
        self.objective_weight = 1e-7
        self.secretion_penalty = 0.001
        self.kcats = np.array([1.0, 2.0, 3.0, 4.0, 5.0])


class FakeSimData:
    def __init__(self):
        self.condition = "basal"
        self.dry_mass_fraction = 0.30
        self.elongation_rate = 22.0
        self.metabolism = FakeMetabolism()


def _install_variant(name: str, apply):
    full = f"ecoli.variants.{name}"
    module = types.ModuleType(full)
    module.apply_variant = apply  # type: ignore[attr-defined]
    sys.modules.setdefault("ecoli", types.ModuleType("ecoli"))
    sys.modules.setdefault("ecoli.variants", types.ModuleType("ecoli.variants"))
    sys.modules[full] = module
    return name


@pytest.fixture
def scalar_design_variant():
    """A design variant that perturbs metabolism.objective_weight."""
    def apply(sim_data, params):
        sim_data.metabolism.objective_weight = params["weight"]
        return sim_data
    name = _install_variant("scalar_design", apply)
    yield name
    sys.modules.pop(f"ecoli.variants.{name}", None)


@pytest.fixture
def categorical_design_variant():
    """A design variant that swaps a string field."""
    def apply(sim_data, params):
        sim_data.condition = params["condition"]
        return sim_data
    name = _install_variant("categorical_design", apply)
    yield name
    sys.modules.pop(f"ecoli.variants.{name}", None)


# ── _design_condition_id ──────────────────────────────────────────────


def test_design_condition_id_format() -> None:
    cid = _design_condition_id("mecillinam_timeline", 3, {"concentration": 0.01})
    assert cid.startswith("mecillinam_timeline_0003_")
    assert len(cid.split("_")[-1]) == 6  # short hash


def test_design_condition_id_deterministic() -> None:
    a = _design_condition_id("foo", 1, {"x": 1, "y": 2})
    b = _design_condition_id("foo", 1, {"y": 2, "x": 1})  # different key order
    assert a == b  # sorted before hashing


def test_design_condition_id_index_pads_to_four_digits() -> None:
    assert "_0000_" in _design_condition_id("m", 0, {})
    assert "_0042_" in _design_condition_id("m", 42, {})
    assert "_9999_" in _design_condition_id("m", 9999, {})


def test_design_condition_id_unique_for_different_params() -> None:
    a = _design_condition_id("m", 0, {"x": 1})
    b = _design_condition_id("m", 0, {"x": 2})
    assert a != b  # different params → different hash even at same index


# ── _apply_design_to_baseline ──────────────────────────────────────────


def test_apply_design_to_baseline_mutates_in_place(scalar_design_variant) -> None:
    sd = FakeSimData()
    result = _apply_design_to_baseline(
        sd, scalar_design_variant, {"weight": 5e-7},
    )
    # Same identity (apply mutates in place) and updated value
    assert result is sd
    assert sd.metabolism.objective_weight == 5e-7


def test_apply_design_to_baseline_handles_none_return(scalar_design_variant) -> None:
    """Some variants mutate in place + return None; helper should still return sim_data."""
    def apply_no_return(sim_data, params):
        sim_data.elongation_rate = params["rate"]
        # NB: no return
    sys.modules[f"ecoli.variants.{scalar_design_variant}"].apply_variant = apply_no_return

    sd = FakeSimData()
    result = _apply_design_to_baseline(
        sd, scalar_design_variant, {"rate": 21.0},
    )
    assert result is sd
    assert sd.elongation_rate == 21.0


# ── _design_layer_attr_paths ───────────────────────────────────────────


def test_design_layer_attr_paths_collects_changed_paths(scalar_design_variant) -> None:
    baseline = FakeSimData()
    entries = [{"weight": 3e-7}, {"weight": 5e-7}, {"weight": 1e-6}]
    paths = _design_layer_attr_paths(baseline, scalar_design_variant, entries)
    assert "metabolism.objective_weight" in paths


def test_design_layer_attr_paths_empty_when_no_change(scalar_design_variant) -> None:
    """Baseline weight = 1e-7; passing 1e-7 means no change for that entry."""
    baseline = FakeSimData()
    entries = [{"weight": 1e-7}]  # equals baseline
    paths = _design_layer_attr_paths(baseline, scalar_design_variant, entries)
    assert paths == set()


def test_design_layer_attr_paths_categorical(categorical_design_variant) -> None:
    """Categorical mutations show up as attr_paths just like scalar ones."""
    baseline = FakeSimData()
    entries = [{"condition": "with_aa"}, {"condition": "no_oxygen"}]
    paths = _design_layer_attr_paths(baseline, categorical_design_variant, entries)
    assert "condition" in paths


def test_design_layer_attr_paths_unions_across_entries(scalar_design_variant) -> None:
    """A heterogeneous design layer mutating different things per entry."""
    # First entry mutates weight; second mutates elongation_rate.
    def apply_heterogeneous(sim_data, params):
        if "weight" in params:
            sim_data.metabolism.objective_weight = params["weight"]
        if "rate" in params:
            sim_data.elongation_rate = params["rate"]
        return sim_data
    sys.modules[f"ecoli.variants.{scalar_design_variant}"].apply_variant = apply_heterogeneous

    baseline = FakeSimData()
    entries = [{"weight": 5e-7}, {"rate": 24.0}]
    paths = _design_layer_attr_paths(baseline, scalar_design_variant, entries)
    assert paths == {"metabolism.objective_weight", "elongation_rate"}


# ── Overlap-check semantics ────────────────────────────────────────────


def test_overlap_with_uq_params_file_detected(scalar_design_variant) -> None:
    """When --params-file lists a path the design layer also writes, the
    overlap check should surface it."""
    baseline = FakeSimData()
    design_paths = _design_layer_attr_paths(
        baseline, scalar_design_variant, [{"weight": 5e-7}],
    )
    uq_paths = {"metabolism.objective_weight", "elongation_rate"}
    overlap = design_paths & uq_paths
    assert overlap == {"metabolism.objective_weight"}


def test_no_overlap_when_paths_disjoint(scalar_design_variant) -> None:
    baseline = FakeSimData()
    design_paths = _design_layer_attr_paths(
        baseline, scalar_design_variant, [{"weight": 5e-7}],
    )
    uq_paths = {"elongation_rate", "dry_mass_fraction"}  # disjoint
    overlap = design_paths & uq_paths
    assert overlap == set()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
