"""Tests for the BYO-variants escape hatch.

Local mode only (--backend vecoli). Two layers:

1. `_build_config` honors a user-supplied `variants` block instead of
   overwriting it with the auto-generated sim_data_setattr section.
2. `_x_from_variants` reverse-maps mutation values back into the X matrix
   so `quantify` regresses Y against perturbations vEcoli actually applied.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from libuq.pipeline.models import SimDataParameter
from uq.vecoli_config import (
    _build_config,
    _build_variants_from_samples,
    _check_oob_variants,
    _design_matrix_conditioning,
    _germ_from_physical,
    _load_variants_from_base_config,
    _pce_basis_size,
    _pce_sample_adequacy,
    _x_from_variants,
)


def _auto_variants() -> dict:
    """The shape `_build_variants_from_samples` produces for two scalar samples."""
    return {
        "sim_data_setattr": {
            "mutations": {
                "value": [
                    {"process.metabolism.kinetic_objective_weight": 1.0e-7},
                    {"process.metabolism.kinetic_objective_weight": 2.0e-7},
                ]
            }
        }
    }


def test_no_base_config_uses_generated_variants(tmp_path: Path) -> None:
    cfg = _build_config(
        sim_data_path="/tmp/simData.cPickle",
        output_dir=str(tmp_path),
        variants_section=_auto_variants(),
        n_init_sims=1,
        generations=1,
    )
    assert "variants" in cfg
    assert "sim_data_setattr" in cfg["variants"]
    values = cfg["variants"]["sim_data_setattr"]["mutations"]["value"]
    assert len(values) == 2


def test_base_config_without_variants_gets_generated(tmp_path: Path) -> None:
    base = {"some_other_key": 42, "analysis_options": {"foo": "bar"}}
    base_path = tmp_path / "base.json"
    base_path.write_text(json.dumps(base))

    cfg = _build_config(
        sim_data_path="/tmp/simData.cPickle",
        output_dir=str(tmp_path),
        variants_section=_auto_variants(),
        n_init_sims=1,
        generations=1,
        base_config_path=str(base_path),
    )
    # Base preserved
    assert cfg["some_other_key"] == 42
    assert cfg["analysis_options"] == {"foo": "bar"}
    # Variants generated since base had none
    assert "sim_data_setattr" in cfg["variants"]
    assert len(cfg["variants"]["sim_data_setattr"]["mutations"]["value"]) == 2


def test_base_config_variants_block_is_preserved(tmp_path: Path) -> None:
    """User-supplied variants block survives — the BYO escape hatch."""
    user_variants = {
        "flux_kinetics": {
            "kcat_scale": {"value": [0.5, 1.5, 2.5]},
        }
    }
    base = {"variants": user_variants, "analysis_options": {"foo": "bar"}}
    base_path = tmp_path / "base.json"
    base_path.write_text(json.dumps(base))

    cfg = _build_config(
        sim_data_path="/tmp/simData.cPickle",
        output_dir=str(tmp_path),
        variants_section=_auto_variants(),
        n_init_sims=1,
        generations=1,
        base_config_path=str(base_path),
    )
    # User variants survived verbatim
    assert cfg["variants"] == user_variants
    # Auto-generated sim_data_setattr block was NOT injected
    assert "sim_data_setattr" not in cfg["variants"]
    # UQ-owned fields still overrode the base
    assert cfg["sim_data_path"] == "/tmp/simData.cPickle"
    assert cfg["emitter"] == "parquet"


def test_base_config_variants_block_works_with_any_module(tmp_path: Path) -> None:
    """Any variant module (not just sim_data_setattr) is honored."""
    user_variants = {
        "condition": {"environment": {"value": ["minimal", "rich"]}},
    }
    base = {"variants": user_variants}
    base_path = tmp_path / "base.json"
    base_path.write_text(json.dumps(base))

    cfg = _build_config(
        sim_data_path="/tmp/simData.cPickle",
        output_dir=str(tmp_path),
        variants_section=_auto_variants(),
        n_init_sims=1,
        generations=1,
        base_config_path=str(base_path),
    )
    assert cfg["variants"] == user_variants


# ── 25-B reverse-mapping ──────────────────────────────────────────────


def _specs() -> list[SimDataParameter]:
    return [
        SimDataParameter(
            name="kow", attr_path="process.metabolism.kinetic_objective_weight",
            bounds=(1e-8, 1e-6),
        ),
        SimDataParameter(
            name="frar", attr_path="process.transcription.fraction_active_rnap_free",
            bounds=(0.25, 0.47),
        ),
    ]


def test_x_from_variants_roundtrip_scalar() -> None:
    specs = _specs()
    X = np.array([
        [5.0e-7, 0.36],
        [1.0e-7, 0.30],
        [8.0e-7, 0.45],
    ])
    variants = _build_variants_from_samples(X, specs)
    X_recovered = _x_from_variants(variants, specs)
    np.testing.assert_allclose(X_recovered, X)


def test_x_from_variants_roundtrip_indexed() -> None:
    """Indexed array mutations (`__index__`/`__value__`) round-trip.

    Note: the encoder overwrites when two specs share an attr_path with
    different indices (one entry per attr_path in the dict), so each
    indexed spec must point at a different array.
    """
    specs = [
        SimDataParameter(name="kcat", attr_path="process.metabolism.kcats",
                         bounds=(1.0, 10.0), index=3),
        SimDataParameter(name="km", attr_path="process.metabolism.kms",
                         bounds=(0.01, 0.1), index=2),
    ]
    X = np.array([[3.5, 0.04], [7.2, 0.08]])
    variants = _build_variants_from_samples(X, specs)
    first = variants["sim_data_setattr"]["mutations"]["value"][0]
    assert first["process.metabolism.kcats"] == {"__index__": 3, "__value__": 3.5}
    assert first["process.metabolism.kms"] == {"__index__": 2, "__value__": 0.04}
    X_recovered = _x_from_variants(variants, specs)
    np.testing.assert_allclose(X_recovered, X)


def test_x_from_variants_rejects_non_sim_data_setattr() -> None:
    bogus = {"flux_kinetics": {"kcat_scale": {"value": [0.5, 1.5]}}}
    with pytest.raises(ValueError, match="sim_data_setattr"):
        _x_from_variants(bogus, _specs())


def test_x_from_variants_rejects_missing_attr_path() -> None:
    """Every spec must appear in every mutation entry."""
    variants = {
        "sim_data_setattr": {
            "mutations": {
                "value": [
                    {"process.metabolism.kinetic_objective_weight": 5e-7},
                    # Missing the `frar` attr_path
                ]
            }
        }
    }
    with pytest.raises(ValueError, match="missing attr_path"):
        _x_from_variants(variants, _specs())


def test_germ_from_physical_is_evalpc_inverse() -> None:
    """For points inside the box, germ values land in [-1, 1]."""
    bounds = np.array([[0.0, 1.0], [10.0, 20.0]])
    X = np.array([[0.0, 10.0], [0.5, 15.0], [1.0, 20.0]])
    germ = _germ_from_physical(X, bounds)
    np.testing.assert_allclose(germ, [[-1.0, -1.0], [0.0, 0.0], [1.0, 1.0]])


def test_germ_from_physical_marks_out_of_bounds() -> None:
    """Values outside the box produce germ outside [-1, 1] (no clipping)."""
    bounds = np.array([[0.0, 1.0]])
    X = np.array([[1.5]])  # 50% past upper bound
    germ = _germ_from_physical(X, bounds)
    assert germ[0, 0] > 1.0


def test_load_variants_from_base_config_extracts_block(tmp_path: Path) -> None:
    base = {
        "variants": {"sim_data_setattr": {"mutations": {"value": [{"a.b": 1.0}]}}},
        "other_key": "preserved",
    }
    base_path = tmp_path / "base.json"
    base_path.write_text(json.dumps(base))
    block = _load_variants_from_base_config(str(base_path))
    assert "sim_data_setattr" in block


def test_load_variants_from_base_config_rejects_missing_block(tmp_path: Path) -> None:
    base_path = tmp_path / "base.json"
    base_path.write_text(json.dumps({"parca_variants": [{"x": 1}]}))
    with pytest.raises(ValueError, match="no 'variants' key"):
        _load_variants_from_base_config(str(base_path))


# ── OOB rejection (triage #1: PyTUQ pdom invariant) ──────────────────


def test_check_oob_variants_empty_when_all_in_bounds() -> None:
    bounds = np.array([[0.0, 1.0], [10.0, 20.0]])
    X = np.array([[0.5, 15.0], [0.1, 11.0], [1.0, 20.0]])  # boundary inclusive
    assert _check_oob_variants(X, bounds, ["a", "b"]) == []


def test_check_oob_variants_finds_violations() -> None:
    bounds = np.array([[0.0, 1.0], [10.0, 20.0]])
    X = np.array([
        [0.5, 15.0],    # ok
        [1.5, 11.0],    # variant 1, dim 0 over
        [0.2, 25.0],    # variant 2, dim 1 over
        [-0.1, 5.0],    # variant 3, both under
    ])
    violations = _check_oob_variants(X, bounds, ["a", "b"])
    assert len(violations) == 4
    # variant 1, dim 0
    assert violations[0] == (1, "a", 1.5, 0.0, 1.0)
    # variant 2, dim 1
    assert violations[1] == (2, "b", 25.0, 10.0, 20.0)
    # variant 3, both dims
    assert violations[2] == (3, "a", -0.1, 0.0, 1.0)
    assert violations[3] == (3, "b", 5.0, 10.0, 20.0)


def test_check_oob_variants_ignores_param_names_length_in_iteration() -> None:
    """Helper iterates over X.shape[1] columns; param_names supplies labels."""
    bounds = np.array([[0.0, 1.0]])
    X = np.array([[0.5], [2.0]])
    out = _check_oob_variants(X, bounds, ["kicker"])
    assert out == [(1, "kicker", 2.0, 0.0, 1.0)]


# ── PCE sample-size adequacy diagnostic ──────────────────────────────


def test_pce_basis_size_matches_total_order_formula() -> None:
    # Known values: C(p+d, d)
    assert _pce_basis_size(order=1, n_params=6) == 7   # C(7,6)
    assert _pce_basis_size(order=2, n_params=6) == 28  # C(8,6)
    assert _pce_basis_size(order=3, n_params=6) == 84  # C(9,6)
    assert _pce_basis_size(order=2, n_params=2) == 6   # C(4,2)


def test_pce_sample_adequacy_underdetermined_fires_below_basis() -> None:
    # order 2, 6 params → basis 28; 12 samples is < basis
    status, msg = _pce_sample_adequacy(n_samples=12, n_params=6, order=2)
    assert status == "underdetermined"
    assert "ill-posed" in msg


def test_pce_sample_adequacy_marginal_fires_between_1x_and_2x() -> None:
    # 40 samples, basis 28 → ratio 1.43×
    status, msg = _pce_sample_adequacy(n_samples=40, n_params=6, order=2)
    assert status == "marginal"
    assert "over-fit" in msg


def test_pce_sample_adequacy_ok_above_2x() -> None:
    # 60 samples, basis 28 → ratio 2.14×
    status, _ = _pce_sample_adequacy(n_samples=60, n_params=6, order=2)
    assert status == "ok"

    # 100 samples comfortable
    status, _ = _pce_sample_adequacy(n_samples=100, n_params=6, order=2)
    assert status == "ok"


def test_pce_sample_adequacy_scales_with_polynomial_order() -> None:
    """Lower order → smaller basis → same n_samples can shift status up."""
    # 12 samples, 6 params, order=2: basis=28 → underdetermined
    status_p2, _ = _pce_sample_adequacy(n_samples=12, n_params=6, order=2)
    assert status_p2 == "underdetermined"

    # Same 12 samples, order=1: basis=7, ratio=1.71 → marginal (up from
    # underdetermined). Confirms dropping order helps.
    status_p1, _ = _pce_sample_adequacy(n_samples=12, n_params=6, order=1)
    assert status_p1 == "marginal"

    # 20 samples, order=1: basis=7, ratio=2.86 → ok
    status_ok, _ = _pce_sample_adequacy(n_samples=20, n_params=6, order=1)
    assert status_ok == "ok"


# ── Design matrix conditioning κ(A) (triage #2: PyTUQ-native) ────────


def _uniform_germ(n: int, d: int, seed: int = 42) -> np.ndarray:
    """Draw n samples uniformly from [-1, 1]^d via numpy (matches sampleGerm)."""
    rng = np.random.default_rng(seed)
    return rng.uniform(-1.0, 1.0, size=(n, d))


def test_design_matrix_kappa_well_conditioned_for_ample_uniform_samples() -> None:
    # 100 uniform samples in 4 dims at order 2: basis size = C(6,4) = 15
    # ratio 100/15 ≈ 6.7×, geometry uniform → κ should be small.
    germ = _uniform_germ(n=100, d=4, seed=0)
    kappa, status, _ = _design_matrix_conditioning(germ, n_params=4, order=2)
    assert status == "well-conditioned", f"got κ={kappa}, status={status}"
    assert kappa < 100.0


def test_design_matrix_kappa_singular_when_underdetermined() -> None:
    # 5 samples in 4 dims at order 2: basis size = 15 > 5 → A is wide →
    # underdetermined → cond(A) is large (effectively ∞ for our purposes).
    germ = _uniform_germ(n=5, d=4, seed=0)
    kappa, status, _ = _design_matrix_conditioning(germ, n_params=4, order=2)
    # np.linalg.cond on a wide matrix is well-defined but huge — accept
    # either explicit non-finite or just very large.
    assert status in ("singular", "ill-conditioned"), f"got κ={kappa}, status={status}"


def test_design_matrix_kappa_ill_conditioned_for_clumped_samples() -> None:
    """Samples clumped in one corner produce near-collinear basis evals."""
    rng = np.random.default_rng(0)
    # Cluster all samples in a tiny corner — basis can barely distinguish them.
    germ = -0.99 + 0.005 * rng.uniform(0.0, 1.0, size=(50, 3))
    kappa, status, _ = _design_matrix_conditioning(germ, n_params=3, order=2)
    assert status in ("ill-conditioned", "marginal"), f"got κ={kappa}, status={status}"
    assert kappa > 100.0


def test_design_matrix_kappa_returns_kappa_and_advisory() -> None:
    """Shape contract: tuple of (float, status_string, advisory_string)."""
    germ = _uniform_germ(n=50, d=2, seed=0)
    kappa, status, advisory = _design_matrix_conditioning(germ, n_params=2, order=2)
    assert isinstance(kappa, float)
    assert status in ("well-conditioned", "marginal", "ill-conditioned", "singular")
    assert isinstance(advisory, str) and len(advisory) > 0


def test_design_matrix_kappa_grows_with_polynomial_order() -> None:
    """Higher order → larger basis → higher κ for the same sample set."""
    germ = _uniform_germ(n=30, d=4, seed=0)
    kappa_p1, _, _ = _design_matrix_conditioning(germ, n_params=4, order=1)
    kappa_p3, _, _ = _design_matrix_conditioning(germ, n_params=4, order=3)
    assert kappa_p3 > kappa_p1, f"p=3 κ={kappa_p3} should exceed p=1 κ={kappa_p1}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
