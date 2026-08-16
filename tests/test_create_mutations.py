"""Tests for `uq create mutations` and ``uq.perturbation`` loader.

End-to-end CLI integration (against real ``simData.cPickle``) is exercised
manually; the helpers covered here are the pure pieces that determine
what params files get generated independent of vEcoli.
"""

from __future__ import annotations

import json
import sys
import textwrap
import types
from pathlib import Path

import pytest

from libuq.pipeline.models import SimDataParameter
from libuq.pipeline.param_loader import DEFAULT_SIM_DATA_PARAMETERS
from uq.perturbation import (
    _filter_kwargs,
    build_parameters_from_scheme,
    load_perturbation_scheme,
)


# ── load_perturbation_scheme ───────────────────────────────────────────


def test_load_none_returns_none() -> None:
    assert load_perturbation_scheme(None) is None


def test_load_from_file_path(tmp_path: Path) -> None:
    scheme_file = tmp_path / "my_scheme.py"
    scheme_file.write_text(textwrap.dedent("""
        from libuq.pipeline.models import SimDataParameter
        PARAMETERS = [
            SimDataParameter(name="a", attr_path="a", bounds=(0.0, 1.0)),
        ]
    """).strip() + "\n")
    scheme = load_perturbation_scheme(str(scheme_file))
    assert scheme is not None
    assert hasattr(scheme, "PARAMETERS")
    assert len(scheme.PARAMETERS) == 1


def test_load_from_dotted_module() -> None:
    """Install a synthetic module via sys.modules and load by dotted name."""
    mod_name = "_uq_test_synth_scheme"
    module = types.ModuleType(mod_name)
    module.PARAMETERS = [  # type: ignore[attr-defined]
        SimDataParameter(name="x", attr_path="x", bounds=(0.0, 1.0)),
    ]
    sys.modules[mod_name] = module
    try:
        scheme = load_perturbation_scheme(mod_name)
        assert scheme is module
    finally:
        sys.modules.pop(mod_name, None)


def test_load_unknown_raises_with_actionable_message() -> None:
    with pytest.raises(ImportError, match="not a file path and not an importable"):
        load_perturbation_scheme("definitely_not_a_real_package.foo")


# ── build_parameters_from_scheme ───────────────────────────────────────


class FakeSimData:
    """Minimal stub with the attributes the dynamic scheme reads."""
    def __init__(self):
        self.elongation_rate = 22.0


def test_no_scheme_returns_default_six() -> None:
    params = build_parameters_from_scheme(None, sim_data=FakeSimData())
    assert len(params) == len(DEFAULT_SIM_DATA_PARAMETERS)
    assert all(isinstance(p, SimDataParameter) for p in params)


def test_static_parameters_list(tmp_path: Path) -> None:
    scheme_file = tmp_path / "static.py"
    scheme_file.write_text(textwrap.dedent("""
        from libuq.pipeline.models import SimDataParameter
        PARAMETERS = [
            SimDataParameter(name="elongation_rate",
                             attr_path="elongation_rate",
                             bounds=(15.0, 28.0)),
        ]
    """).strip() + "\n")
    scheme = load_perturbation_scheme(str(scheme_file))
    params = build_parameters_from_scheme(scheme, sim_data=FakeSimData())
    assert len(params) == 1
    assert params[0].name == "elongation_rate"
    assert params[0].bounds == (15.0, 28.0)


def test_build_parameters_function_takes_priority_over_static(tmp_path: Path) -> None:
    """When both build_parameters and PARAMETERS exist, the function wins."""
    scheme_file = tmp_path / "both.py"
    scheme_file.write_text(textwrap.dedent("""
        from libuq.pipeline.models import SimDataParameter

        PARAMETERS = [
            SimDataParameter(name="static", attr_path="static", bounds=(0.0, 1.0)),
        ]

        def build_parameters(sim_data, n_samples=0, seed=42):
            return [
                SimDataParameter(name="dynamic", attr_path="dynamic", bounds=(0.0, 2.0)),
            ]
    """).strip() + "\n")
    scheme = load_perturbation_scheme(str(scheme_file))
    params = build_parameters_from_scheme(scheme, sim_data=FakeSimData())
    assert len(params) == 1
    assert params[0].name == "dynamic"


def test_build_parameters_pct_around_baseline(tmp_path: Path) -> None:
    """The bundled example scheme reads baseline values and emits ±30%."""
    scheme = load_perturbation_scheme(
        "examples/perturbation_schemes/pct_around_baseline.py"
    )
    # Stub a sim_data with one of the attr_paths reachable.
    class SimDataWithRate:
        class process:
            class translation:
                basal_elongation_rate = 22.0
            class transcription:
                fraction_active_rnap_free = 0.36
                fraction_active_rnap_bound = 0.17
            class metabolism:
                kinetic_objective_weight = 1e-7
                secretion_penalty_coeff = 0.001
        class mass:
            cell_dry_mass_fraction = 0.30

    params = build_parameters_from_scheme(scheme, sim_data=SimDataWithRate())
    # All six default attrs are reachable on the stub
    assert len(params) == 6
    rate_param = next(p for p in params if p.name == "basal_elongation_rate")
    assert rate_param.bounds[0] == pytest.approx(22.0 * 0.7, rel=0.01)
    assert rate_param.bounds[1] == pytest.approx(22.0 * 1.3, rel=0.01)


def test_scheme_without_either_raises_clear_error(tmp_path: Path) -> None:
    scheme_file = tmp_path / "empty.py"
    scheme_file.write_text("# no PARAMETERS, no build_parameters\n")
    scheme = load_perturbation_scheme(str(scheme_file))
    with pytest.raises(ValueError, match="neither build_parameters"):
        build_parameters_from_scheme(scheme, sim_data=FakeSimData())


def test_scheme_returning_non_list_raises(tmp_path: Path) -> None:
    scheme_file = tmp_path / "bad_return.py"
    scheme_file.write_text(textwrap.dedent("""
        def build_parameters(sim_data, n_samples=0, seed=42):
            return "not a list"
    """).strip() + "\n")
    scheme = load_perturbation_scheme(str(scheme_file))
    with pytest.raises(TypeError, match="must return list"):
        build_parameters_from_scheme(scheme, sim_data=FakeSimData())


def test_static_parameters_must_be_list(tmp_path: Path) -> None:
    scheme_file = tmp_path / "bad_static.py"
    scheme_file.write_text("PARAMETERS = {'not': 'a list'}\n")
    scheme = load_perturbation_scheme(str(scheme_file))
    with pytest.raises(TypeError, match="PARAMETERS must be a"):
        build_parameters_from_scheme(scheme, sim_data=FakeSimData())


# ── _filter_kwargs ─────────────────────────────────────────────────────


def test_filter_kwargs_passes_only_accepted() -> None:
    def fn(sim_data, n_samples=0):
        pass
    accepted = _filter_kwargs(fn, n_samples=10, seed=42)
    assert accepted == {"n_samples": 10}


def test_filter_kwargs_var_keyword_accepts_all() -> None:
    def fn(sim_data, **kwargs):
        pass
    accepted = _filter_kwargs(fn, n_samples=10, seed=42, extra="ok")
    assert accepted == {"n_samples": 10, "seed": 42, "extra": "ok"}


def test_filter_kwargs_no_accepted_returns_empty() -> None:
    def fn(sim_data):
        pass
    accepted = _filter_kwargs(fn, n_samples=10, seed=42)
    assert accepted == {}


# ── Output payload round-trip ──────────────────────────────────────────


def test_default_params_round_trip_to_json(tmp_path: Path) -> None:
    params = build_parameters_from_scheme(None, sim_data=FakeSimData())
    payload = [p.model_dump() for p in params]
    out = tmp_path / "out.json"
    out.write_text(json.dumps(payload, indent=2))
    loaded = json.loads(out.read_text())
    assert len(loaded) == len(DEFAULT_SIM_DATA_PARAMETERS)
    rehydrated = [SimDataParameter.from_dict(d) for d in loaded]
    assert rehydrated[0].name == DEFAULT_SIM_DATA_PARAMETERS[0].name


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
