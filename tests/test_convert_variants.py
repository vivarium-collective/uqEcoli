"""Tests for `uq.convert_variants`.

Synthetic ``apply_variant`` functions and lightweight test doubles for
``SimulationDataEcoli`` so unit tests don't need a real vEcoli install.
"""

from __future__ import annotations

import json
import sys
import types
from pathlib import Path

import numpy as np
import pytest

from uq.convert_variants import (
    DiffEntry,
    _diff_to_mutation_dict,
    _fallback_parse_variants,
    _infer_params_file,
    _is_opaque_leaf,
    _values_equal,
    _walk_diff,
    convert_variants_config,
    write_params_file,
)


# ── Test doubles ──────────────────────────────────────────────────────


class FakeProcess:
    def __init__(self, kcat: float, km: float):
        self.kcat = kcat
        self.km = km


class FakeMetabolism:
    def __init__(self):
        self.objective_weight = 1e-7
        self.secretion_penalty = 0.001
        self.kcats = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        self.process = FakeProcess(kcat=10.0, km=0.5)


class FakeSimData:
    """Lightweight stand-in for SimulationDataEcoli."""

    def __init__(self):
        self.condition = "basal"
        self.dry_mass_fraction = 0.30
        self.elongation_rate = 22.0
        self.metabolism = FakeMetabolism()


def _make_fake_sim_data() -> FakeSimData:
    return FakeSimData()


# ── Synthetic variant modules ─────────────────────────────────────────


def _install_synthetic_variant(name: str, apply_variant) -> str:
    """Install a fake variant module under ecoli.variants.<name> for the test."""
    full_name = f"ecoli.variants.{name}"
    module = types.ModuleType(full_name)
    module.apply_variant = apply_variant  # type: ignore[attr-defined]
    # Ensure parent packages exist as namespaces
    sys.modules.setdefault("ecoli", types.ModuleType("ecoli"))
    sys.modules.setdefault("ecoli.variants", types.ModuleType("ecoli.variants"))
    sys.modules[full_name] = module
    return name


@pytest.fixture
def synthetic_scalar_variant():
    """A variant that perturbs two scalar attributes."""
    def apply(sim_data, params):
        sim_data.metabolism.objective_weight = params["weight"]
        sim_data.elongation_rate = params["rate"]
        return sim_data
    name = _install_synthetic_variant("scalar_sweep", apply)
    yield name
    del sys.modules[f"ecoli.variants.{name}"]


@pytest.fixture
def synthetic_categorical_variant():
    """A variant that swaps a string field."""
    def apply(sim_data, params):
        sim_data.condition = params["condition"]
        return sim_data
    name = _install_synthetic_variant("condition_sweep", apply)
    yield name
    del sys.modules[f"ecoli.variants.{name}"]


@pytest.fixture
def synthetic_array_variant():
    """A variant that mutates one element of a 1-D array."""
    def apply(sim_data, params):
        sim_data.metabolism.kcats[params["index"]] = params["value"]
        return sim_data
    name = _install_synthetic_variant("array_sweep", apply)
    yield name
    del sys.modules[f"ecoli.variants.{name}"]


# ── Pure-function tests ───────────────────────────────────────────────


def test_walk_diff_finds_scalar_change() -> None:
    a = FakeSimData()
    b = FakeSimData()
    b.elongation_rate = 25.0
    diff = _walk_diff(a, b)
    paths = {(e.attr_path, e.kind) for e in diff}
    assert ("elongation_rate", "scalar") in paths


def test_walk_diff_finds_nested_scalar_change() -> None:
    a = FakeSimData()
    b = FakeSimData()
    b.metabolism.objective_weight = 5e-7
    diff = _walk_diff(a, b)
    paths = {(e.attr_path, e.kind) for e in diff}
    assert ("metabolism.objective_weight", "scalar") in paths


def test_walk_diff_finds_categorical_change() -> None:
    a = FakeSimData()
    b = FakeSimData()
    b.condition = "with_aa"
    diff = _walk_diff(a, b)
    cat_entries = [e for e in diff if e.kind == "categorical"]
    assert len(cat_entries) == 1
    assert cat_entries[0].attr_path == "condition"
    assert cat_entries[0].mutated_value == "with_aa"


def test_walk_diff_skips_arrays_by_default() -> None:
    a = FakeSimData()
    b = FakeSimData()
    b.metabolism.kcats = np.array([1.0, 99.0, 3.0, 4.0, 5.0])
    diff = _walk_diff(a, b, include_arrays=False)
    kinds = {e.kind for e in diff if e.attr_path == "metabolism.kcats"}
    assert kinds == {"complex_skipped"}


def test_walk_diff_emits_indexed_when_include_arrays() -> None:
    a = FakeSimData()
    b = FakeSimData()
    b.metabolism.kcats = np.array([1.0, 99.0, 3.0, 4.0, 5.0])
    diff = _walk_diff(a, b, include_arrays=True)
    indexed = [e for e in diff if e.kind == "indexed"]
    assert len(indexed) == 1
    assert indexed[0].attr_path == "metabolism.kcats"
    assert indexed[0].indices == (1,)
    assert indexed[0].mutated_value == 99.0


def test_walk_diff_no_change_returns_empty() -> None:
    a = FakeSimData()
    b = FakeSimData()
    assert _walk_diff(a, b) == []


def test_diff_to_mutation_dict_scalar() -> None:
    diff = [DiffEntry("metabolism.weight", 1e-7, 5e-7, "scalar")]
    md, counts, warnings = _diff_to_mutation_dict(diff)
    assert md == {"metabolism.weight": 5e-7}
    assert counts["n_scalar"] == 1
    assert warnings == []


def test_diff_to_mutation_dict_categorical_warns() -> None:
    diff = [DiffEntry("condition", "basal", "with_aa", "categorical")]
    md, counts, warnings = _diff_to_mutation_dict(diff)
    assert md == {"condition": "with_aa"}
    assert counts["n_categorical"] == 1
    assert any("categorical" in w for w in warnings)
    assert any("unusable by `uq quantify`" in w for w in warnings)


def test_diff_to_mutation_dict_drops_second_indexed_per_path() -> None:
    """sim_data_setattr accepts one indexed write per attr_path; warn on drops."""
    diff = [
        DiffEntry("arr", 1.0, 10.0, "indexed", indices=(0,)),
        DiffEntry("arr", 2.0, 20.0, "indexed", indices=(1,)),  # dropped
    ]
    md, counts, warnings = _diff_to_mutation_dict(diff)
    assert md == {"arr": {"__index__": 0, "__value__": 10.0}}
    assert counts["n_indexed"] == 1
    assert any("dropped indexed mutation" in w for w in warnings)


def test_diff_to_mutation_dict_warns_over_cap() -> None:
    diff = [
        DiffEntry(f"a{i}", 0.0, float(i), "scalar")
        for i in range(50)
    ]
    md, counts, warnings = _diff_to_mutation_dict(diff, max_mutations=10)
    # Convert+warn mode: still emits all
    assert len(md) == 50
    assert any("usable mutations" in w for w in warnings)


def test_fallback_parse_variants_value_form() -> None:
    config = {"condition": {"value": ["basal", "with_aa", "no_oxygen"]}}
    entries = _fallback_parse_variants(config)
    assert entries == [
        {"condition": "basal"},
        {"condition": "with_aa"},
        {"condition": "no_oxygen"},
    ]


def test_fallback_parse_variants_op_zip() -> None:
    config = {"a": {"value": [1, 2, 3]}, "b": {"value": [10, 20, 30]}, "op": "zip"}
    entries = _fallback_parse_variants(config)
    assert entries == [
        {"a": 1, "b": 10},
        {"a": 2, "b": 20},
        {"a": 3, "b": 30},
    ]


def test_fallback_parse_variants_op_zip_length_mismatch() -> None:
    config = {"a": {"value": [1, 2, 3]}, "b": {"value": [10, 20]}, "op": "zip"}
    with pytest.raises(ValueError, match="must share length"):
        _fallback_parse_variants(config)


def test_fallback_parse_variants_op_prod() -> None:
    config = {
        "a": {"value": [1, 2]},
        "b": {"value": [10, 20, 30]},
        "op": "prod",
    }
    entries = _fallback_parse_variants(config)
    assert len(entries) == 6  # 2 × 3 cartesian product
    assert {"a": 1, "b": 10} in entries
    assert {"a": 2, "b": 30} in entries


def test_fallback_parse_variants_op_add() -> None:
    config = {"a": {"value": [1, 2]}, "b": {"value": [10, 20]}, "op": "add"}
    entries = _fallback_parse_variants(config)
    # add concatenates under {param_a}__{param_b} key name
    assert entries == [
        {"a__b": 1}, {"a__b": 2},
        {"a__b": 10}, {"a__b": 20},
    ]


def test_fallback_parse_variants_linspace() -> None:
    config = {"x": {"linspace": {"start": 0.0, "stop": 1.0, "num": 5}}}
    entries = _fallback_parse_variants(config)
    assert len(entries) == 5
    assert entries[0]["x"] == 0.0
    assert entries[-1]["x"] == 1.0


def test_fallback_parse_variants_unknown_np_func_errors() -> None:
    config = {"x": {"definitely_not_a_numpy_function": {}}}
    with pytest.raises(ValueError, match="unknown type"):
        _fallback_parse_variants(config)


def test_infer_params_file_scalar_bounds() -> None:
    from uq.convert_variants import VariantConversion
    per_variant = [
        VariantConversion({}, {"a.b": 1.0}, 1, 0, 0, 0),
        VariantConversion({}, {"a.b": 5.0}, 1, 0, 0, 0),
        VariantConversion({}, {"a.b": 3.0}, 1, 0, 0, 0),
    ]
    params = _infer_params_file(per_variant)
    assert len(params) == 1
    assert params[0].attr_path == "a.b"
    assert params[0].bounds == (1.0, 5.0)


def test_infer_params_file_single_value_widens() -> None:
    """Single-value sweeps get a small margin so PCRV doesn't degenerate."""
    from uq.convert_variants import VariantConversion
    per_variant = [VariantConversion({}, {"a.b": 2.0}, 1, 0, 0, 0)]
    params = _infer_params_file(per_variant)
    lo, hi = params[0].bounds
    assert lo < 2.0 < hi


# ── Top-level convert_variants_config ─────────────────────────────────


def test_passthrough_sim_data_setattr() -> None:
    cfg = {
        "sim_data_path": "/tmp/sd.pickle",
        "variants": {
            "sim_data_setattr": {
                "mutations": {
                    "value": [
                        {"a.b": 1.0, "a.c": 2.0},
                        {"a.b": 1.5, "a.c": 2.5},
                    ]
                }
            }
        }
    }
    result = convert_variants_config(cfg, sim_data=_make_fake_sim_data())
    out_block = result.output_config["variants"]
    assert "sim_data_setattr" in out_block
    assert out_block["sim_data_setattr"]["mutations"]["value"][0] == {"a.b": 1.0, "a.c": 2.0}
    assert result.n_variants == 2
    # Preserves other top-level keys
    assert result.output_config["sim_data_path"] == "/tmp/sd.pickle"


def test_converts_synthetic_scalar_variant(synthetic_scalar_variant) -> None:
    name = synthetic_scalar_variant
    cfg = {
        "variants": {
            name: {
                # Baseline objective_weight = 1e-7, elongation_rate = 22.0 —
                # all sweep values differ from baseline so every variant produces
                # a non-empty diff.
                "weight": {"value": [3e-7, 5e-7, 1e-6]},
                "rate": {"value": [20.0, 21.0, 24.0]},
                "op": "zip",
            }
        }
    }
    result = convert_variants_config(cfg, sim_data=_make_fake_sim_data())
    assert result.n_variants == 3
    for i, v in enumerate(result.per_variant):
        assert "metabolism.objective_weight" in v.mutation_dict
        assert "elongation_rate" in v.mutation_dict


def test_converts_synthetic_scalar_variant_single_param(synthetic_scalar_variant) -> None:
    """Single-param form works with the fallback parser."""
    name = synthetic_scalar_variant
    # Baseline elongation_rate = 22.0 — sweep values all differ from baseline
    # so every variant produces a non-empty diff.
    cfg = {
        "variants": {
            name: {
                "rate": {"value": [20.0, 21.0, 24.0]},
            }
        }
    }
    def apply_lenient(sim_data, params):
        if "weight" in params:
            sim_data.metabolism.objective_weight = params["weight"]
        if "rate" in params:
            sim_data.elongation_rate = params["rate"]
        return sim_data
    sys.modules[f"ecoli.variants.{name}"].apply_variant = apply_lenient

    result = convert_variants_config(cfg, sim_data=_make_fake_sim_data())
    assert result.n_variants == 3
    assert all("elongation_rate" in v.mutation_dict for v in result.per_variant)
    params = result.inferred_params
    assert any(p.attr_path == "elongation_rate" for p in params)
    rate_param = next(p for p in params if p.attr_path == "elongation_rate")
    assert rate_param.bounds == (20.0, 24.0)


def test_categorical_variant_emits_warning(synthetic_categorical_variant) -> None:
    name = synthetic_categorical_variant
    # Baseline condition = "basal" — sweep values all differ.
    cfg = {
        "variants": {
            name: {
                "condition": {"value": ["with_aa", "no_oxygen"]},
            }
        }
    }
    result = convert_variants_config(cfg, sim_data=_make_fake_sim_data())
    assert result.n_variants == 2
    assert all("condition" in v.mutation_dict for v in result.per_variant)
    assert all(v.n_categorical == 1 for v in result.per_variant)
    assert any("categorical" in w and "quantify" in w for w in result.warnings)


def test_array_variant_skipped_by_default(synthetic_array_variant) -> None:
    name = synthetic_array_variant
    cfg = {
        "variants": {
            name: {
                "index": {"value": [1]},  # mutate kcats[1]
            }
        }
    }
    # Single-param form — works with fallback parser. But the variant
    # also needs params["value"] — patch apply to mutate kcats[index] to
    # a fixed value when value is absent.
    def apply_fixed(sim_data, params):
        sim_data.metabolism.kcats[params["index"]] = 99.0
        return sim_data
    sys.modules[f"ecoli.variants.{name}"].apply_variant = apply_fixed

    result = convert_variants_config(
        cfg, sim_data=_make_fake_sim_data(), include_arrays=False,
    )
    # Without --include-arrays, the array change is skipped
    assert all(v.n_indexed == 0 for v in result.per_variant)
    # And one complex_skipped should be recorded
    assert all(v.n_complex_skipped >= 1 for v in result.per_variant)


def test_array_variant_emits_indexed_with_include_arrays(synthetic_array_variant) -> None:
    name = synthetic_array_variant
    cfg = {
        "variants": {
            name: {
                "index": {"value": [1]},
            }
        }
    }
    def apply_fixed(sim_data, params):
        sim_data.metabolism.kcats[params["index"]] = 99.0
        return sim_data
    sys.modules[f"ecoli.variants.{name}"].apply_variant = apply_fixed

    result = convert_variants_config(
        cfg, sim_data=_make_fake_sim_data(), include_arrays=True,
    )
    assert all(v.n_indexed == 1 for v in result.per_variant)
    md = result.per_variant[0].mutation_dict
    assert md["metabolism.kcats"] == {"__index__": 1, "__value__": 99.0}


def test_parca_variants_rejected() -> None:
    cfg = {
        "variants": {"sim_data_setattr": {"mutations": {"value": []}}},
        "parca_variants": [{"rnaseq_basal_dataset_id": "glucose"}],
    }
    with pytest.raises(ValueError, match="parca_variants"):
        convert_variants_config(cfg, sim_data=_make_fake_sim_data())


def test_missing_variants_key_rejected() -> None:
    cfg = {"sim_data_path": "/tmp/sd.pickle"}
    with pytest.raises(ValueError, match="no 'variants' key"):
        convert_variants_config(cfg, sim_data=_make_fake_sim_data())


def test_unknown_variant_module_raises_clear_error() -> None:
    cfg = {"variants": {"definitely_not_real_module_xyz": {"x": {"value": [1, 2]}}}}
    with pytest.raises(ImportError, match="--variant-module"):
        convert_variants_config(cfg, sim_data=_make_fake_sim_data())


def test_output_preserves_non_variants_keys(synthetic_scalar_variant) -> None:
    name = synthetic_scalar_variant
    def apply(sim_data, params):
        sim_data.elongation_rate = params["rate"]
        return sim_data
    sys.modules[f"ecoli.variants.{name}"].apply_variant = apply

    cfg = {
        "sim_data_path": "/tmp/sd.pickle",
        "experiment_id": "test_exp",
        "analysis_options": {"foo": "bar"},
        "variants": {name: {"rate": {"value": [20.0, 22.0]}}},
    }
    result = convert_variants_config(cfg, sim_data=_make_fake_sim_data())
    assert result.output_config["experiment_id"] == "test_exp"
    assert result.output_config["analysis_options"] == {"foo": "bar"}
    assert result.output_config["sim_data_path"] == "/tmp/sd.pickle"


def test_write_params_file_roundtrip(tmp_path: Path, synthetic_scalar_variant) -> None:
    name = synthetic_scalar_variant
    def apply(sim_data, params):
        sim_data.elongation_rate = params["rate"]
        return sim_data
    sys.modules[f"ecoli.variants.{name}"].apply_variant = apply

    cfg = {"variants": {name: {"rate": {"value": [20.0, 22.0, 24.0]}}}}
    result = convert_variants_config(cfg, sim_data=_make_fake_sim_data())

    out = tmp_path / "params.json"
    write_params_file(result.inferred_params, out)
    data = json.loads(out.read_text())
    assert isinstance(data, list)
    assert any(p["attr_path"] == "elongation_rate" for p in data)


# ── Walker A: opaque-leaf detection + pandas-aware equality ──────────

pd = pytest.importorskip("pandas", reason="pandas DataFrame tests need pandas")


def test_is_opaque_leaf_pandas_types() -> None:
    assert _is_opaque_leaf(pd.DataFrame({"a": [1, 2]}))
    assert _is_opaque_leaf(pd.Series([1, 2, 3]))
    assert _is_opaque_leaf(pd.Index(["x", "y"]))


def test_is_opaque_leaf_datetime() -> None:
    import datetime
    assert _is_opaque_leaf(datetime.date.today())
    assert _is_opaque_leaf(datetime.timedelta(days=1))


def test_is_opaque_leaf_does_not_swallow_normal_types() -> None:
    assert not _is_opaque_leaf({"a": 1})
    assert not _is_opaque_leaf([1, 2, 3])
    assert not _is_opaque_leaf(np.array([1.0, 2.0]))
    assert not _is_opaque_leaf("string")
    assert not _is_opaque_leaf(42)


def test_values_equal_unchanged_dataframe_after_deepcopy() -> None:
    """The bug that produced 918 warnings: unchanged DataFrames must compare equal."""
    import copy as _copy
    df = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
    df_copy = _copy.deepcopy(df)
    assert _values_equal(df, df_copy)


def test_values_equal_changed_dataframe_reports_difference() -> None:
    df1 = pd.DataFrame({"a": [1, 2, 3]})
    df2 = pd.DataFrame({"a": [1, 2, 99]})
    assert not _values_equal(df1, df2)


def test_values_equal_series_roundtrip() -> None:
    import copy as _copy
    s = pd.Series([1.0, 2.0, 3.0])
    assert _values_equal(s, _copy.deepcopy(s))
    assert not _values_equal(s, pd.Series([1.0, 2.0, 99.0]))


def test_walk_diff_does_not_overrecurse_into_dataframe() -> None:
    """Walker doesn't enumerate column names as fake attribute differences."""
    class Wrapper:
        def __init__(self, df):
            self.df = df

    df = pd.DataFrame({"counts": [1, 2], "weight": [0.5, 0.6]})
    a = Wrapper(df)
    b = Wrapper(df.copy())  # different object identity, same values
    diff = _walk_diff(a, b)
    # Should find zero differences — the DataFrame is unchanged.
    assert diff == [], f"got spurious diff entries: {[(e.attr_path, e.kind) for e in diff]}"


def test_walk_diff_marks_changed_dataframe_as_complex_skipped() -> None:
    class Wrapper:
        def __init__(self, df):
            self.df = df

    a = Wrapper(pd.DataFrame({"a": [1, 2, 3]}))
    b = Wrapper(pd.DataFrame({"a": [1, 2, 99]}))
    diff = _walk_diff(a, b)
    assert len(diff) == 1
    assert diff[0].attr_path == "df"
    assert diff[0].kind == "complex_skipped"


def test_walk_diff_pandas_change_inside_nested_dict() -> None:
    """Nested dicts of DataFrames diff cleanly — no column-name leakage."""
    class Wrapper:
        def __init__(self, recipes):
            self.recipes = recipes

    a = Wrapper({
        "rec_a": pd.DataFrame({"counts": [1, 2]}),
        "rec_b": pd.DataFrame({"counts": [3, 4]}),
    })
    b = Wrapper({
        "rec_a": pd.DataFrame({"counts": [1, 2]}),       # unchanged
        "rec_b": pd.DataFrame({"counts": [3, 99]}),      # one element changed
    })
    diff = _walk_diff(a, b)
    paths = [(e.attr_path, e.kind) for e in diff]
    # Only rec_b should show up
    assert paths == [("recipes.rec_b", "complex_skipped")], paths


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
