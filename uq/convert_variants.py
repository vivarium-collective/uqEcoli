"""Convert arbitrary vEcoli variant configs to the UQ-canonical
``sim_data_setattr`` form.

The canonical UQ contract requires that every model evaluation correspond
to a known vector x ∈ R^d, encoded as a ``sim_data_setattr`` mutation
dict. This converter takes a config using any vEcoli variant module
(``condition``, ``flux_kinetics``, ``new_gene_internal_shift``, custom
modules), applies each variant entry in-process against a baseline
``simData.cPickle``, diffs the result, and emits the changes as a list
of ``sim_data_setattr`` mutation dicts that reproduce the same vEcoli
behavior.

Observational by design (we run the variant and record what changed),
not analytical (parsing the variant module's source would only work for
trivial cases). Module-agnostic.

Limitations:
  * ``sim_data_setattr`` mutation dicts can carry at most one indexed
    mutation per ``attr_path`` key (Python dict semantics). Whole-array
    rewrites cannot round-trip; we emit per-index entries up to a
    configurable cap and refuse beyond that.
  * Non-numeric mutations (strings, booleans) round-trip the *behavior*
    but cannot drive a PCE — the resulting cache is unusable by
    ``uq quantify``. Warn and continue per user direction (convert + warn
    rather than refuse).
  * ``parca_variants`` perturbs Parca recomputation upstream of ``apply_variant``
    and is rejected; users should drive that path through ``uq sample
    --conditions ...`` (cross-condition GSA).
"""

from __future__ import annotations

import copy
import dataclasses
import importlib
import itertools
import json
import sys
from pathlib import Path
from typing import Any, Callable, Literal

import numpy as np

from libuq.pipeline.models import SimDataParameter

# Match libuq.pipeline.param_loader._SKIP_ATTRS — these are expensive or
# non-data attributes (JIT compile artifacts, etc.) that should not
# participate in the diff.
_SKIP_ATTRS = frozenset({"derivatives_jit", "derivatives", "jit", "jacobian"})

# Variant modules that already use ``sim_data_setattr`` semantics or that
# we explicitly handle elsewhere in the pipeline (parca_variants is
# multi-condition territory, refused here).
_PASSTHROUGH_MODULES = frozenset({"sim_data_setattr"})

DiffKind = Literal["scalar", "indexed", "categorical", "complex_skipped"]


@dataclasses.dataclass(frozen=True)
class DiffEntry:
    """One leaf-level change between baseline and mutated sim_data."""

    attr_path: str
    baseline_value: Any
    mutated_value: Any
    kind: DiffKind
    indices: tuple[int, ...] | None = None  # for kind="indexed"


@dataclasses.dataclass
class VariantConversion:
    """The converted form of a single variant entry."""

    original_params: dict[str, Any]
    mutation_dict: dict[str, Any]
    n_scalar: int
    n_indexed: int
    n_categorical: int
    n_complex_skipped: int


@dataclasses.dataclass
class ConvertResult:
    """Top-level result of converting a vEcoli variants config."""

    output_config: dict[str, Any]
    per_variant: list[VariantConversion]
    inferred_params: list[SimDataParameter]
    warnings: list[str]

    @property
    def n_variants(self) -> int:
        return len(self.per_variant)

    @property
    def dimension(self) -> int:
        """Number of distinct dot-paths across all variants."""
        return len(self.inferred_params)


def _is_skippable_attr(name: str) -> bool:
    return name in _SKIP_ATTRS or name.startswith("_") or name.startswith("__")


def _is_scalar(value: Any) -> bool:
    """True iff value is a single numeric/bool/None we can write back via setattr."""
    if value is None:
        return True
    if isinstance(value, (bool, int, float, np.floating, np.integer)):
        return True
    return False


def _is_categorical(value: Any) -> bool:
    """Strings — round-trippable via setattr but not UQ-usable."""
    return isinstance(value, str)


def _is_opaque_leaf(value: Any) -> bool:
    """Types we treat as atomic for diff purposes.

    Walking inside these is either unsafe (pandas internals leak column
    names through ``__dict__``) or pointless (datetime, units, etc. — not
    representable as sim_data_setattr mutations). They're compared as
    whole values via ``_values_equal`` and emitted as ``complex_skipped``
    when different.
    """
    # pandas types — checked by class-name to avoid importing pandas at module
    # load. Class hierarchy: DataFrame/Series/Index/MultiIndex/CategoricalIndex.
    cls_module = type(value).__module__
    if cls_module.startswith("pandas."):
        return True
    # python datetime/date/timedelta — not PCE-tractable
    if cls_module.startswith("datetime"):
        return True
    return False


def _values_equal(a: Any, b: Any) -> bool:
    """Equality test that handles numpy arrays, pandas types, and scalars.

    Returns True when the values are equivalent (even if they live at
    different memory addresses, e.g. after ``deepcopy``). The default
    ``bool(a == b)`` path is the last resort because for many opaque types
    (pandas DataFrame, Series, Index) ``a == b`` returns an array-like
    object whose ``bool()`` raises — leading to false-positive "different"
    reports for unchanged data.
    """
    if a is b:
        return True
    if a is None and b is None:
        return True
    if a is None or b is None:
        return False

    # Pandas types — use pandas.testing.assert_*. Import lazily so the
    # converter doesn't hard-depend on pandas.
    a_mod, b_mod = type(a).__module__, type(b).__module__
    if a_mod.startswith("pandas.") or b_mod.startswith("pandas."):
        try:
            import pandas as pd
            from pandas import testing as pdt
        except ImportError:
            try:
                return bool(a == b)  # last-ditch fallback
            except (TypeError, ValueError):
                return False
        a_cls = type(a)
        if a_cls is not type(b):
            return False
        try:
            if isinstance(a, pd.DataFrame):
                pdt.assert_frame_equal(a, b, check_exact=False)
            elif isinstance(a, pd.Series):
                pdt.assert_series_equal(a, b, check_exact=False)
            elif isinstance(a, pd.Index):
                pdt.assert_index_equal(a, b, check_exact=False)
            else:
                return bool(a.equals(b)) if hasattr(a, "equals") else False
            return True
        except (AssertionError, TypeError, ValueError):
            return False

    if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        a_arr, b_arr = np.asarray(a), np.asarray(b)
        if a_arr.shape != b_arr.shape:
            return False
        if a_arr.dtype.kind in {"U", "S", "O"} or b_arr.dtype.kind in {"U", "S", "O"}:
            try:
                return bool((a_arr == b_arr).all())
            except (TypeError, ValueError):
                return False
        try:
            return bool(np.allclose(a_arr, b_arr, equal_nan=True))
        except (TypeError, ValueError):
            return bool((a_arr == b_arr).all())
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        return len(a) == len(b) and all(_values_equal(x, y) for x, y in zip(a, b))
    if isinstance(a, dict) and isinstance(b, dict):
        if set(a.keys()) != set(b.keys()):
            return False
        return all(_values_equal(a[k], b[k]) for k in a)
    try:
        result = a == b
    except (TypeError, ValueError):
        return False
    # Guard against types whose __eq__ returns array-like
    if isinstance(result, (np.ndarray, list, tuple)):
        try:
            return bool(np.asarray(result).all())
        except (TypeError, ValueError):
            return False
    try:
        return bool(result)
    except (TypeError, ValueError):
        return False


def _walk_diff(
    baseline: Any,
    mutated: Any,
    path: tuple[str, ...] = (),
    max_depth: int = 6,
    include_arrays: bool = False,
) -> list[DiffEntry]:
    """Recursively collect leaf-level changes between two sim_data objects.

    Walks attribute trees up to ``max_depth`` and records every leaf where
    baseline and mutated differ. Arrays produce one ``DiffEntry`` per
    changed index when ``include_arrays=True``; otherwise they're recorded
    as ``complex_skipped``.
    """
    if max_depth < 0:
        return []
    if baseline is mutated:  # identical object reference — nothing to diff
        return []

    entries: list[DiffEntry] = []
    dot_path = ".".join(path)

    # Opaque types (pandas DataFrame/Series/Index, datetime, etc.) — leaf-compare
    # via _values_equal rather than descending into their __dict__ where pandas
    # internals leak column names and other non-attribute structure.
    if _is_opaque_leaf(baseline) or _is_opaque_leaf(mutated):
        if not _values_equal(baseline, mutated):
            entries.append(DiffEntry(
                attr_path=dot_path,
                baseline_value=baseline,
                mutated_value=mutated,
                kind="complex_skipped",
            ))
        return entries

    # Categorical leaf
    if _is_categorical(baseline) or _is_categorical(mutated):
        if not _values_equal(baseline, mutated):
            entries.append(DiffEntry(
                attr_path=dot_path,
                baseline_value=baseline,
                mutated_value=mutated,
                kind="categorical",
            ))
        return entries

    # Scalar leaf
    if _is_scalar(baseline) and _is_scalar(mutated):
        if not _values_equal(baseline, mutated):
            entries.append(DiffEntry(
                attr_path=dot_path,
                baseline_value=baseline,
                mutated_value=mutated,
                kind="scalar",
            ))
        return entries

    # ndarray leaf
    if isinstance(baseline, np.ndarray) and isinstance(mutated, np.ndarray):
        if _values_equal(baseline, mutated):
            return entries
        if not include_arrays:
            entries.append(DiffEntry(
                attr_path=dot_path,
                baseline_value=baseline,
                mutated_value=mutated,
                kind="complex_skipped",
            ))
            return entries
        if baseline.shape != mutated.shape:
            entries.append(DiffEntry(
                attr_path=dot_path,
                baseline_value=baseline,
                mutated_value=mutated,
                kind="complex_skipped",
            ))
            return entries
        if baseline.dtype.kind in {"U", "S", "O"}:
            # Non-numeric array
            entries.append(DiffEntry(
                attr_path=dot_path,
                baseline_value=baseline,
                mutated_value=mutated,
                kind="complex_skipped",
            ))
            return entries
        # Numeric array — per-element indexed entries
        if baseline.ndim != 1:
            entries.append(DiffEntry(
                attr_path=dot_path,
                baseline_value=baseline,
                mutated_value=mutated,
                kind="complex_skipped",
            ))
            return entries
        diff_mask = ~np.isclose(baseline, mutated, equal_nan=True)
        for idx in np.flatnonzero(diff_mask):
            entries.append(DiffEntry(
                attr_path=dot_path,
                baseline_value=float(baseline[idx]),
                mutated_value=float(mutated[idx]),
                kind="indexed",
                indices=(int(idx),),
            ))
        return entries

    # Lists / tuples — only diff at top-level lengths; nested structures
    # are too varied to handle generically without going off the rails.
    if isinstance(baseline, (list, tuple)) and isinstance(mutated, (list, tuple)):
        if _values_equal(baseline, mutated):
            return entries
        entries.append(DiffEntry(
            attr_path=dot_path,
            baseline_value=baseline,
            mutated_value=mutated,
            kind="complex_skipped",
        ))
        return entries

    # Dictionaries — walk by key
    if isinstance(baseline, dict) and isinstance(mutated, dict):
        for key in set(baseline.keys()) | set(mutated.keys()):
            sub_path = (*path, str(key))
            if key not in baseline or key not in mutated:
                entries.append(DiffEntry(
                    attr_path=".".join(sub_path),
                    baseline_value=baseline.get(key),
                    mutated_value=mutated.get(key),
                    kind="complex_skipped",
                ))
                continue
            entries.extend(_walk_diff(
                baseline[key], mutated[key], sub_path,
                max_depth=max_depth - 1, include_arrays=include_arrays,
            ))
        return entries

    # Plain object — walk by attribute. Skip callables, dunders, jit artifacts.
    if hasattr(baseline, "__dict__") and hasattr(mutated, "__dict__"):
        base_attrs = vars(baseline)
        mut_attrs = vars(mutated)
        for name in set(base_attrs.keys()) | set(mut_attrs.keys()):
            if _is_skippable_attr(name):
                continue
            base_val = base_attrs.get(name)
            mut_val = mut_attrs.get(name)
            if callable(base_val) or callable(mut_val):
                continue
            sub_path = (*path, name)
            entries.extend(_walk_diff(
                base_val, mut_val, sub_path,
                max_depth=max_depth - 1, include_arrays=include_arrays,
            ))
        return entries

    # Anything else (dataframes, mappingproxies, etc.) — skip with a complex marker
    if not _values_equal(baseline, mutated):
        entries.append(DiffEntry(
            attr_path=dot_path,
            baseline_value=baseline,
            mutated_value=mutated,
            kind="complex_skipped",
        ))
    return entries


def _diff_to_mutation_dict(
    diff_entries: list[DiffEntry],
    max_mutations: int = 200,
) -> tuple[dict[str, Any], dict[str, int], list[str]]:
    """Encode a diff as a ``sim_data_setattr`` mutation dict.

    Returns:
        (mutation_dict, counts, warnings) where counts is a dict with keys
        ``n_scalar``, ``n_indexed``, ``n_categorical``, ``n_complex_skipped``.

    Drops indexed mutations after the first per attr_path (sim_data_setattr
    dict semantics — only one indexed entry per key survives). Emits a
    warning when this happens so the user knows array-level perturbations
    won't round-trip.
    """
    mutation_dict: dict[str, Any] = {}
    counts = {"n_scalar": 0, "n_indexed": 0, "n_categorical": 0, "n_complex_skipped": 0}
    warnings: list[str] = []
    indexed_paths_seen: set[str] = set()

    total_useful = sum(
        1 for e in diff_entries
        if e.kind in {"scalar", "indexed", "categorical"}
    )
    if total_useful > max_mutations:
        warnings.append(
            f"variant produced {total_useful} usable mutations (> --max-mutations "
            f"{max_mutations}). Convert+warn mode active; emitting all. "
            "Consider --max-mutations override or a smaller variant scope."
        )

    for entry in diff_entries:
        if entry.kind == "scalar":
            mutation_dict[entry.attr_path] = (
                float(entry.mutated_value)
                if isinstance(entry.mutated_value, (int, float, np.integer, np.floating))
                else entry.mutated_value
            )
            counts["n_scalar"] += 1
        elif entry.kind == "indexed":
            if entry.indices is None:
                continue
            if entry.attr_path in indexed_paths_seen:
                warnings.append(
                    f"dropped indexed mutation at {entry.attr_path}[{entry.indices[0]}]: "
                    "sim_data_setattr accepts only one indexed write per attr_path "
                    "(dict-key collision). Array-level perturbations cannot round-trip."
                )
                continue
            mutation_dict[entry.attr_path] = {
                "__index__": int(entry.indices[0]),
                "__value__": float(entry.mutated_value),
            }
            indexed_paths_seen.add(entry.attr_path)
            counts["n_indexed"] += 1
        elif entry.kind == "categorical":
            mutation_dict[entry.attr_path] = entry.mutated_value
            counts["n_categorical"] += 1
            warnings.append(
                f"categorical mutation at {entry.attr_path}: round-trips vEcoli "
                "behavior but cannot drive a PCE — the resulting cache will be "
                "unusable by `uq quantify`."
            )
        else:  # complex_skipped
            counts["n_complex_skipped"] += 1
            warnings.append(
                f"skipped complex mutation at {entry.attr_path or '<root>'}: "
                "structural change (dataframe, non-1D array, dict shape, etc.) "
                "not representable as a sim_data_setattr scalar/indexed write."
            )

    return mutation_dict, counts, warnings


def _resolve_apply_variant(
    module_name: str,
    module_override: str | None,
) -> Callable[[Any, dict[str, Any]], Any]:
    """Locate the ``apply_variant`` callable for a given variant module name.

    Default: canonical vEcoli pattern ``ecoli.variants.<name>.apply_variant``.
    Override: caller-supplied module path (e.g. ``my_pkg.variants.foo``).
    """
    target = module_override if module_override is not None else f"ecoli.variants.{module_name}"
    try:
        module = importlib.import_module(target)
    except ImportError as exc:
        raise ImportError(
            f"Could not import variant module {target!r}. Pass --variant-module "
            "<pkg.path> to override the default ecoli.variants.<name> location."
        ) from exc
    if not hasattr(module, "apply_variant"):
        raise AttributeError(
            f"Module {target!r} has no apply_variant(sim_data, params) function. "
            "Custom variant modules must expose this entry point."
        )
    return module.apply_variant  # type: ignore[no-any-return]


def _resolve_variant_entries(
    inner_config: dict[str, Any],
) -> list[dict[str, Any]]:
    """Delegate to vEcoli's parse_variants to expand value/linspace/nested forms.

    Falls back to a local resolver if vEcoli isn't importable. vEcoli's
    ``runscripts/create_variants.py`` does ``from configs import CONFIG_DIR_PATH``
    at module load — that import only works when the vEcoli repo root is on
    ``sys.path``. We prepend it here using the same root-detection logic
    that ``uq sample`` uses for the workflow subprocess.
    """
    _ensure_vecoli_on_path()
    try:
        from runscripts.create_variants import parse_variants  # type: ignore[import-not-found]
        return parse_variants(copy.deepcopy(inner_config))
    except (ImportError, ModuleNotFoundError):
        return _fallback_parse_variants(inner_config)


def _ensure_vecoli_on_path() -> None:
    """Prepend vEcoli's repo root to sys.path so its internal imports work.

    ``runscripts/create_variants.py`` does ``from configs import CONFIG_DIR_PATH``
    at module-load time. The ``configs`` package lives at the vEcoli repo
    root, not under the ``ecoli`` editable install — so the editable
    install alone doesn't make that import work.
    """
    try:
        from uq.vecoli_config import _get_vecoli_root
        vecoli_root = _get_vecoli_root()
        if vecoli_root not in sys.path:
            sys.path.insert(0, vecoli_root)
    except (ImportError, RuntimeError):
        # No vEcoli on the system — the caller will fall through to the
        # local parser, which handles value/linspace/op:zip/op:prod.
        pass


def _fallback_parse_variants(inner_config: dict[str, Any]) -> list[dict[str, Any]]:
    """Local re-implementation of vEcoli's parse_variants.

    Mirrors the canonical logic in ``runscripts/create_variants.py::parse_variants``
    so this converter works even when vEcoli's repo root isn't on sys.path
    (e.g. in CI, in editable-install-only environments, or when vEcoli's
    own imports fail downstream). Sticks to vEcoli's grammar.

    Supports per-param types: ``value``, ``linspace``, ``nested``, and any
    other ``np.<func>`` (matching vEcoli's ``getattr(np, param_type)`` path).
    Supports operations: ``zip``, ``prod``, ``add`` (matching vEcoli verbatim).
    """
    cfg = copy.deepcopy(inner_config)
    operation = None
    if len(cfg) > 1:
        if "op" not in cfg:
            raise ValueError(
                "Variant has more than 1 parameter but no 'op' key — "
                "match vEcoli's parse_variants contract."
            )
        operation = cfg.pop("op")
    elif "op" in cfg:
        raise ValueError(
            "Variant has a single parameter and should not define 'op'."
        )

    parsed: dict[str, list[Any]] = {}
    for param_name, param_conf in cfg.items():
        if not isinstance(param_conf, dict) or len(param_conf) != 1:
            raise ValueError(
                f"Param {param_name!r} should have exactly one type key "
                f"(value/linspace/nested/...). Got {param_conf!r}."
            )
        param_type, param_vals = next(iter(param_conf.items()))
        if param_type == "value":
            if not isinstance(param_vals, list):
                raise ValueError(f"{param_name} should have a list value.")
            parsed[param_name] = list(param_vals)
        elif param_type == "nested":
            parsed[param_name] = _fallback_parse_variants(param_vals)
        else:
            try:
                np_func = getattr(np, param_type)
            except AttributeError as exc:
                raise ValueError(
                    f"Param {param_name!r} has unknown type {param_type!r}."
                ) from exc
            parsed[param_name] = np_func(**param_vals).tolist()

    if operation == "prod":
        param_tuples = itertools.product(*(parsed[k] for k in parsed))
        return [
            {name: val for name, val in zip(parsed.keys(), tup)}
            for tup in param_tuples
        ]
    if operation == "zip":
        n_combos = -1
        for name, val in parsed.items():
            if n_combos == -1:
                n_combos = len(val)
            elif len(val) != n_combos:
                raise ValueError(
                    f"Param {name!r} has {len(val)} values; expected {n_combos} "
                    "(all params under 'op: zip' must share length)."
                )
        return [
            {name: vals[i] for name, vals in parsed.items()}
            for i in range(n_combos)
        ]
    if operation == "add":
        combined = "__".join(parsed)
        out: list[dict[str, Any]] = []
        for vals in parsed.values():
            out.extend({combined: v} for v in vals)
        return out
    if operation is None:
        # Single-param case — name preserved
        param_name, vals = next(iter(parsed.items()))
        return [{param_name: v} for v in vals]
    raise ValueError(f"Unknown op {operation!r}; expected zip/prod/add.")


def _infer_params_file(
    per_variant: list[VariantConversion],
) -> list[SimDataParameter]:
    """Derive a --params-file from the union of dot-paths across variants.

    For each dot-path that appears as a scalar in at least one variant,
    emit a ``SimDataParameter`` with bounds = (min, max) of the observed
    values across all variants. Indexed mutations get one parameter per
    (attr_path, index) pair so the column ordering matches ``X``.
    """
    from collections import defaultdict

    scalar_values: dict[str, list[float]] = defaultdict(list)
    indexed_values: dict[tuple[str, int], list[float]] = defaultdict(list)

    for v in per_variant:
        for attr_path, value in v.mutation_dict.items():
            if isinstance(value, dict) and "__index__" in value:
                key = (attr_path, value["__index__"])
                indexed_values[key].append(float(value["__value__"]))
            elif isinstance(value, (int, float, np.integer, np.floating)):
                scalar_values[attr_path].append(float(value))

    params: list[SimDataParameter] = []
    for attr_path, vals in sorted(scalar_values.items()):
        lo, hi = (min(vals), max(vals)) if vals else (0.0, 1.0)
        if lo == hi:
            # Single-value sweep — widen by 5% on each side so PCRV doesn't
            # degenerate at a point.
            margin = max(abs(lo) * 0.05, 1e-12)
            lo, hi = lo - margin, hi + margin
        params.append(SimDataParameter(
            name=attr_path.replace(".", "_"),
            attr_path=attr_path,
            bounds=(lo, hi),
            description=f"inferred from convert-variants ({len(vals)} samples)",
        ))
    for (attr_path, idx), vals in sorted(indexed_values.items()):
        lo, hi = (min(vals), max(vals)) if vals else (0.0, 1.0)
        if lo == hi:
            margin = max(abs(lo) * 0.05, 1e-12)
            lo, hi = lo - margin, hi + margin
        params.append(SimDataParameter(
            name=f"{attr_path.replace('.', '_')}_{idx}",
            attr_path=attr_path,
            bounds=(lo, hi),
            index=idx,
            description=f"inferred from convert-variants ({len(vals)} samples)",
        ))
    return params


def convert_variants_config(
    in_config: dict[str, Any],
    sim_data: Any,
    max_mutations: int = 200,
    include_arrays: bool = False,
    variant_module_override: str | None = None,
) -> ConvertResult:
    """Convert a vEcoli variants config to UQ-canonical sim_data_setattr.

    Args:
        in_config: Full vEcoli workflow config dict (must contain a ``variants``
            key; other keys are preserved verbatim in the output).
        sim_data: A loaded ``SimulationDataEcoli`` baseline (or any object
            with attribute access — synthetic test doubles work too).
        max_mutations: Per-variant cap on usable mutation count. Beyond
            this, convert+warn mode emits a warning but keeps going (per
            user spec: easily switched to refuse later).
        include_arrays: When True, 1-D numeric array changes are emitted
            as per-index indexed mutations. When False (default), array
            changes are reported as ``complex_skipped`` and dropped.
        variant_module_override: Optional Python module path used in place of
            the canonical ``ecoli.variants.<name>`` pattern. Falls back to
            the canonical path if None.

    Returns:
        ConvertResult containing the output config dict, the per-variant
        conversion records, inferred ``SimDataParameter`` specs, and any
        warnings.

    Raises:
        ValueError: when ``in_config`` lacks a ``variants`` key, declares
            ``parca_variants`` (multi-condition territory; use
            ``uq sample --conditions ...`` instead), or any other refusal.
        ImportError: when a non-canonical variant module name isn't
            importable.
    """
    if "variants" not in in_config:
        raise ValueError(
            "in_config has no 'variants' key — nothing to convert. The input "
            "must be a vEcoli workflow config with a top-level variants block."
        )
    if "parca_variants" in in_config:
        raise ValueError(
            "in_config has a 'parca_variants' block. parca_variants drives "
            "Parca recomputation (multi-condition GSA) and is not a scalar "
            "mutation surface. Use `uq sample --conditions <cid1> --conditions "
            "<cid2> ...` for cross-condition UQ instead."
        )

    variants_block = in_config["variants"]
    if not isinstance(variants_block, dict) or len(variants_block) == 0:
        raise ValueError(
            f"variants block must be a non-empty dict; got {variants_block!r}."
        )

    per_variant: list[VariantConversion] = []
    warnings: list[str] = []

    for module_name, inner_config in variants_block.items():
        # Fast path: sim_data_setattr is already in the target form. Just
        # pass the mutation entries through.
        if module_name in _PASSTHROUGH_MODULES:
            entries = inner_config.get("mutations", {}).get("value", [])
            if not isinstance(entries, list):
                raise ValueError(
                    f"sim_data_setattr.mutations.value must be a list; got "
                    f"{type(entries).__name__}."
                )
            for params_dict in entries:
                per_variant.append(VariantConversion(
                    original_params={"mutations": params_dict},
                    mutation_dict=dict(params_dict),
                    n_scalar=sum(
                        1 for v in params_dict.values()
                        if not isinstance(v, dict)
                    ),
                    n_indexed=sum(
                        1 for v in params_dict.values()
                        if isinstance(v, dict) and "__index__" in v
                    ),
                    n_categorical=0,
                    n_complex_skipped=0,
                ))
            continue

        apply_variant = _resolve_apply_variant(module_name, variant_module_override)
        variant_entries = _resolve_variant_entries(inner_config)

        for params_dict in variant_entries:
            baseline_copy = copy.deepcopy(sim_data)
            mutated = apply_variant(copy.deepcopy(sim_data), params_dict)
            if mutated is None:
                mutated = baseline_copy  # apply_variant mutated in place but
                # returned None — refresh the reference. Most modules return
                # sim_data; this is defensive.
            diff_entries = _walk_diff(
                baseline_copy, mutated, include_arrays=include_arrays,
            )
            mutation_dict, counts, var_warnings = _diff_to_mutation_dict(
                diff_entries, max_mutations=max_mutations,
            )
            warnings.extend(
                f"[{module_name} entry {len(per_variant)}] {w}"
                for w in var_warnings
            )
            per_variant.append(VariantConversion(
                original_params=params_dict,
                mutation_dict=mutation_dict,
                n_scalar=counts["n_scalar"],
                n_indexed=counts["n_indexed"],
                n_categorical=counts["n_categorical"],
                n_complex_skipped=counts["n_complex_skipped"],
            ))

    output_config = copy.deepcopy(in_config)
    output_config["variants"] = {
        "sim_data_setattr": {
            "mutations": {
                "value": [v.mutation_dict for v in per_variant],
            }
        }
    }

    inferred_params = _infer_params_file(per_variant)

    return ConvertResult(
        output_config=output_config,
        per_variant=per_variant,
        inferred_params=inferred_params,
        warnings=warnings,
    )


def write_params_file(
    params: list[SimDataParameter],
    out_path: str | Path,
) -> None:
    """Serialize inferred SimDataParameter specs to a --params-file JSON."""
    payload = [p.model_dump() for p in params]
    Path(out_path).write_text(json.dumps(payload, indent=2))
