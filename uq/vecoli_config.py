"""vEcoli workflow-config helpers for the ``uq sample --backend vecoli`` path.

These are the building blocks the CLI uses to encode PCRV samples as
vEcoli ``sim_data_setattr`` variants, locate the vEcoli repo root,
write the workflow JSON, and read Parquet outputs back. The CLI is the
primary consumer; the TUI and the ``UQPipeline`` process re-import from
here to avoid a CLI → TUI dependency inversion.
"""

from __future__ import annotations

import contextlib
import dataclasses
import json
from math import comb
from pathlib import Path
from typing import Any, Literal

import numpy as np
import polars as pl

from libuq.pipeline.models import SimDataParameter

PceAdequacy = Literal["ok", "marginal", "underdetermined"]
DesignConditioning = Literal["well-conditioned", "marginal", "ill-conditioned", "singular"]


def _get_vecoli_root() -> str:
    import ecoli  # type: ignore[import-not-found]

    root = Path(ecoli.__file__).resolve().parent.parent
    if (root / "configs" / "__init__.py").exists():
        return str(root)
    raise RuntimeError("Cannot find vEcoli repo root")


def _build_config(
    sim_data_path: str,
    output_dir: str,
    variants_section: dict[str, Any],
    experiment_id: str = "uqpc_batch",
    n_init_sims: int = 1,
    generations: int = 1,
    max_duration: float = 10800.0,
    base_config_path: str | None = None,
    conditions: list[str] | None = None,
) -> dict[str, Any]:
    """Build a vEcoli workflow config JSON.

    Args:
        base_config_path: Optional path to a base vEcoli config to merge with.
            Preserves ``parca_variants``, ``analysis_options``, and other
            multi-parca keys from the base config.
        conditions: Optional list of ``rnaseq_basal_dataset_id`` strings.
            When provided, populates ``parca_variants`` for multi-parca
            cross-condition UQ.
    """
    if base_config_path:
        base = json.loads(Path(base_config_path).read_text())
    else:
        base = {}

    # UQ fields override base; everything else preserved.
    # BYO-variants escape hatch: if --base-config already declares a `variants`
    # block, trust it verbatim and skip the auto-generated sim_data_setattr.
    # Caller is then responsible for X ↔ Y correspondence in the cache.
    # Scoped to local (--backend vecoli) only; remote/v2ecoli paths unchanged.
    upd: dict[str, Any] = {
        "sim_data_path": sim_data_path,
        "experiment_id": experiment_id,
        "emitter": "parquet",
        "emitter_arg": {
            "out_dir": output_dir,
        },
        "max_duration": max_duration,
        "n_init_sims": n_init_sims,
        "generations": generations,
        "single_daughters": True,
        "suffix_time": False,
    }
    if "variants" not in base:
        upd["variants"] = variants_section
    base.update(upd)

    # Multi-condition: populate parca_variants
    if conditions:
        base["parca_variants"] = [
            {"rnaseq_basal_dataset_id": cond_id}
            for cond_id in conditions
        ]

    return base


def _build_variants_from_samples(
    X: np.ndarray,
    param_specs: list[SimDataParameter],
) -> dict[str, Any]:
    """Encode N germ samples as sim_data_setattr variants."""
    mutations_list = []
    for i in range(X.shape[0]):
        mutations = {}
        for j, spec in enumerate(param_specs):
            val = float(X[i, j])
            if hasattr(spec, "index") and spec.index is not None:
                mutations[spec.attr_path] = {"__index__": spec.index, "__value__": val}
            else:
                mutations[spec.attr_path] = val  # type: ignore[assignment]
        mutations_list.append(mutations)
    return {"sim_data_setattr": {"mutations": {"value": mutations_list}}}


def _x_from_variants(
    variants_block: dict[str, Any],
    param_specs: list[SimDataParameter],
) -> np.ndarray:
    """Inverse of `_build_variants_from_samples`: variants → X.

    Reads mutation values out of a user-supplied `sim_data_setattr` variants
    block and reconstructs the (n_variants, n_params) X matrix in the column
    order given by `param_specs`. Used by BYO-variants mode so that `quantify`
    can regress Y against the perturbations vEcoli actually applied.

    Raises:
        ValueError: if the variants block isn't `sim_data_setattr`, or any
            mutation entry is missing an `attr_path` that appears in
            `param_specs`. Non-`sim_data_setattr` modules can't be reverse-
            mapped (no scalar values to extract) — caller should refuse to
            cache and skip `quantify` rather than producing bogus Sobol.
    """
    if "sim_data_setattr" not in variants_block:
        modules = sorted(variants_block.keys())
        raise ValueError(
            f"_x_from_variants: variants block uses module(s) {modules!r}; "
            "only `sim_data_setattr` can be reverse-mapped to X (scalar "
            "mutation values per param). Other modules: skip --variants-source "
            "base-config and run without `quantify`."
        )
    mutations_list = variants_block["sim_data_setattr"]["mutations"]["value"]
    n_variants = len(mutations_list)
    n_params = len(param_specs)
    X = np.empty((n_variants, n_params), dtype=np.float64)
    for i, mutations in enumerate(mutations_list):
        for j, spec in enumerate(param_specs):
            if spec.attr_path not in mutations:
                raise ValueError(
                    f"_x_from_variants: mutation {i} missing attr_path "
                    f"'{spec.attr_path}' (param '{spec.name}'). Every variant "
                    "must specify every parameter for reverse-mapping."
                )
            raw = mutations[spec.attr_path]
            if isinstance(raw, dict) and "__index__" in raw:
                if spec.index is None or spec.index != raw["__index__"]:
                    raise ValueError(
                        f"_x_from_variants: mutation {i} for '{spec.attr_path}' "
                        f"uses index {raw['__index__']} but spec.index is "
                        f"{spec.index}. Index mismatch."
                    )
                X[i, j] = float(raw["__value__"])
            else:
                X[i, j] = float(raw)
    return X


def _pce_basis_size(order: int, n_params: int) -> int:
    """Number of total-order Legendre basis terms for the given (p, d).

    Matches PyTUQ's `get_mi(order, dim)` row count: C(p + d, d).
    """
    return comb(order + n_params, n_params)


def _pce_sample_adequacy(
    n_samples: int,
    n_params: int,
    order: int = 2,
) -> tuple[PceAdequacy, str]:
    """Judge whether `n_samples` is enough to fit a stable order-`order` PCE.

    Rules of thumb (per PyTUQ / UQPC practice):
        ratio = n_samples / basis_size
        ratio < 1   → underdetermined; least-squares fit is ill-posed
        ratio < 2   → marginal; high over-fit risk, Sobol indices unreliable
        ratio >= 2  → ok; >= 3 is comfortable

    Returns a (status, advisory) tuple. Caller decides whether to warn or
    block. Used by the BYO-variants CLI path, where `n_samples` is forced
    to the user's variants-list length and can't be chosen for PCE needs.
    """
    basis = _pce_basis_size(order, n_params)
    ratio = n_samples / basis if basis > 0 else 0.0
    rec_min = 2 * basis
    rec_good = 3 * basis
    if ratio < 1.0:
        return (
            "underdetermined",
            f"n_samples={n_samples} < basis_size={basis} for PCE order={order}, "
            f"{n_params} params. Least-squares is ill-posed. "
            f"Need at least {basis} samples; {rec_min}+ recommended. "
            f"Either add variants, drop --polynomial-order to {max(1, order - 1)} "
            f"(basis_size={_pce_basis_size(max(1, order - 1), n_params)}), "
            f"or use --regression bcs (sparse, tolerates fewer samples).",
        )
    if ratio < 2.0:
        return (
            "marginal",
            f"n_samples={n_samples} vs basis_size={basis} (ratio {ratio:.1f}×) "
            f"for PCE order={order}, {n_params} params. Fit will run but is "
            f"over-fit-prone; Sobol indices may be unstable. "
            f"{rec_min}+ samples recommended ({rec_good}+ comfortable). "
            f"Consider --regression bcs for sparser fits at this sample count.",
        )
    return (
        "ok",
        f"n_samples={n_samples} vs basis_size={basis} (ratio {ratio:.1f}×) — "
        f"adequate for PCE order={order}, {n_params} params.",
    )


@dataclasses.dataclass(frozen=True)
class ComputeAllocation:
    """A candidate (n_samples, n_init_sims, generations) triple with its
    PCE adequacy assessment at a given polynomial order and dimension.

    Under multi-condition planning (``n_conditions > 1``), ``n_samples`` is
    the *per-condition* sample count and the adequacy assessment applies to
    each condition's PCE fit individually. ``total_runs`` accounts for the
    full ``n_conditions × n_samples × n_init_sims × generations`` cost.
    """

    label: str  # e.g. "recommended", "halve generations", "single replicate"
    n_samples: int  # per condition under multi-condition mode
    n_init_sims: int
    generations: int
    total_runs: int  # n_conditions × n_samples × n_init_sims × generations
    basis_size: int
    adequacy_ratio: float  # n_samples / basis_size (per condition)
    adequacy_status: PceAdequacy
    noise_estimable: bool  # n_init_sims >= 2
    n_conditions: int = 1


def _recommend_compute_allocation(
    budget: int,
    n_params: int,
    polynomial_order: int = 2,
    min_replicates: int = 4,
    generations: int = 4,
    n_conditions: int = 1,
) -> list[ComputeAllocation]:
    """Recommend (n_samples, n_init_sims, generations) splits for a budget.

    Total vEcoli runs = ``n_conditions × n_samples × n_init_sims × generations``.
    Holding n_init_sims and generations fixed, n_samples scales the per-condition
    PCE adequacy ratio; pushing n_init_sims to 1 doubles n_samples but kills
    noise-floor estimability (Gap 2). Under multi-condition planning
    (``n_conditions > 1``, e.g. for ``uq sample --design-config``), the budget is
    divided evenly across conditions and the adequacy assessment applies per
    condition's PCE fit. The returned list contains the user-requested setting
    plus alternatives showing the trade-off.

    Args:
        budget: Total vEcoli runs available across all conditions.
        n_params: PCE input dimension (d).
        polynomial_order: PCE order at quantify time.
        min_replicates: Minimum n_init_sims required to estimate the noise
            floor (Triage 5). Default 4.
        generations: Target generations per variant (cell-cycle coverage).
            Default 4.
        n_conditions: Number of structural design conditions (``M`` from
            ``--design-config`` or ``--conditions``). Default 1 (single
            condition).

    Returns:
        A list of ``ComputeAllocation`` candidates, recommended first.
        The PyTUQ adequacy status (``ok`` / ``marginal`` / ``underdetermined``)
        on the recommended row tells the user whether their budget is
        enough for a stable per-condition PCE fit at the requested order.
    """
    basis = _pce_basis_size(polynomial_order, n_params)
    n_c = max(1, n_conditions)
    per_cond_budget = max(1, budget // n_c)

    def _make(label: str, n_s: int, n_i: int, gens: int) -> ComputeAllocation:
        status, _ = _pce_sample_adequacy(n_s, n_params, polynomial_order)
        return ComputeAllocation(
            label=label,
            n_samples=n_s,
            n_init_sims=n_i,
            generations=gens,
            total_runs=n_c * n_s * n_i * gens,
            basis_size=basis,
            adequacy_ratio=(n_s / basis) if basis > 0 else 0.0,
            adequacy_status=status,
            noise_estimable=(n_i >= 2),
            n_conditions=n_c,
        )

    candidates: list[ComputeAllocation] = []

    # Recommended: user's requested settings, n_samples derived from
    # per-condition budget.
    n_s_rec = max(1, per_cond_budget // max(1, min_replicates * generations))
    candidates.append(_make("recommended", n_s_rec, min_replicates, generations))

    # Alternative 1: halve generations → 2× n_samples → better PCE adequacy
    # but less cell-cycle coverage. Only emit if gens > 1.
    if generations > 1:
        alt_g = generations // 2
        n_s_alt = max(1, per_cond_budget // max(1, min_replicates * alt_g))
        candidates.append(_make("halve generations", n_s_alt, min_replicates, alt_g))

    # Alternative 2: single replicate → max n_samples but no noise floor.
    # Only emit if the recommended config wasn't already at min_replicates=1.
    if min_replicates > 1:
        n_s_alt = max(1, per_cond_budget // max(1, generations))
        candidates.append(_make("single replicate", n_s_alt, 1, generations))

    return candidates


def _design_matrix_conditioning(
    germ: np.ndarray,
    n_params: int,
    order: int = 2,
) -> tuple[float, DesignConditioning, str]:
    """Condition number κ(A) of the PCE design matrix at the given order.

    PyTUQ-native joint design-quality measure: builds an output PCRV with
    a total-order Legendre basis at the given order, evaluates the basis
    matrix ``A = pcrv.evalBases(germ, 0)``, and returns ``κ(A)`` via
    ``np.linalg.cond``. κ subsumes marginal uniformity (a clumped sample
    set produces near-collinear basis evaluations regardless of marginal
    distributions) and complements ``_pce_sample_adequacy``: count
    catches ``N < B``; κ catches ``N >= B`` with bad geometry.

    Thresholds (log10(κ) ≈ digits of precision lost in coefficient fit):

    * κ ≤ 100      → ``well-conditioned``
    * κ ≤ 1e4      → ``marginal`` (Sobol indices noisy but usable)
    * κ > 1e4      → ``ill-conditioned`` (results suspect)
    * non-finite   → ``singular`` (typically caught by the count check first)

    Warn-only semantics — PyTUQ's PCE fits don't refuse on high κ, they
    just propagate the noise. The user decides whether to proceed.
    """
    from pytuq.rv.pcrv import PCRV  # type: ignore[import-untyped]
    from pytuq.utils.mindex import get_mi  # type: ignore[import-untyped]

    mi = get_mi(order, n_params)
    pc = PCRV(n_params, 1, "LU", mi=mi)
    A = pc.evalBases(germ, 0)
    n_rows, n_cols = A.shape

    # Rank-deficient by shape: more basis terms than samples → no unique
    # least-squares solution. np.linalg.cond on a wide matrix uses the
    # pseudo-inverse and reports a *small* number over the few nonzero
    # singular values, which is misleading — handle this case explicitly.
    if n_rows < n_cols:
        return (
            float("inf"),
            "singular",
            f"Design matrix is rank-deficient: {n_rows} samples < "
            f"{n_cols} basis terms. No unique PCE coefficient vector "
            "exists. Caught by the count check too; add samples or "
            "drop --polynomial-order.",
        )

    kappa = float(np.linalg.cond(A))

    if not np.isfinite(kappa):
        return (
            kappa,
            "singular",
            "Design matrix is singular (κ = ∞). The basis evaluations at "
            "your germ samples are linearly dependent — possibly duplicate "
            "rows or a degenerate sample geometry (all samples on a "
            "lower-dimensional subspace of the box).",
        )
    if kappa <= 100.0:
        return (
            kappa,
            "well-conditioned",
            f"κ(A) = {kappa:.1f}  (≤ 100) — design matrix is well-conditioned "
            f"at PCE order={order}, {n_params} params.",
        )
    if kappa <= 1e4:
        return (
            kappa,
            "marginal",
            f"κ(A) = {kappa:.1f} for PCE order={order}, {n_params} params. "
            "Sobol indices will be noisier than ideal but usable; "
            f"~{int(np.log10(kappa))} significant digits of precision lost "
            "in the coefficient fit. Consider a more uniform sample geometry "
            "or --regression bcs for sparser, better-conditioned fits.",
        )
    return (
        kappa,
        "ill-conditioned",
        f"κ(A) = {kappa:.2e} for PCE order={order}, {n_params} params. "
        "Basis matrix is poorly conditioned — least-squares amplifies any Y "
        f"noise by ~{kappa:.0e}× into the coefficients, so Sobol indices "
        "are unreliable. Either: (a) add variants that fill the box more "
        "uniformly, (b) drop --polynomial-order to reduce the basis size, "
        "or (c) use --regression bcs (sparse fits tolerate worse "
        "conditioning).",
    )


def _design_condition_id(
    module_name: str,
    index: int,
    params: dict[str, Any],
) -> str:
    """Deterministic, human-readable ID for one design point.

    Format: ``<module>_<idx>_<short_param_digest>`` so condition directories
    sort naturally by index while still distinguishing same-index points
    across reruns with different params.

    Example: ``mecillinam_timeline_0003_a1b2c3`` for design entry 3.
    """
    import hashlib
    digest = hashlib.sha1(
        repr(sorted(params.items())).encode("utf-8")
    ).hexdigest()[:6]
    return f"{module_name}_{index:04d}_{digest}"


def _apply_design_to_baseline(
    sim_data: Any,
    module_name: str,
    params: dict[str, Any],
    module_override: str | None = None,
) -> Any:
    """Apply one design-variant entry to a deepcopied baseline sim_data.

    vEcoli's ``runscripts/create_variants.py`` rejects multi-module variants
    blocks ("Only one variant name allowed"), so we cannot stack design +
    sim_data_setattr in one workflow. Instead we pre-bake the design layer
    out-of-band: this helper applies the design's ``apply_variant`` to a
    deepcopied baseline, returning the modified ``sim_data`` ready to be
    pickled and pointed at by a downstream UQ vEcoli run.

    Caller is responsible for ``copy.deepcopy(sim_data)`` before calling if
    they need the baseline preserved — this function mutates in place per
    vEcoli's ``apply_variant`` contract.
    """
    from uq.convert_variants import _resolve_apply_variant
    apply_variant = _resolve_apply_variant(module_name, module_override)
    result = apply_variant(sim_data, params)
    return result if result is not None else sim_data


def _design_layer_attr_paths(
    baseline: Any,
    module_name: str,
    design_entries: list[dict[str, Any]],
    module_override: str | None = None,
) -> set[str]:
    """Set of dot-paths the design layer touches across all its entries.

    Used by the overlap safety check in ``uq sample --design-config``:
    warn (or refuse) when a user's ``--params-file`` lists attr_paths that
    the design layer also writes — the UQ-sampled value would silently
    overwrite the design's value.

    Uses the same ``_walk_diff`` machinery as ``convert-variants``.
    """
    import copy as _copy
    from uq.convert_variants import _walk_diff

    paths: set[str] = set()
    for entry in design_entries:
        baseline_copy = _copy.deepcopy(baseline)
        mutated = _apply_design_to_baseline(
            _copy.deepcopy(baseline), module_name, entry, module_override,
        )
        for diff_entry in _walk_diff(baseline_copy, mutated, include_arrays=True):
            paths.add(diff_entry.attr_path)
    return paths


def _check_oob_variants(
    X: np.ndarray,
    bounds: np.ndarray,
    param_names: list[str],
) -> list[tuple[int, str, float, float, float]]:
    """Return (variant_idx, param_name, value, lb, ub) tuples for OOB cells.

    PyTUQ paradigm: ``pdom`` is a hard constraint on the parameter space,
    not a soft prior. vEcoli paradigm: workflow configs fail loudly on
    invalid fields rather than auto-repairing. So this is the hard
    correctness check that lives between sample generation and execution
    — called in both BYO and default modes (cheap; PCRV.evalPC should
    never produce OOB under the default sampler, but we still verify).

    An empty return value means every (variant, parameter) cell is within
    ``[lb, ub]``. Caller is expected to hard-fail on any non-empty list
    and direct the user to widen ``--params-file`` bounds.
    """
    violations: list[tuple[int, str, float, float, float]] = []
    for i in range(X.shape[0]):
        for j, name in enumerate(param_names):
            v = float(X[i, j])
            lb, ub = float(bounds[j, 0]), float(bounds[j, 1])
            if v < lb or v > ub:
                violations.append((i, name, v, lb, ub))
    return violations


def _germ_from_physical(X: np.ndarray, bounds: np.ndarray) -> np.ndarray:
    """Affine inverse of the input PCRV's evalPC: physical → germ in [-1, 1].

    Mirrors `uq.workflow._physical_to_germ` but lives here to keep the BYO
    path free of `uq.workflow` imports. Values outside the bound box map
    outside [-1, 1] — caller should warn rather than clip, since clipping
    silently distorts the input distribution `quantify` assumes.
    """
    lb, ub = bounds[:, 0], bounds[:, 1]
    span = ub - lb
    span = np.where(span > 0, span, 1.0)
    return 2.0 * (X - lb) / span - 1.0  # type: ignore[no-any-return]


def _load_variants_from_base_config(base_config_path: str) -> dict[str, Any]:
    """Read a vEcoli config JSON and return its top-level `variants` block.

    Raises:
        ValueError: if the file has no `variants` key. BYO mode is a no-op
            without one; better to fail fast than fall back to PCRV silently.
    """
    base = json.loads(Path(base_config_path).read_text())
    if "variants" not in base:
        raise ValueError(
            f"--variants-source base-config: {base_config_path} has no "
            "'variants' key. Add one, or drop --variants-source to use "
            "PCRV sampling from --params-file."
        )
    return base["variants"]  # type: ignore[no-any-return]


def _count_completed_variants(history_base: Path, _n_expected: int = 0) -> int:
    """Count how many variant directories have Parquet files.

    Handles vEcoli's hive-partitioned output structure:
    history/experiment_id=X/variant=N/lineage_seed=S/generation=G/agent_id=A/NNN.pq
    """
    if not history_base.exists():
        return 0
    found_variants: set[int] = set()
    for pq in history_base.rglob("*.pq"):
        for part in pq.parts:
            if part.startswith("variant="):
                with contextlib.suppress(ValueError):
                    found_variants.add(int(part.split("=")[1]))
                break
    return len(found_variants)


def _collect_variant_timeseries(
    history_base: Path,
    n_samples: int,
    obs_columns: list[str],
) -> tuple[np.ndarray, list[np.ndarray], list[dict[str, np.ndarray]]]:
    """Read hive-partitioned Parquet and extract per-variant arrays."""
    pq_files = list(history_base.rglob("*.pq"))
    if not pq_files:
        pq_files = list(history_base.rglob("*.parquet"))
    if not pq_files:
        raise FileNotFoundError(f"No Parquet files under {history_base}")

    df = pl.read_parquet([str(p) for p in pq_files], hive_partitioning=True)

    available = [c for c in obs_columns if c in df.columns]
    if not available:
        raise ValueError(f"None of {obs_columns} found in {df.columns[:20]}")

    sort_cols = [c for c in ["variant", "lineage_seed", "generation", "time"] if c in df.columns]
    if sort_cols:
        df = df.sort(sort_cols)

    baseline_df = df.filter(pl.col("variant") == 0) if "variant" in df.columns else None

    Y_list, Y_ts, Y_meta = [], [], []
    for i in range(n_samples):
        vidx = i + 1
        if "variant" in df.columns:
            sdf = df.filter(pl.col("variant") == vidx)
            if sdf.height == 0:
                sdf = df.filter(pl.col("variant") == i)
            if sdf.height == 0 and baseline_df is not None:
                sdf = baseline_df
        else:
            sdf = df

        obs_df = sdf.select(available).fill_null(0.0)
        ts = obs_df.to_numpy().astype(np.float64)
        Y_ts.append(ts)
        Y_list.append(ts.mean(axis=0))

        meta: dict[str, np.ndarray] = {}
        if "generation" in sdf.columns:
            meta["generation"] = sdf["generation"].fill_null(0).to_numpy().astype(np.int64)
        if "lineage_seed" in sdf.columns:
            meta["lineage_seed"] = sdf["lineage_seed"].fill_null(0).to_numpy().astype(np.int64)
        Y_meta.append(meta)

    Y_agg = np.vstack(Y_list)
    has_meta = Y_meta and any(m for m in Y_meta)
    return Y_agg, Y_ts, Y_meta if has_meta else None  # type: ignore[return-value]
