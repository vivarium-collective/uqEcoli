"""Observable extraction presets matching vEcoli's ``cd1_`` analysis modules.

Each preset knows which Parquet columns to read, whether they are scalar or
array-typed, what normalization to apply, and how to name the output features.
Presets can be composed: ``--observables mass exchange_fluxes`` extracts both
and concatenates them column-wise into Y.

The presets mirror the six ``cd1_*`` modules in
``vEcoli-private/ecoli/analysis/multiseed/`` so that the UQ pipeline analyses
the *same* derived outputs that CD1 evaluated against experimental data.

Usage
-----
>>> from uq.observables import collect_observables, PRESETS
>>> Y, names, ts, meta = collect_observables(
...     history_base, n_variants=20,
...     presets=["mass", "exchange_fluxes"],
...     generation_lower_bound=2,
... )
"""

from __future__ import annotations

import fnmatch
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════
# Preset definitions
# ═══════════════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class ObservablePreset:
    """Definition of one observable category."""

    name: str
    description: str
    columns: list[str]
    is_array: bool = False
    column_pattern: str | None = None  # fnmatch pattern for dynamic column discovery
    normalize_by_dry_mass: bool = False
    has_derived: bool = False  # if True, _compute_derived() adds extra columns


PRESETS: dict[str, ObservablePreset] = {
    # ── Scalar presets ────────────────────────────────────────────────
    "mass": ObservablePreset(
        name="mass",
        description="Raw mass/growth scalars",
        columns=[
            "listeners__mass__dry_mass",
            "listeners__mass__cell_mass",
            "listeners__mass__volume",
            "listeners__mass__growth",
            "listeners__mass__instantaneous_growth_rate",
        ],
    ),
    "higher_order": ObservablePreset(
        name="higher_order",
        description="Derived cell properties — cd1_higher_order_properties "
        "(doubling time, DNA/RNA fractions, biomass composition)",
        columns=[
            "listeners__mass__dry_mass",
            "listeners__mass__cell_mass",
            "listeners__mass__volume",
            "listeners__mass__dna_mass",
            "listeners__mass__rna_mass",
            "listeners__mass__instantaneous_growth_rate",
        ],
        has_derived=True,
    ),
    # ── Array presets (cd1 analysis modules) ──────────────────────────
    "exchange_fluxes": ObservablePreset(
        name="exchange_fluxes",
        description="External exchange fluxes — cd1_exchange_fluxes",
        columns=["listeners__fba_results__external_exchange_fluxes"],
        is_array=True,
    ),
    "transcriptome": ObservablePreset(
        name="transcriptome",
        description="mRNA cistron counts ~4300 genes — cd1_transcriptomics",
        columns=["listeners__rna_counts__mRNA_cistron_counts"],
        is_array=True,
    ),
    "proteome": ObservablePreset(
        name="proteome",
        description="Protein monomer counts ~4300 monomers — cd1_proteomics",
        columns=["listeners__monomer_counts"],
        is_array=True,
    ),
    "fluxome": ObservablePreset(
        name="fluxome",
        description="Base reaction fluxes ~2800 reactions — cd1_fluxomics "
        "(normalized by dry mass)",
        columns=["listeners__fba_results__base_reaction_fluxes"],
        is_array=True,
        normalize_by_dry_mass=True,
    ),
    "reaction_fluxes": ObservablePreset(
        name="reaction_fluxes",
        description="All reaction fluxes incl. enzyme kinetics",
        columns=["listeners__fba_results__reaction_fluxes"],
        is_array=True,
    ),
}

ALL_PRESET_NAMES = list(PRESETS.keys())
DEFAULT_PRESETS = ["mass"]


# ═══════════════════════════════════════════════════════════════════════
# Extraction
# ═══════════════════════════════════════════════════════════════════════


def _read_parquet(history_base: Path) -> pl.DataFrame:
    """Read all Parquet files under a hive-partitioned history directory."""
    pq_files = list(history_base.rglob("*.pq"))
    if not pq_files:
        pq_files = list(history_base.rglob("*.parquet"))
    if not pq_files:
        raise FileNotFoundError(f"No Parquet files under {history_base}")
    return pl.read_parquet([str(p) for p in pq_files], hive_partitioning=True)


def _resolve_columns(
    preset: ObservablePreset,
    all_columns: list[str],
) -> list[str]:
    """Resolve a preset's columns against the actual Parquet schema."""
    if preset.column_pattern:
        matched = fnmatch.filter(all_columns, preset.column_pattern)
        return sorted(matched)
    return [c for c in preset.columns if c in all_columns]


def _compute_derived(
    sdf: pl.DataFrame,
) -> tuple[np.ndarray, list[str]]:
    """Compute cd1_higher_order_properties-style derived metrics.

    Derived per-row (before time-averaging) so that the mean of the
    ratio is taken, matching the cd1 aggregation pattern.

    Returns:
        (n_rows, n_derived) array + feature names.
    """
    n = sdf.height
    cols: list[np.ndarray] = []
    names: list[str] = []

    # Growth rate in per-hour (public-repo metric)
    if "listeners__mass__instantaneous_growth_rate" in sdf.columns:
        gr = sdf["listeners__mass__instantaneous_growth_rate"].fill_null(0.0).to_numpy().astype(np.float64)
        gr_h = gr * 3600.0
        cols.append(gr_h)
        names.append("growth_rate_per_hour")

        # Doubling time (hours) — derived from mean growth rate, not
        # mean of per-timestep 1/μ (avoids 1/ε outliers at t=0)
        mean_gr = float(np.mean(gr_h))
        dt_h = np.log(2) / mean_gr if mean_gr > 1e-6 else np.nan
        cols.append(np.full(n, dt_h))
        names.append("doubling_time_hours")

    # DNA fraction (g DNA / g dry weight)
    if "listeners__mass__dna_mass" in sdf.columns and "listeners__mass__dry_mass" in sdf.columns:
        dna = sdf["listeners__mass__dna_mass"].fill_null(0.0).to_numpy().astype(np.float64)
        dry = sdf["listeners__mass__dry_mass"].fill_null(1.0).to_numpy().astype(np.float64)
        dry_safe = np.where(dry > 0, dry, 1.0)
        cols.append(dna / dry_safe)
        names.append("dna_fraction_g_per_g_dw")

    # RNA fraction (g RNA / g dry weight)
    if "listeners__mass__rna_mass" in sdf.columns and "listeners__mass__dry_mass" in sdf.columns:
        rna = sdf["listeners__mass__rna_mass"].fill_null(0.0).to_numpy().astype(np.float64)
        dry = sdf["listeners__mass__dry_mass"].fill_null(1.0).to_numpy().astype(np.float64)
        dry_safe = np.where(dry > 0, dry, 1.0)
        cols.append(rna / dry_safe)
        names.append("rna_fraction_g_per_g_dw")

    # Dry mass fraction
    if "listeners__mass__dry_mass" in sdf.columns and "listeners__mass__cell_mass" in sdf.columns:
        dry = sdf["listeners__mass__dry_mass"].fill_null(0.0).to_numpy().astype(np.float64)
        cell = sdf["listeners__mass__cell_mass"].fill_null(1.0).to_numpy().astype(np.float64)
        cell_safe = np.where(cell > 0, cell, 1.0)
        cols.append(dry / cell_safe)
        names.append("dry_mass_fraction")

    # Cell volume (already scalar, but we include it here for the
    # higher_order preset's completeness)
    if "listeners__mass__volume" in sdf.columns:
        vol = sdf["listeners__mass__volume"].fill_null(0.0).to_numpy().astype(np.float64)
        cols.append(vol)
        names.append("cell_volume_um3")

    if not cols:
        return np.zeros((n, 0)), []

    return np.column_stack(cols), names


def _extract_scalar_columns(
    sdf: pl.DataFrame,
    columns: list[str],
) -> tuple[np.ndarray, list[str]]:
    """Extract scalar columns → (n_rows, n_cols) array + column names."""
    available = [c for c in columns if c in sdf.columns]
    if not available:
        return np.zeros((sdf.height, 0)), []
    obs_df = sdf.select(available).fill_null(0.0)
    return obs_df.to_numpy().astype(np.float64), available


def _extract_array_column(
    sdf: pl.DataFrame,
    column: str,
    normalize_values: np.ndarray | None = None,
) -> tuple[np.ndarray, list[str]]:
    """Extract one array/list column → (n_rows, array_len) + feature names.

    Each row in a list column becomes one row in the output.  The column
    ``listeners__rna_counts__mRNA_cistron_counts`` with ~2000 genes per
    row becomes a ``(n_rows, 2000)`` array.
    """
    if column not in sdf.columns:
        return np.zeros((sdf.height, 0)), []

    series = sdf[column]

    # Convert list-of-lists to a stacked numpy array
    try:
        rows = series.to_list()
        arr = np.vstack([np.asarray(r, dtype=np.float64) for r in rows if r is not None])
    except Exception as e:
        logger.warning("Failed to unpack array column %s: %s", column, e)
        return np.zeros((sdf.height, 0)), []

    if normalize_values is not None and normalize_values.shape[0] == arr.shape[0]:
        denom = normalize_values.copy()
        denom[denom == 0] = 1.0
        arr = arr / denom[:, np.newaxis]

    n_features = arr.shape[1]
    short = column.split("__")[-1]
    names = [f"{short}_{i}" for i in range(n_features)]

    return arr, names


def _extract_preset(
    sdf: pl.DataFrame,
    preset: ObservablePreset,
    resolved_columns: list[str],
) -> tuple[np.ndarray, list[str]]:
    """Extract one preset from a per-variant sub-DataFrame."""
    if preset.has_derived:
        # Derived preset — compute cd1-style ratios and metrics
        return _compute_derived(sdf)
    elif preset.is_array:
        # Array columns: extract each and concatenate
        arrays, names = [], []
        for col in resolved_columns:
            # Optional dry-mass normalization (cd1_fluxomics style)
            norm = None
            if preset.normalize_by_dry_mass and "listeners__mass__dry_mass" in sdf.columns:
                norm = sdf["listeners__mass__dry_mass"].fill_null(1.0).to_numpy().astype(np.float64)
            arr, n = _extract_array_column(sdf, col, normalize_values=norm)
            if arr.shape[1] > 0:
                arrays.append(arr)
                names.extend(n)
        if not arrays:
            return np.zeros((sdf.height, 0)), []
        return np.hstack(arrays), names
    else:
        return _extract_scalar_columns(sdf, resolved_columns)


# ═══════════════════════════════════════════════════════════════════════
# Dimension guard
# ═══════════════════════════════════════════════════════════════════════


def _align_variant_dimensions(
    Y_list: list[np.ndarray],
    Y_ts: list[np.ndarray],
    observable_names: list[str],
    strict: bool = True,
) -> tuple[np.ndarray, list[str]]:
    """Validate that all variants have the same observable count.

    In strict mode (default, RFC006-compliant), raises an error on
    dimension mismatch — all variants within a single condition must
    have identical observable shape.

    In non-strict mode (multi-condition), truncates to the minimum
    common dimension and warns.  This is needed when different ParCa
    datasets produce slightly different gene sets.

    Args:
        Y_list: Per-variant time-averaged arrays, each shape ``(n_obs,)``.
        Y_ts: Per-variant raw timeseries (mutated in-place if truncated).
        observable_names: Feature names corresponding to the first variant.
        strict: If True, raise on mismatch. If False, truncate and warn.

    Returns:
        ``(Y_agg, observable_names)`` — possibly truncated to common dimension.
    """
    widths = [arr.shape[0] for arr in Y_list]
    min_w = min(widths)
    max_w = max(widths)

    if min_w != max_w:
        if strict:
            raise ValueError(
                f"Observable dimension mismatch across variants: min={min_w}, max={max_w}. "
                f"All variants must produce the same observable count within a single condition. "
                f"If this is a multi-condition run with different ParCa datasets, "
                f"use strict=False to truncate to the common dimension."
            )
        logger.warning(
            "Observable dimension mismatch across variants: min=%d, max=%d. "
            "Truncating to common dimension %d. "
            "This can happen when different ParCa datasets produce different gene sets.",
            min_w,
            max_w,
            min_w,
        )
        for i in range(len(Y_list)):
            Y_list[i] = Y_list[i][:min_w]
        for i in range(len(Y_ts)):
            Y_ts[i] = Y_ts[i][:, :min_w]
        observable_names = observable_names[:min_w]

    return np.vstack(Y_list), observable_names


# ═══════════════════════════════════════════════════════════════════════
# Public API
# ═══════════════════════════════════════════════════════════════════════


def collect_observables(
    history_base: Path,
    n_variants: int,
    presets: list[str] | None = None,
    generation_lower_bound: int = 0,
    time_lower_bound: float = 0.0,
    strict: bool = True,
) -> tuple[np.ndarray, list[str], list[np.ndarray], list[dict[str, np.ndarray]] | None]:
    """Extract per-variant observables from hive-partitioned Parquet.

    Drop-in replacement for ``uq.tui._collect_variant_timeseries`` with
    support for multiple observable presets and generation filtering.

    Args:
        history_base: Path to the ``history/`` directory.
        n_variants: Number of variants to extract (variant indices 1..N).
        presets: List of preset names (see :data:`PRESETS`).
            Defaults to ``["mass"]``.
        generation_lower_bound: Skip generations below this value
            (cd1 ``generation_lower_bound`` parameter).  0 = keep all.
        time_lower_bound: Skip timesteps below this value (seconds).
            0 = keep all.

    Returns:
        ``(Y_agg, observable_names, Y_timeseries, Y_meta)``
        matching the signature of ``_collect_variant_timeseries``.

        - ``Y_agg``: ``(n_variants, n_observables)`` time-averaged.
        - ``observable_names``: length-``n_observables`` list of feature
          labels.
        - ``Y_timeseries``: per-variant raw arrays for strategies 2-4.
        - ``Y_meta``: per-variant generation/seed labels (or None).
    """
    preset_names = presets or DEFAULT_PRESETS
    preset_objs = []
    for name in preset_names:
        if name not in PRESETS:
            raise ValueError(
                f"Unknown observable preset {name!r}. "
                f"Available: {ALL_PRESET_NAMES}"
            )
        preset_objs.append(PRESETS[name])

    df = _read_parquet(history_base)

    # ── Resolve columns against actual schema ──
    all_cols = df.columns
    resolved: list[tuple[ObservablePreset, list[str]]] = []
    for p in preset_objs:
        cols = _resolve_columns(p, all_cols)
        if not cols:
            logger.warning("Preset %r: no matching columns in Parquet (available: %s)", p.name, all_cols[:10])
        resolved.append((p, cols))

    # For array presets: also ensure dry_mass is loaded if normalization needed
    needs_dry_mass = any(p.normalize_by_dry_mass for p, _ in resolved)
    extra_cols = []
    if needs_dry_mass and "listeners__mass__dry_mass" not in all_cols:
        logger.warning("dry_mass column missing — flux normalization will be skipped")
    elif needs_dry_mass:
        extra_cols.append("listeners__mass__dry_mass")

    # ── Apply generation / time filtering (cd1 pattern) ──
    sort_cols = [c for c in ["variant", "lineage_seed", "generation", "time"] if c in df.columns]
    if sort_cols:
        df = df.sort(sort_cols)

    if generation_lower_bound > 0 and "generation" in df.columns:
        before = df.height
        df = df.filter(pl.col("generation") >= generation_lower_bound)
        logger.info(
            "Generation filter (>= %d): %d → %d rows",
            generation_lower_bound,
            before,
            df.height,
        )

    if time_lower_bound > 0 and "time" in df.columns:
        df = df.filter(pl.col("time") >= time_lower_bound)

    # ── Per-variant extraction ──
    baseline_df = df.filter(pl.col("variant") == 0) if "variant" in df.columns else None
    observable_names: list[str] | None = None

    Y_list: list[np.ndarray] = []
    Y_ts: list[np.ndarray] = []
    Y_meta_list: list[dict[str, np.ndarray]] = []

    for i in range(n_variants):
        vidx = i + 1
        if "variant" in df.columns:
            sdf = df.filter(pl.col("variant") == vidx)
            if sdf.height == 0:
                sdf = df.filter(pl.col("variant") == i)
            if sdf.height == 0 and baseline_df is not None:
                sdf = baseline_df
        else:
            sdf = df

        # Extract each preset and concatenate
        arrays: list[np.ndarray] = []
        names: list[str] = []
        for preset, cols in resolved:
            arr, n = _extract_preset(sdf, preset, cols)
            if arr.shape[1] > 0:
                arrays.append(arr)
                names.extend(n)

        if not arrays:
            logger.warning("Variant %d: no observables extracted", vidx)
            continue

        ts = np.hstack(arrays)
        Y_ts.append(ts)
        Y_list.append(ts.mean(axis=0))

        if observable_names is None:
            observable_names = names

        # Generation / seed metadata (for strategies 2-3)
        meta: dict[str, np.ndarray] = {}
        if "generation" in sdf.columns:
            meta["generation"] = sdf["generation"].fill_null(0).to_numpy().astype(np.int64)
        if "lineage_seed" in sdf.columns:
            meta["lineage_seed"] = sdf["lineage_seed"].fill_null(0).to_numpy().astype(np.int64)
        Y_meta_list.append(meta)

    if not Y_list:
        raise RuntimeError(
            f"No observables collected from {history_base} for {n_variants} variants. "
            f"Presets: {preset_names}"
        )

    # ── Dimension guard: detect column mismatches across variants ──
    Y_agg, observable_names = _align_variant_dimensions(
        Y_list, Y_ts, observable_names or [], strict=strict,
    )
    has_meta = Y_meta_list and any(m for m in Y_meta_list)

    logger.info(
        "Collected: Y=%s, %d features (%s), gen_lower_bound=%d",
        Y_agg.shape,
        len(observable_names or []),
        "+".join(preset_names),
        generation_lower_bound,
    )

    return (
        Y_agg,
        observable_names or [],
        Y_ts,
        Y_meta_list if has_meta else None,  # type: ignore[return-value]
    )


def collect_baseline_observables(
    history_base: Path,
    presets: list[str] | None = None,
    generation_lower_bound: int = 0,
) -> tuple[np.ndarray, list[str], np.ndarray, dict[str, np.ndarray]]:
    """Extract variant=0 (baseline) observables for Q1 variance decomposition.

    Unlike ``collect_observables`` (which collects variants 1..N), this
    collects the single baseline variant with its full per-timestep
    generation/seed structure — needed for σ² budget computation.

    Returns:
        ``(Y_agg, observable_names, Y_timeseries, Y_meta)``

        - ``Y_agg``: time-averaged baseline, shape ``(n_obs,)``.
        - ``observable_names``: feature labels.
        - ``Y_timeseries``: raw array, shape ``(n_timesteps, n_obs)``.
        - ``Y_meta``: dict with ``generation`` and ``lineage_seed`` arrays.
    """
    preset_names = presets or DEFAULT_PRESETS
    preset_objs = []
    for name in preset_names:
        if name not in PRESETS:
            raise ValueError(f"Unknown observable preset {name!r}. Available: {ALL_PRESET_NAMES}")
        preset_objs.append(PRESETS[name])

    df = _read_parquet(history_base)

    # Resolve columns
    all_cols = df.columns
    resolved: list[tuple[ObservablePreset, list[str]]] = []
    for p in preset_objs:
        cols = _resolve_columns(p, all_cols)
        resolved.append((p, cols))

    # Sort and filter to variant=0
    sort_cols = [c for c in ["variant", "lineage_seed", "generation", "time"] if c in df.columns]
    if sort_cols:
        df = df.sort(sort_cols)

    if "variant" in df.columns:
        df = df.filter(pl.col("variant") == 0)
    if df.height == 0:
        raise RuntimeError(f"No data for variant=0 (baseline) in {history_base}")

    if generation_lower_bound > 0 and "generation" in df.columns:
        df = df.filter(pl.col("generation") >= generation_lower_bound)

    # Extract preset columns
    arrays: list[np.ndarray] = []
    names: list[str] = []
    for preset, cols in resolved:
        arr, n = _extract_preset(df, preset, cols)
        if arr.shape[1] > 0:
            arrays.append(arr)
            names.extend(n)

    if not arrays:
        raise RuntimeError(f"No observables extracted for baseline from {history_base}")

    ts = np.hstack(arrays)
    Y_agg = ts.mean(axis=0)

    meta: dict[str, np.ndarray] = {}
    if "generation" in df.columns:
        meta["generation"] = df["generation"].fill_null(0).to_numpy().astype(np.int64)
    if "lineage_seed" in df.columns:
        meta["lineage_seed"] = df["lineage_seed"].fill_null(0).to_numpy().astype(np.int64)

    return Y_agg, names, ts, meta


def list_presets() -> str:
    """Return a human-readable summary of all presets."""
    lines = []
    for name, p in PRESETS.items():
        cols = ", ".join(p.columns[:3]) if p.columns else f"pattern: {p.column_pattern}"
        tag = " [array]" if p.is_array else ""
        lines.append(f"  {name:<20s} {p.description}{tag}")
    return "\n".join(lines)
