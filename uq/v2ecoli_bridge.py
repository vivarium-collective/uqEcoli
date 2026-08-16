"""Observable path mapping between v2ecoli state and UQ pipeline names.

Bridges the gap between:

- **v2ecoli state paths**: Nested dict access into the composite state,
  e.g. ``composite.state['agents']['0']['listeners']['mass']['dry_mass']``
- **UQ observable names**: Dot-path strings used throughout the UQ pipeline,
  e.g. ``listeners__mass__dry_mass``

Two types of observables:
  1. **Scalar** — one state path → one float value (mass, higher_order)
  2. **Array** — one state path → one numpy array, flattened to
     ``{preset}_{index}`` columns (transcriptome ~4300 genes,
     proteome ~4300 monomers, fluxome ~2800 reactions,
     exchange_fluxes ~87 metabolites).
"""

from __future__ import annotations

from typing import Any

import numpy as np

# ═══════════════════════════════════════════════════════════════════════
# Scalar state path → observable name mapping
# ═══════════════════════════════════════════════════════════════════════

# Each entry maps a UQ observable name (dot-path, same as Parquet column name)
# to a list of nested dict keys for v2ecoli composite state traversal.
# The path is relative to a single agent's cell state, e.g.:
#   composite.state['agents'][agent_id]  →  cell_state
#   cell_state['listeners']['mass']['dry_mass']

OBSERVABLE_PATHS: dict[str, list[str]] = {
    # ── Mass scalars ─────────────────────────────────────────────────
    "listeners__mass__dry_mass": ["listeners", "mass", "dry_mass"],
    "listeners__mass__cell_mass": ["listeners", "mass", "cell_mass"],
    "listeners__mass__volume": ["listeners", "mass", "volume"],
    "listeners__mass__growth": ["listeners", "mass", "growth"],
    "listeners__mass__instantaneous_growth_rate": [
        "listeners", "mass", "instantaneous_growth_rate"
    ],
    "listeners__mass__protein_mass": ["listeners", "mass", "protein_mass"],
    "listeners__mass__rna_mass": ["listeners", "mass", "rna_mass"],
    "listeners__mass__dna_mass": ["listeners", "mass", "dna_mass"],
    "listeners__mass__smallMolecule_mass": [
        "listeners", "mass", "smallMolecule_mass"
    ],
    "listeners__mass__water_mass": ["listeners", "mass", "water_mass"],

    # ── Growth / division ────────────────────────────────────────────
    "listeners__mass__rRna_mass": ["listeners", "mass", "rRna_mass"],
    "listeners__mass__tRna_mass": ["listeners", "mass", "tRna_mass"],
    "listeners__mass__mRna_mass": ["listeners", "mass", "mRna_mass"],
    "divide": ["divide"],
    "global_time": ["global_time"],
}

# ═══════════════════════════════════════════════════════════════════════
# Array observable paths — each maps a preset name to a v2ecoli
# state path pointing to a numpy array.
# ═══════════════════════════════════════════════════════════════════════

# Array observables are flattened to individual columns during extraction:
# e.g. ``transcriptome_0``, ``transcriptome_1``, …, ``transcriptome_N``
# for each element of the array at that state path.

ARRAY_OBSERVABLE_PATHS: dict[str, list[str]] = {
    "transcriptome": [
        "listeners", "rna_counts", "mRNA_cistron_counts"
    ],
    "proteome": [
        "listeners", "monomer_counts"
    ],
    "fluxome": [
        "listeners", "fba_results", "base_reaction_fluxes"
    ],
    "exchange_fluxes": [
        "listeners", "fba_results", "external_exchange_fluxes"
    ],
}

# ═══════════════════════════════════════════════════════════════════════
# Preset composition — maps UQ observable preset names to the
# observable names they comprise (mirrors ``uq/observables.py``).
# ═══════════════════════════════════════════════════════════════════════

# Scalar presets list their individual column names.
# Array presets have an empty list — extraction uses ``ARRAY_OBSERVABLE_PATHS``
# and flattens the array into ``{preset}_{index}`` columns at runtime.

SCALAR_PRESETS = frozenset({"mass", "higher_order"})
ARRAY_PRESETS = frozenset(ARRAY_OBSERVABLE_PATHS.keys())
ALL_PRESETS = SCALAR_PRESETS | ARRAY_PRESETS

PRESET_MAP: dict[str, list[str]] = {
    "mass": [
        "listeners__mass__dry_mass",
        "listeners__mass__cell_mass",
        "listeners__mass__volume",
        "listeners__mass__growth",
        "listeners__mass__instantaneous_growth_rate",
    ],
    "higher_order": [
        "listeners__mass__dry_mass",
        "listeners__mass__cell_mass",
        "listeners__mass__volume",
        "listeners__mass__dna_mass",
        "listeners__mass__rna_mass",
        "listeners__mass__instantaneous_growth_rate",
    ],
    # Array presets — empty list signals runtime flattening
    "transcriptome": [],
    "proteome": [],
    "fluxome": [],
    "exchange_fluxes": [],
}

# ═══════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════


def get_state_path(observable_name: str) -> list[str] | None:
    """Return the v2ecoli state path for a UQ observable name.

    Args:
        observable_name: UQ dot-path name
            (e.g. ``listeners__mass__dry_mass``).

    Returns:
        List of nested dict keys, or ``None`` if not found.
    """
    return OBSERVABLE_PATHS.get(observable_name)


def get_observable_from_state(
    cell_state: dict[str, Any], observable_name: str
) -> float:
    """Extract a single scalar observable value from v2ecoli cell state.

    Args:
        cell_state: The ``composite.state['agents'][agent_id]`` dict.
        observable_name: UQ dot-path name.

    Returns:
        Scalar value, or 0.0 if the path is not found.
    """
    path = OBSERVABLE_PATHS.get(observable_name)
    if path is None:
        return 0.0

    target = cell_state
    for key in path:
        if isinstance(target, dict):
            target = target.get(key)
        else:
            return 0.0

    if hasattr(target, "item"):
        return float(target.item())
    return float(target) if target is not None else 0.0


def get_array_observable_from_state(
    cell_state: dict[str, Any], preset_name: str
) -> np.ndarray | None:
    """Extract an array observable from v2ecoli cell state.

    Args:
        cell_state: The ``composite.state['agents'][agent_id]`` dict.
        preset_name: One of the keys in ``ARRAY_OBSERVABLE_PATHS``.

    Returns:
        1-D numpy array, or ``None`` if the path is not found
        or the target is not an array.
    """
    path = ARRAY_OBSERVABLE_PATHS.get(preset_name)
    if path is None:
        return None

    target = cell_state
    for key in path:
        if isinstance(target, dict):
            target = target.get(key)
        else:
            return None

    if target is None:
        return None

    arr = np.asarray(target, dtype=np.float64)
    return arr.ravel() if arr.ndim > 1 else arr


def is_array_preset(preset_name: str) -> bool:
    """Return ``True`` if *preset_name* is an array-type preset."""
    return preset_name in ARRAY_PRESETS


def is_scalar_preset(preset_name: str) -> bool:
    """Return ``True`` if *preset_name* is a scalar-type preset."""
    return preset_name in SCALAR_PRESETS


def observable_names_for_presets(presets: list[str]) -> list[str]:
    """Resolve a list of preset names to observable column names.

    Scalar presets (mass, higher_order) are resolved to their
    pre-defined column names.  Array presets (transcriptome, proteome,
    fluxome, exchange_fluxes) return a placeholder entry — the actual
    column names are generated at runtime after array extraction.

    Args:
        presets: List of preset names
            (e.g. ``["mass", "higher_order", "transcriptome"]``).

    Returns:
        Deduplicated list of observable names.  Array presets appear
        as a single sentinel entry ``"__array__:{name}__"`` so the
        caller knows to expand them.
    """
    seen: set[str] = set()
    names: list[str] = []
    for preset_name in presets:
        if is_array_preset(preset_name):
            key = f"__array__:{preset_name}__"
            if key not in seen:
                seen.add(key)
                names.append(key)
        else:
            for obs_name in PRESET_MAP.get(preset_name, []):
                if obs_name not in seen:
                    seen.add(obs_name)
                    names.append(obs_name)
    if not names:
        names = list(OBSERVABLE_PATHS.keys())
    return names
