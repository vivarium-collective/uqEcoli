"""build_core() — bigraph-schema core with UQ + v2ecoli types registered.

Provides a single ``build_core()`` entry point that registers all types
needed by the UQ pipeline composites, including ECOLI_TYPES from v2ecoli
when running with the v2ecoli backend.

Usage:
    from pbg_uqEcoli.core import build_core

    core = build_core()
    # ... build and run composites with UQ pipeline
"""

from __future__ import annotations

from typing import Any

from bigraph_schema import allocate_core


def build_core(with_v2ecoli: bool = False, with_viva_munk: bool = False) -> Any:
    """Allocate a bigraph-schema core with optional v2ecoli/viva_munk types.

    Args:
        with_v2ecoli: If True, register ECOLI_TYPES for v2ecoli composites.
        with_viva_munk: If True, register viva_munk colony types.

    Returns:
        Configured bigraph-schema Core.
    """
    core = allocate_core()

    if with_v2ecoli:
        from v2ecoli.types import ECOLI_TYPES

        core.register_types(ECOLI_TYPES)
        from v2ecoli.bridge import EcoliWCM

        core.register_link("EcoliWCM", EcoliWCM)

    if with_viva_munk:
        from viva_munk import core_import

        core = core_import()

    return core
