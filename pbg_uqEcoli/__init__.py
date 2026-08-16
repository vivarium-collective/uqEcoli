"""pbg_uqEcoli — process-bigraph integration layer for the UQ pipeline.

Wraps the full RFC006 UQ workflow as process-bigraph Processes and
Composites, enabling embedding inside larger bigraph simulations,
chaining with other processes, and visualization via the dashboard's
Composite Explorer.

Provides:
- ``UQPipeline`` — a Process that wraps the entire sample → quantify → report workflow
- ``pipeline_document`` — convenience generator for embedding in composites
"""

from __future__ import annotations

from typing import Any

from process_bigraph import Composite, Process

from pbg_uqEcoli.processes import UQPipeline


def pipeline_document(
    sim_data_path: str,
    cache_dir: str = "./uq_cache",
    export_path: str = "./uq_results",
    n_samples: int = 20,
    seed: int = 42,
    polynomial_order: int = 2,
    n_bins: int = 10,
    regression: str = "lsq",
    backend: str = "vecoli",
    **kwargs: Any,
) -> dict:
    """Build a process-bigraph document for the full UQ pipeline.

    The resulting document can be passed to ``Composite(doc, core=core)``.

    Args:
        sim_data_path: Path to simData.cPickle.
        cache_dir: Cache output directory.
        export_path: Export directory for artifacts.
        n_samples: Number of PCRV samples.
        seed: Random seed.
        polynomial_order: PCE polynomial order.
        n_bins: Growth-progress bins (Strategy 4).
        regression: Regression method ('lsq', 'bcs', 'anl').
        backend: Simulation backend ('vecoli' or 'v2ecoli').
        **kwargs: Additional config passed to UQPipeline.

    Returns:
        Process-bigraph document dict.
    """
    config = {
        "sim_data_path": str(Path(sim_data_path).resolve()),
        "cache_dir": str(Path(cache_dir).resolve()),
        "export_path": str(Path(export_path).resolve()),
        "n_samples": n_samples,
        "seed": seed,
        "polynomial_order": polynomial_order,
        "n_bins": n_bins,
        "regression": regression,
        "backend": backend,
        **kwargs,
    }

    from pathlib import Path

    return {
        "state": {
            "pipeline": {
                "_type": "process",
                "address": "local:UQPipeline",
                "config": config,
                "inputs": {},
                "outputs": {
                    "pipeline_complete": ["pipeline_complete"],
                    "results_path": ["results_path"],
                    "report_path": ["report_path"],
                },
            },
            "pipeline_complete": False,
            "results_path": "",
            "report_path": "",
        }
    }


__all__ = ["UQPipeline", "pipeline_document"]
