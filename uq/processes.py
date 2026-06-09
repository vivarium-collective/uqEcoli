"""
process-bigraph Steps for the RFC006 UQ pipeline.

Maps the two-stage ``uq.workflow`` API (sample → quantify) plus
visualization (dashboard) into composable process-bigraph Steps
that share state through typed stores.

Architecture:

    ┌─────────────┐     ┌─────────────────┐     ┌──────────────┐
    │ SetupInputs │──▶──│  RunSimulations │──▶──│  CollectCache │
    └─────────────┘     └─────────────────┘     └──────┬───────┘
                                                        │
                                                        ▼
                                                ┌──────────────┐
                                                │   Quantify   │
                                                └──────┬───────┘
                                                        │
                                                        ▼
                                                ┌──────────────┐
                                                │    Export     │
                                                └──────────────┘

Each Step fires once when its input dependencies are satisfied.
The full pipeline is composed via ``build_uq_composite()``.

All computation delegates to ``uq.workflow`` and ``uq.tui`` helpers —
these Steps are thin wiring, not reimplementations.
"""

from __future__ import annotations

import json as _json
import logging
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
from bigraph_schema import allocate_core
from process_bigraph import Composite, Step
from process_bigraph.emitter import RAMEmitter

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════
#  Step 1: SetupInputs — load simData, build parameter space, PCRV
# ═══════════════════════════════════════════════════════════════════


class SetupInputs(Step):
    """Load simData.cPickle and generate PCRV germ-space samples.

    UQPC Steps 1-2: construct the input PC object from parameter
    bounds (Legendre, order 1) and draw ``n_samples`` realizations
    via ``PCRV.sampleGerm()``.

    Writes ``X_train`` (physical space), ``germ_train`` (germ space),
    ``parameter_names``, and ``bounds`` into the shared state.
    """

    config_schema = {
        "params_file": {"_type": "string", "_default": ""},
    }

    def inputs(self):
        return {
            "sim_data_path": "string",
            "n_samples": "integer",
            "seed": "integer",
        }

    def outputs(self):
        return {
            "X_train": "string",  # JSON-serialized ndarray
            "germ_train": "string",  # JSON-serialized ndarray
            "parameter_names": "string",  # JSON list
            "bounds": "string",  # JSON-serialized ndarray
            "n_parameters": "integer",
        }

    def update(self, state: dict) -> dict:
        from libuq.pipeline.models import SimDataParameter
        from libuq.pipeline.param_loader import DEFAULT_SIM_DATA_PARAMETERS, ParameterDataset
        from uq.workflow import _setup_input_pc

        sim_data_path = str(Path(state["sim_data_path"]).resolve())
        n_samples = state.get("n_samples", 20)
        seed = state.get("seed", 42)

        ds = ParameterDataset(sim_data_path=sim_data_path)

        params_file = self.config.get("params_file", "")
        if params_file:
            raw = _json.loads(Path(params_file).read_text())
            parameters = [SimDataParameter.from_dict(p) for p in raw]
        else:
            parameters = None

        param_space = ds.to_parameter_space(parameters=parameters)
        bounds = np.array(param_space.parameter_bounds)

        input_pc, _, _ = _setup_input_pc(bounds)
        np.random.seed(seed)
        germ_train = input_pc.sampleGerm(n_samples)
        X_train = input_pc.evalPC(germ_train)

        logger.info(
            "SetupInputs: %d params, %d samples, X shape %s",
            param_space.n_parameters,
            n_samples,
            X_train.shape,
        )

        return {
            "X_train": _json.dumps(X_train.tolist()),
            "germ_train": _json.dumps(germ_train.tolist()),
            "parameter_names": _json.dumps(param_space.parameter_names),
            "bounds": _json.dumps(bounds.tolist()),
            "n_parameters": param_space.n_parameters,
        }


# ═══════════════════════════════════════════════════════════════════
#  Step 2: RunSimulations — execute vEcoli workflow.py subprocess
# ═══════════════════════════════════════════════════════════════════


class RunSimulations(Step):
    """Launch vEcoli simulations via workflow.py subprocess.

    UQPC Step 3: builds a workflow config from the PCRV samples,
    launches ``runscripts/workflow.py`` as a subprocess, and waits
    for completion.  The output directory (hive-partitioned Parquet)
    is written to shared state for the next step.
    """

    config_schema = {
        "max_duration": {"_type": "float", "_default": 10800.0},
        "params_file": {"_type": "string", "_default": ""},
    }

    def inputs(self):
        return {
            "sim_data_path": "string",
            "cache_dir": "string",
            "X_train": "string",  # JSON ndarray from SetupInputs
            "parameter_names": "string",  # JSON list from SetupInputs
            "bounds": "string",  # JSON ndarray from SetupInputs
            "n_samples": "integer",
            "generations": "integer",
            "n_init_sims": "integer",
        }

    def outputs(self):
        return {
            "history_base": "string",
            "return_code": "integer",
        }

    def update(self, state: dict) -> dict:
        from libuq.pipeline.models import SimDataParameter
        from libuq.pipeline.param_loader import ParameterDataset
        from uq.vecoli_config import (
            _build_config,
            _build_variants_from_samples,
            _get_vecoli_root,
        )

        sim_data_path = str(Path(state["sim_data_path"]).resolve())
        cache_path = Path(state["cache_dir"]).resolve()
        X_train = np.array(_json.loads(state["X_train"]))
        n_samples = state.get("n_samples", 20)
        generations = state.get("generations", 1)
        n_init_sims = state.get("n_init_sims", 1)
        max_duration = self.config.get("max_duration", 10800.0)

        # Rebuild param space for variant encoding
        ds = ParameterDataset(sim_data_path=sim_data_path)
        params_file = self.config.get("params_file", "")
        if params_file:
            raw = _json.loads(Path(params_file).read_text())
            parameters = [SimDataParameter.from_dict(p) for p in raw]
        else:
            parameters = None
        param_space = ds.to_parameter_space(parameters=parameters)

        # Build batch directory
        batch_dir = cache_path / "_batch"
        if batch_dir.exists():
            shutil.rmtree(batch_dir)
        batch_dir.mkdir(parents=True, exist_ok=True)
        output_dir = batch_dir / "output"
        output_dir.mkdir(exist_ok=True)
        experiment_id = "uqpc_batch"

        variants = _build_variants_from_samples(X_train, param_space._sim_data_parameters)
        config = _build_config(
            sim_data_path=sim_data_path,
            output_dir=str(output_dir),
            variants_section=variants,
            experiment_id=experiment_id,
            n_init_sims=n_init_sims,
            generations=generations,
            max_duration=max_duration,
        )
        config_path = batch_dir / "workflow_config.json"
        config_path.write_text(_json.dumps(config, indent=2))

        vecoli_root = _get_vecoli_root()
        nf_temp = Path(vecoli_root) / "nextflow_temp" / experiment_id
        if nf_temp.exists():
            shutil.rmtree(nf_temp)

        env = os.environ.copy()
        existing = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = vecoli_root + (os.pathsep + existing if existing else "")
        env["PYTHONUNBUFFERED"] = "1"

        workflow_script = os.path.join(vecoli_root, "runscripts", "workflow.py")
        cmd = [sys.executable, workflow_script, "--config", str(config_path)]

        logger.info("RunSimulations: launching workflow.py (%d samples)", n_samples)
        proc = subprocess.run(cmd, capture_output=True, text=True, cwd=vecoli_root, env=env)

        history_base = output_dir / experiment_id / "history"
        if not history_base.exists():
            candidates = list(output_dir.glob("*/history"))
            if candidates:
                history_base = candidates[0]

        logger.info("RunSimulations: exit code %d, history at %s", proc.returncode, history_base)

        return {
            "history_base": str(history_base),
            "return_code": proc.returncode,
        }


# ═══════════════════════════════════════════════════════════════════
#  Step 3: CollectCache — read Parquet, build PrecomputedCache
# ═══════════════════════════════════════════════════════════════════


class CollectCache(Step):
    """Collect hive-partitioned Parquet outputs into a PrecomputedCache.

    Reads the simulation outputs, aggregates per-variant timeseries,
    and writes (X, Y, timeseries, metadata) to the cache directory.
    """

    config_schema = {}

    def inputs(self):
        return {
            "history_base": "string",
            "cache_dir": "string",
            "X_train": "string",
            "germ_train": "string",
            "parameter_names": "string",
            "bounds": "string",
            "n_samples": "integer",
            "seed": "integer",
        }

    def outputs(self):
        return {
            "cache_ready": "boolean",
            "n_cached_samples": "integer",
        }

    def update(self, state: dict) -> dict:
        from libuq.sampling import PrecomputedCache
        from uq.vecoli_config import _collect_variant_timeseries

        history_base = Path(state["history_base"])
        cache_path = Path(state["cache_dir"]).resolve()
        X_train = np.array(_json.loads(state["X_train"]))
        germ_train = np.array(_json.loads(state["germ_train"]))
        parameter_names = _json.loads(state["parameter_names"])
        bounds = _json.loads(state["bounds"])
        n_samples = state.get("n_samples", 20)
        seed = state.get("seed", 42)

        obs = [
            "listeners__mass__dry_mass",
            "listeners__mass__cell_mass",
            "listeners__mass__volume",
            "listeners__mass__growth",
        ]

        Y_agg, Y_ts, Y_meta = _collect_variant_timeseries(history_base, n_samples, obs)

        cache_path.mkdir(parents=True, exist_ok=True)
        cache = PrecomputedCache(
            cache_dir=cache_path,
            X=X_train,
            Y=Y_agg,
            parameter_names=parameter_names,
            metadata={"bounds": bounds, "seed": seed, "observable_columns": obs},
            Y_timeseries=Y_ts,
            Y_timeseries_meta=Y_meta,
        )
        cache.save()
        np.save(cache_path / "germ_train.npy", germ_train)

        logger.info("CollectCache: cached %d samples to %s", Y_agg.shape[0], cache_path)

        return {
            "cache_ready": True,
            "n_cached_samples": int(Y_agg.shape[0]),
        }


# ═══════════════════════════════════════════════════════════════════
#  Step 4: Quantify — fit PCE surrogates + compute Sobol indices
# ═══════════════════════════════════════════════════════════════════


class Quantify(Step):
    """Fit PCE surrogates and compute Sobol indices (all 4 strategies).

    UQPC Steps 4-5: loads the PrecomputedCache and delegates to
    ``uq.workflow.quantify()`` which runs all four RFC006 strategies:
      1. Uniform (bulk)
      2. By generation
      3. By lineage seed
      4. Growth-stratified
    """

    config_schema = {
        "params_file": {"_type": "string", "_default": ""},
    }

    def inputs(self):
        return {
            "sim_data_path": "string",
            "cache_dir": "string",
            "export_path": "string",
            "cache_ready": "boolean",
            "polynomial_order": "integer",
            "n_bins": "integer",
            "regression": "string",
        }

    def outputs(self):
        return {
            "quantify_complete": "boolean",
            "results_json": "string",  # JSON-serialized summary
        }

    def update(self, state: dict) -> dict:
        from uq.workflow import quantify as wf_quantify

        cache_dir = state["cache_dir"]
        sim_data_path = state["sim_data_path"]
        export_path = state.get("export_path", "./uq_results")
        polynomial_order = state.get("polynomial_order", 2)
        n_bins = state.get("n_bins", 10)
        regression = state.get("regression", "lsq")

        logger.info(
            "Quantify: order=%d, bins=%d, regression=%s",
            polynomial_order,
            n_bins,
            regression,
        )

        result = wf_quantify(
            cache_dir=cache_dir,
            sim_data_path=sim_data_path,
            polynomial_order=polynomial_order,
            n_bins=n_bins,
            regression=regression,
            export_path=export_path,
        )

        # Read the exported JSON for state propagation
        results_json_path = Path(export_path) / "uq_results.json"
        if results_json_path.exists():
            results_json = results_json_path.read_text()
        else:
            results_json = "{}"

        logger.info("Quantify: complete, exported to %s", export_path)

        return {
            "quantify_complete": True,
            "results_json": results_json,
        }


# ═══════════════════════════════════════════════════════════════════
#  Step 5: Export — write artifacts + launch dashboard (optional)
# ═══════════════════════════════════════════════════════════════════


class Export(Step):
    """Final reporting step — artifacts are already written by Quantify.

    Reads the results JSON and produces a summary dict for the
    emitter / downstream consumers.  Optionally logs the report.
    """

    config_schema = {
        "log_report": {"_type": "boolean", "_default": True},
    }

    def inputs(self):
        return {
            "quantify_complete": "boolean",
            "results_json": "string",
            "export_path": "string",
        }

    def outputs(self):
        return {
            "pipeline_complete": "boolean",
            "summary": "string",  # JSON summary
        }

    def update(self, state: dict) -> dict:
        results = _json.loads(state.get("results_json", "{}"))
        export_path = state.get("export_path", "./uq_results")

        # Build compact summary
        params = list(results.get("parameters", {}).keys())
        s1 = results.get("phase1_population", {}).get("sobol_total_order", {})
        n_stages = results.get("phase2_growth_stratified", {}).get("n_stages", 0)
        s2_n = results.get("strategy2_by_generation", {}).get("n_generations", 0)
        s3_n = results.get("strategy3_by_seed", {}).get("n_seeds", 0)

        summary = {
            "parameters": params,
            "n_parameters": len(params),
            "strategy1_top": sorted(s1.items(), key=lambda kv: -kv[1])[:3] if s1 else [],
            "strategy2_n_generations": s2_n,
            "strategy3_n_seeds": s3_n,
            "strategy4_n_stages": n_stages,
            "export_path": export_path,
        }

        if self.config.get("log_report", True):
            logger.info("Export: pipeline complete — %d params, %d stages", len(params), n_stages)
            for name, val in summary.get("strategy1_top", []):
                logger.info("  S_Ti %s: %.1f%%", name, val * 100)

        return {
            "pipeline_complete": True,
            "summary": _json.dumps(summary),
        }


# ═══════════════════════════════════════════════════════════════════
#  Composite builder
# ═══════════════════════════════════════════════════════════════════


def get_core(**extra_types: type) -> Any:
    """Allocate a Core with the UQ Step types registered."""
    top = {
        "SetupInputs": SetupInputs,
        "RunSimulations": RunSimulations,
        "CollectCache": CollectCache,
        "Quantify": Quantify,
        "Export": Export,
        "RAMEmitter": RAMEmitter,
        **extra_types,
    }
    return allocate_core(top=top)


def build_state(
    sim_data_path: str,
    cache_dir: str = "./uq_cache",
    export_path: str = "./uq_results",
    n_samples: int = 20,
    seed: int = 42,
    generations: int = 1,
    n_init_sims: int = 1,
    polynomial_order: int = 2,
    n_bins: int = 10,
    regression: str = "lsq",
    max_duration: float = 10800.0,
    params_file: str = "",
    with_emitter: bool = True,
) -> dict[str, Any]:
    """Build the composite state document for the full UQ pipeline.

    The pipeline is a DAG of five Steps:

        SetupInputs → RunSimulations → CollectCache → Quantify → Export

    Each Step fires once when its input ports are satisfied.  Shared
    state stores (e.g. ``X_train``, ``cache_ready``) provide the
    data flow between steps.

    Args:
        sim_data_path: Path to simData.cPickle.
        cache_dir: Directory for the PrecomputedCache.
        export_path: Directory for final artifacts (uq_results.json etc.).
        n_samples: Number of PCRV samples.
        seed: Random seed.
        generations: Number of cell generations per simulation.
        n_init_sims: Number of initial seeds per variant.
        polynomial_order: PCE polynomial order.
        n_bins: Number of growth-progress bins (Strategy 4).
        regression: PyTUQ regression method ('lsq', 'bcs', 'anl').
        max_duration: vEcoli simulation wall-clock limit (seconds).
        params_file: Optional JSON file with SimDataParameter specs.
        with_emitter: Attach a RAMEmitter to observe final state.

    Returns:
        State dict consumable by ``Composite({"state": state})``.
    """
    process_config = {}
    if params_file:
        process_config["params_file"] = params_file

    sim_config = dict(process_config)
    sim_config["max_duration"] = max_duration

    state: dict[str, Any] = {
        # ── Pipeline parameters (shared stores) ──
        "sim_data_path": str(Path(sim_data_path).resolve()),
        "cache_dir": str(Path(cache_dir).resolve()),
        "export_path": str(Path(export_path).resolve()),
        "n_samples": n_samples,
        "seed": seed,
        "generations": generations,
        "n_init_sims": n_init_sims,
        "polynomial_order": polynomial_order,
        "n_bins": n_bins,
        "regression": regression,
        # ── Intermediate stores (written by Steps) ──
        "X_train": "",
        "germ_train": "",
        "parameter_names": "",
        "bounds": "",
        "n_parameters": 0,
        "history_base": "",
        "return_code": 0,
        "cache_ready": False,
        "n_cached_samples": 0,
        "quantify_complete": False,
        "results_json": "",
        "pipeline_complete": False,
        "summary": "",
        # ── Step 1: SetupInputs ──
        "setup_inputs": {
            "_type": "step",
            "address": "local:SetupInputs",
            "config": process_config,
            "inputs": {
                "sim_data_path": ["sim_data_path"],
                "n_samples": ["n_samples"],
                "seed": ["seed"],
            },
            "outputs": {
                "X_train": ["X_train"],
                "germ_train": ["germ_train"],
                "parameter_names": ["parameter_names"],
                "bounds": ["bounds"],
                "n_parameters": ["n_parameters"],
            },
        },
        # ── Step 2: RunSimulations ──
        "run_simulations": {
            "_type": "step",
            "address": "local:RunSimulations",
            "config": sim_config,
            "inputs": {
                "sim_data_path": ["sim_data_path"],
                "cache_dir": ["cache_dir"],
                "X_train": ["X_train"],
                "parameter_names": ["parameter_names"],
                "bounds": ["bounds"],
                "n_samples": ["n_samples"],
                "generations": ["generations"],
                "n_init_sims": ["n_init_sims"],
            },
            "outputs": {
                "history_base": ["history_base"],
                "return_code": ["return_code"],
            },
        },
        # ── Step 3: CollectCache ──
        "collect_cache": {
            "_type": "step",
            "address": "local:CollectCache",
            "config": {},
            "inputs": {
                "history_base": ["history_base"],
                "cache_dir": ["cache_dir"],
                "X_train": ["X_train"],
                "germ_train": ["germ_train"],
                "parameter_names": ["parameter_names"],
                "bounds": ["bounds"],
                "n_samples": ["n_samples"],
                "seed": ["seed"],
            },
            "outputs": {
                "cache_ready": ["cache_ready"],
                "n_cached_samples": ["n_cached_samples"],
            },
        },
        # ── Step 4: Quantify ──
        "quantify": {
            "_type": "step",
            "address": "local:Quantify",
            "config": process_config,
            "inputs": {
                "sim_data_path": ["sim_data_path"],
                "cache_dir": ["cache_dir"],
                "export_path": ["export_path"],
                "cache_ready": ["cache_ready"],
                "polynomial_order": ["polynomial_order"],
                "n_bins": ["n_bins"],
                "regression": ["regression"],
            },
            "outputs": {
                "quantify_complete": ["quantify_complete"],
                "results_json": ["results_json"],
            },
        },
        # ── Step 5: Export ──
        "export": {
            "_type": "step",
            "address": "local:Export",
            "config": {"log_report": True},
            "inputs": {
                "quantify_complete": ["quantify_complete"],
                "results_json": ["results_json"],
                "export_path": ["export_path"],
            },
            "outputs": {
                "pipeline_complete": ["pipeline_complete"],
                "summary": ["summary"],
            },
        },
    }

    if with_emitter:
        state["emitter"] = {
            "_type": "step",
            "address": "local:RAMEmitter",
            "config": {
                "emit": {
                    "pipeline_complete": "boolean",
                    "summary": "string",
                    "return_code": "integer",
                },
            },
            "inputs": {
                "pipeline_complete": ["pipeline_complete"],
                "summary": ["summary"],
                "return_code": ["return_code"],
            },
            "outputs": {},
        }

    return state


def build_composite(
    sim_data_path: str,
    cache_dir: str = "./uq_cache",
    export_path: str = "./uq_results",
    core: Any | None = None,
    **kwargs: Any,
) -> Composite:
    """Build and return a ready-to-run Composite for the full UQ pipeline.

    Usage::

        from uq.processes import build_composite

        composite = build_composite(
            sim_data_path="/path/to/simData.cPickle",
            n_samples=20,
            polynomial_order=2,
        )
        composite.run(1.0)

        print(composite.state["summary"])
        print(composite.state["pipeline_complete"])

    Args:
        sim_data_path: Path to simData.cPickle.
        cache_dir: Cache directory for samples.
        export_path: Export directory for artifacts.
        core: Optional pre-allocated Core.
        **kwargs: Forwarded to ``build_state()``.

    Returns:
        A ``Composite`` instance. Call ``.run(1.0)`` to execute.
    """
    if core is None:
        core = get_core()

    state = build_state(
        sim_data_path=sim_data_path,
        cache_dir=cache_dir,
        export_path=export_path,
        **kwargs,
    )

    return Composite({"state": state}, core=core)
