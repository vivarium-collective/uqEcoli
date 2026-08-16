"""
Atomic process-bigraph Process/Step implementations for the UQ pipeline.

Three categories of improvement over the procedural code in ``uq.cli``:

1. **Subprocess management** — ``RunSimulations`` wraps vEcoli's
   ``workflow.py`` as a Process (not Step) so the composite can poll
   progress at each interval tick.  The ``n_completed`` output port
   drives progress bars in any UI without manual ``os.read()`` loops.

2. **PCE evaluation** — ``PCEEvaluate`` is a stateless Step that
   loads surrogate artifacts and evaluates the PCE at arbitrary
   parameter values.  All three dashboards (tkinter, marimo, TUI)
   can wire to the same Step instead of duplicating
   ``legendre_eval()``.

3. **Per-strategy fitting** — ``StrategyFit`` runs ``run_uqpc()``
   for a single strategy/group.  The composite can launch multiple
   StrategyFit Steps in parallel (one per generation, seed, or
   growth stage) via the bigraph runtime.
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
from process_bigraph import Process, Step

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════
#  SetupInputs — load simData, build parameter space, PCRV samples
# ═══════════════════════════════════════════════════════════════════


class SetupInputs(Step):
    """UQPC Steps 1-2: build parameter space + PCRV germ samples.

    Loads ``simData.cPickle``, constructs the input PC (Legendre,
    order 1), and draws ``n_samples`` via ``PCRV.sampleGerm()``.
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
            "X_train": "string",
            "germ_train": "string",
            "parameter_names": "string",
            "bounds": "string",
            "n_parameters": "integer",
        }

    def update(self, state: dict) -> dict:
        from libuq.pipeline.models import SimDataParameter
        from libuq.pipeline.param_loader import ParameterDataset
        from uq.workflow import _setup_input_pc

        sim_data_path = str(Path(state["sim_data_path"]).resolve())
        n_samples = state.get("n_samples", 20)
        seed = state.get("seed", 42)

        ds = ParameterDataset(sim_data_path=sim_data_path)
        params_file = self.config.get("params_file", "")
        parameters = None
        if params_file:
            raw = _json.loads(Path(params_file).read_text())
            parameters = [SimDataParameter.from_dict(p) for p in raw]

        param_space = ds.to_parameter_space(parameters=parameters)
        bounds = np.array(param_space.parameter_bounds)

        input_pc, _, _ = _setup_input_pc(bounds)
        np.random.seed(seed)
        germ_train = input_pc.sampleGerm(n_samples)
        X_train = input_pc.evalPC(germ_train)

        logger.info("SetupInputs: %d params, %d samples", param_space.n_parameters, n_samples)

        return {
            "X_train": _json.dumps(X_train.tolist()),
            "germ_train": _json.dumps(germ_train.tolist()),
            "parameter_names": _json.dumps(param_space.parameter_names),
            "bounds": _json.dumps(bounds.tolist()),
            "n_parameters": param_space.n_parameters,
        }


# ═══════════════════════════════════════════════════════════════════
#  RunSimulations — vEcoli subprocess with real-time progress
# ═══════════════════════════════════════════════════════════════════
#
#  WHY a Process instead of a Step:
#
#  ``uq.cli.sample()`` uses a Popen + polling loop to stream
#  Nextflow stdout and count completed variant directories every
#  0.5 s.  This maps naturally to a time-driven Process whose
#  ``update()`` is called at each composite interval tick:
#
#    - First tick:  launch subprocess
#    - Subsequent ticks:  poll stdout, count variants, emit progress
#    - Final tick (proc finished):  emit return_code
#
#  The composite's interval replaces ``_time.sleep(0.5)`` and the
#  ``n_completed`` output port replaces manual Rich progress updates.
# ═══════════════════════════════════════════════════════════════════


class RunSimulations(Process):
    """Launch vEcoli workflow.py and emit real-time progress.

    Unlike the Step version in ``uq.processes`` which blocks until
    completion, this Process launches the subprocess on the first
    tick and emits ``n_completed`` / ``total_sims`` on each
    subsequent tick.  UIs wire to these ports for live progress.

    The subprocess is launched exactly once (guarded by an internal
    flag).  Subsequent ``update()`` calls poll stdout and count
    completed variant directories.
    """

    config_schema = {
        "max_duration": {"_type": "float", "_default": 10800.0},
        "params_file": {"_type": "string", "_default": ""},
    }

    def __init__(self, config=None, core=None, **kwargs):
        super().__init__(config or {}, core=core, **kwargs)
        self._proc = None
        self._launched = False
        self._history_base = None
        self._total_sims = 0

    def inputs(self):
        return {
            "sim_data_path": "string",
            "cache_dir": "string",
            "X_train": "string",
            "parameter_names": "string",
            "bounds": "string",
            "n_samples": "integer",
            "generations": "integer",
            "n_init_sims": "integer",
        }

    def outputs(self):
        return {
            "history_base": "string",
            "return_code": "integer",
            "n_completed": "integer",
            "total_sims": "integer",
            "sim_stdout": "string",
        }

    def update(self, state: dict, interval: float) -> dict:
        import re

        from libuq.pipeline.models import SimDataParameter
        from libuq.pipeline.param_loader import ParameterDataset
        from uq.tui import (
            _build_config,
            _build_variants_from_samples,
            _count_completed_variants,
            _get_vecoli_root,
        )

        # ── First tick: launch subprocess ──
        if not self._launched:
            self._launched = True

            sim_data_path = str(Path(state["sim_data_path"]).resolve())
            cache_path = Path(state["cache_dir"]).resolve()
            X_train = np.array(_json.loads(state["X_train"]))
            n_samples = state.get("n_samples", 20)
            generations = state.get("generations", 1)
            n_init_sims = state.get("n_init_sims", 1)
            max_duration = self.config.get("max_duration", 10800.0)

            ds = ParameterDataset(sim_data_path=sim_data_path)
            params_file = self.config.get("params_file", "")
            parameters = None
            if params_file:
                raw = _json.loads(Path(params_file).read_text())
                parameters = [SimDataParameter.from_dict(p) for p in raw]
            param_space = ds.to_parameter_space(parameters=parameters)

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

            self._proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                cwd=vecoli_root,
                env=env,
            )
            self._total_sims = (n_samples + 1) * n_init_sims * generations
            self._history_base = output_dir / experiment_id / "history"

            logger.info("RunSimulations: launched (%d total sims)", self._total_sims)
            return {
                "history_base": str(self._history_base),
                "return_code": -1,
                "n_completed": 0,
                "total_sims": self._total_sims,
                "sim_stdout": "",
            }

        # ── Subsequent ticks: poll progress ──
        stdout_chunk = ""
        if self._proc and self._proc.stdout is not None and self._proc.poll() is None:
            fd = self._proc.stdout.fileno()
            try:
                chunk = os.read(fd, 8192)
                if chunk:
                    ansi_re = re.compile(r"\x1b\[[\d;]*[A-Za-z]|\x1b\[\d*[A-GJK]|\x07")
                    stdout_chunk = ansi_re.sub("", chunk.decode("utf-8", errors="replace"))
            except OSError:
                pass

        n_done = _count_completed_variants(self._history_base) if self._history_base else 0

        # ── Check if done ──
        return_code = -1
        if self._proc and self._proc.poll() is not None:
            return_code = self._proc.returncode
            # Drain remaining
            if self._proc.stdout:
                remaining = self._proc.stdout.read()
                if remaining:
                    stdout_chunk += remaining.decode("utf-8", errors="replace")
            # Resolve history base
            if self._history_base and not self._history_base.exists():
                cache_path = Path(state["cache_dir"]).resolve()
                output_dir = cache_path / "_batch" / "output"
                candidates = list(output_dir.glob("*/history"))
                if candidates:
                    self._history_base = candidates[0]
            logger.info("RunSimulations: finished (exit %d, %d/%d done)", return_code, n_done, self._total_sims)

        return {
            "history_base": str(self._history_base or ""),
            "return_code": return_code,
            "n_completed": n_done,
            "total_sims": self._total_sims,
            "sim_stdout": stdout_chunk,
        }


# ═══════════════════════════════════════════════════════════════════
#  CollectCache — Parquet → PrecomputedCache
# ═══════════════════════════════════════════════════════════════════


class CollectCache(Step):
    """Read hive-partitioned Parquet, build PrecomputedCache."""

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

        logger.info("CollectCache: %d samples → %s", Y_agg.shape[0], cache_path)
        return {"cache_ready": True, "n_cached_samples": int(Y_agg.shape[0])}


# ═══════════════════════════════════════════════════════════════════
#  StrategyFit — single-strategy UQPC fit (parallelizable)
# ═══════════════════════════════════════════════════════════════════
#
#  WHY a separate Step:
#
#  ``uq.workflow.quantify()`` runs strategies 1-4 sequentially.
#  But strategies 1, 2, 3 are independent — each calls run_uqpc()
#  on different Y matrices.  Strategy 4 is also independent (uses
#  Y_timeseries with growth binning).
#
#  By decomposing into one StrategyFit Step per strategy, the
#  composite runtime can execute them in parallel.  For strategy 2
#  (by-generation) and strategy 3 (by-seed), multiple StrategyFit
#  instances run — one per generation/seed.
# ═══════════════════════════════════════════════════════════════════


class StrategyFit(Step):
    """Run ``run_uqpc()`` for a single strategy or group.

    Config ``strategy`` selects the mode:
      - ``"uniform"``  — Strategy 1 (bulk Y)
      - ``"generation"`` — Strategy 2 (filter by generation)
      - ``"seed"`` — Strategy 3 (filter by lineage seed)
      - ``"growth_stage"`` — Strategy 4 (bin by θ)

    Config ``group_value`` selects the specific group for strategies
    2-3 (e.g. generation=1, seed=0).  For strategy 4, ``group_value``
    is the stage index.

    The Step loads the cache, extracts the appropriate Y slice, and
    fits a PCE + computes Sobol.
    """

    config_schema = {
        "strategy": {"_type": "string", "_default": "uniform"},
        "group_value": {"_type": "integer", "_default": -1},
        "params_file": {"_type": "string", "_default": ""},
    }

    def inputs(self):
        return {
            "sim_data_path": "string",
            "cache_dir": "string",
            "cache_ready": "boolean",
            "polynomial_order": "integer",
            "n_bins": "integer",
            "regression": "string",
        }

    def outputs(self):
        return {
            "sobol_json": "string",
            "fit_complete": "boolean",
        }

    def update(self, state: dict) -> dict:
        from libuq.pipeline.param_loader import DEFAULT_SIM_DATA_PARAMETERS, ParameterDataset
        from libuq.sampling import PrecomputedCache
        from uq.workflow import (
            _aggregate_by_group,
            _bin_by_growth_stage,
            _compute_growth_fraction,
            run_uqpc,
        )

        cache_dir = Path(state["cache_dir"])
        cache = PrecomputedCache.load(cache_dir)
        X, Y = cache.X, cache.Y

        ds = ParameterDataset(sim_data_path=state["sim_data_path"])
        params_file = self.config.get("params_file", "")
        parameters = None
        if params_file:
            from libuq.pipeline.models import SimDataParameter

            raw = _json.loads(Path(params_file).read_text())
            parameters = [SimDataParameter.from_dict(p) for p in raw]
        param_space = ds.to_parameter_space(parameters=parameters)

        poly_order = state.get("polynomial_order", 2)
        regression = state.get("regression", "lsq")
        strategy = self.config.get("strategy", "uniform")
        group_val = self.config.get("group_value", -1)

        if strategy == "uniform":
            result = run_uqpc(
                param_space=param_space, Y_train=Y, X_train=X, polynomial_order=poly_order, regression=regression
            )
            label = "strategy1_uniform"

        elif strategy == "generation":
            grouped = _aggregate_by_group(cache.Y_timeseries, cache.Y_timeseries_meta, "generation")
            Y_g = grouped.get(group_val, Y)
            result = run_uqpc(
                param_space=param_space, Y_train=Y_g, X_train=X, polynomial_order=poly_order, regression=regression
            )
            label = f"strategy2_gen{group_val}"

        elif strategy == "seed":
            grouped = _aggregate_by_group(cache.Y_timeseries, cache.Y_timeseries_meta, "lineage_seed")
            Y_s = grouped.get(group_val, Y)
            result = run_uqpc(
                param_space=param_space, Y_train=Y_s, X_train=X, polynomial_order=poly_order, regression=regression
            )
            label = f"strategy3_seed{group_val}"

        elif strategy == "growth_stage":
            n_bins = state.get("n_bins", 10)
            n_obs = cache.Y_timeseries[0].shape[1]
            Y_stage_list = []
            for ts in cache.Y_timeseries:
                theta = _compute_growth_fraction(ts, 0)
                bins = _bin_by_growth_stage(theta, n_bins)
                stage_means = np.zeros(n_obs)
                mask = bins == group_val
                if np.any(mask):
                    stage_means = ts[mask].mean(axis=0)
                Y_stage_list.append(stage_means)
            Y_stage = np.vstack(Y_stage_list)
            result = run_uqpc(
                param_space=param_space, Y_train=Y_stage, X_train=X, polynomial_order=poly_order, regression=regression
            )
            label = f"strategy4_stage{group_val}"
        else:
            raise ValueError(f"Unknown strategy: {strategy!r}")

        names = param_space.parameter_names
        sobol_dict = {
            "label": label,
            "strategy": strategy,
            "group_value": group_val,
            "sobol_total_order": {n: round(float(v), 6) for n, v in zip(names, result.sobol.total_order)},
            "sobol_first_order": {n: round(float(v), 6) for n, v in zip(names, result.sobol.first_order)},
            "relerr_train": [round(float(e), 6) for e in result.relerr_train],
        }

        logger.info(
            "StrategyFit[%s]: done, top S_Ti=%s",
            label,
            sorted(sobol_dict["sobol_total_order"].items(), key=lambda kv: -kv[1])[:2],
        )

        return {
            "sobol_json": _json.dumps(sobol_dict),
            "fit_complete": True,
        }


# ═══════════════════════════════════════════════════════════════════
#  Quantify — all 4 strategies (delegates to uq.workflow.quantify)
# ═══════════════════════════════════════════════════════════════════


class Quantify(Step):
    """Fit PCE + Sobol for all 4 RFC006 strategies.

    Delegates to ``uq.workflow.quantify()`` which handles the full
    pipeline.  Use ``StrategyFit`` instead if you need per-strategy
    parallelism.
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
            "results_json": "string",
        }

    def update(self, state: dict) -> dict:
        from uq.workflow import quantify as wf_quantify

        result = wf_quantify(
            cache_dir=state["cache_dir"],
            sim_data_path=state["sim_data_path"],
            polynomial_order=state.get("polynomial_order", 2),
            n_bins=state.get("n_bins", 10),
            regression=state.get("regression", "lsq"),
            export_path=state.get("export_path", "./uq_results"),
        )

        results_path = Path(state.get("export_path", "./uq_results")) / "uq_results.json"
        results_json = results_path.read_text() if results_path.exists() else "{}"

        logger.info("Quantify: complete")
        return {"quantify_complete": True, "results_json": results_json}


# ═══════════════════════════════════════════════════════════════════
#  PCEEvaluate — shared surrogate evaluation for all UIs
# ═══════════════════════════════════════════════════════════════════
#
#  WHY this belongs in process-bigraph:
#
#  Three separate files contain nearly identical Legendre evaluation
#  code:
#    - app/dashboard_simple.py:113-130  (marimo cell)
#    - app/uq_daw_simple.py:47-63      (standalone function)
#    - uq/tui.py                        (implicit via workflow)
#
#  By wrapping PCE evaluation as a Step, all UIs wire to the same
#  implementation.  The Step loads surrogate artifacts once (from
#  the export directory) and evaluates at parameter values provided
#  via input ports.
# ═══════════════════════════════════════════════════════════════════


class PCEEvaluate(Step):
    """Evaluate a PCE surrogate at given parameter values.

    Loads ``population_surrogate/`` artifacts from the export
    directory and evaluates the Legendre PCE at the parameter
    vector provided in the ``x_physical`` input port.

    Also computes per-parameter sweep curves and local sensitivity
    (numerical gradient) for dashboard visualization.

    This replaces the duplicated ``legendre_eval()`` functions in
    ``app/dashboard_simple.py`` and ``app/uq_daw_simple.py``.
    """

    config_schema = {
        "n_sweep": {"_type": "integer", "_default": 80},
    }

    def inputs(self):
        return {
            "export_path": "string",
            "x_physical": "string",  # JSON array of param values
        }

    def outputs(self):
        return {
            "y_hat": "float",
            "sweep_curves": "string",  # JSON: {param: {x: [...], y: [...]}}
            "local_sensitivity": "string",  # JSON: {param: float}
        }

    def update(self, state: dict) -> dict:
        export_dir = Path(state["export_path"])
        pop_dir = export_dir / "population_surrogate"

        coeffs = np.load(pop_dir / "coefficients.npy")
        mi = np.load(pop_dir / "multi_indices.npy")
        bounds = np.load(pop_dir / "input_bounds.npy")

        x = np.array(_json.loads(state["x_physical"]))
        x_norm = 2.0 * (x - bounds[:, 0]) / (bounds[:, 1] - bounds[:, 0] + 1e-12) - 1.0
        y_hat = float(_legendre_eval(x_norm, coeffs, mi))

        # Sweep curves
        n_sweep = self.config.get("n_sweep", 80)
        results_path = export_dir / "uq_results.json"
        param_names = (
            list(_json.loads(results_path.read_text())["parameters"].keys())
            if results_path.exists()
            else [f"x{i}" for i in range(len(x))]
        )

        sweep_curves = {}
        local_sens = {}
        for pi, pname in enumerate(param_names):
            lo, hi = float(bounds[pi, 0]), float(bounds[pi, 1])
            sv = np.linspace(lo, hi, n_sweep)
            sy = []
            for v in sv:
                x_sw = x.copy()
                x_sw[pi] = v
                x_sw_n = 2.0 * (x_sw - bounds[:, 0]) / (bounds[:, 1] - bounds[:, 0] + 1e-12) - 1.0
                sy.append(float(_legendre_eval(x_sw_n, coeffs, mi)))
            sweep_curves[pname] = {"x": sv.tolist(), "y": sy}

            delta = (hi - lo) * 0.005
            x_plus, x_minus = x.copy(), x.copy()
            x_plus[pi] = min(x[pi] + delta, hi)
            x_minus[pi] = max(x[pi] - delta, lo)
            y_plus = _legendre_eval(
                2.0 * (x_plus - bounds[:, 0]) / (bounds[:, 1] - bounds[:, 0] + 1e-12) - 1.0, coeffs, mi
            )
            y_minus = _legendre_eval(
                2.0 * (x_minus - bounds[:, 0]) / (bounds[:, 1] - bounds[:, 0] + 1e-12) - 1.0, coeffs, mi
            )
            local_sens[pname] = float(abs(y_plus - y_minus) / (2 * delta + 1e-12))

        return {
            "y_hat": y_hat,
            "sweep_curves": _json.dumps(sweep_curves),
            "local_sensitivity": _json.dumps(local_sens),
        }


# ═══════════════════════════════════════════════════════════════════
#  Export — final reporting
# ═══════════════════════════════════════════════════════════════════


class Export(Step):
    """Read results JSON and produce a compact summary."""

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
            "summary": "string",
        }

    def update(self, state: dict) -> dict:
        results = _json.loads(state.get("results_json", "{}"))
        export_path = state.get("export_path", "./uq_results")

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
            logger.info("Export: %d params, %d stages", len(params), n_stages)

        return {"pipeline_complete": True, "summary": _json.dumps(summary)}


# ═══════════════════════════════════════════════════════════════════
#  Shared Legendre evaluation (canonical implementation)
# ═══════════════════════════════════════════════════════════════════


def _legendre_eval(x_norm: np.ndarray, coeffs: np.ndarray, mi: np.ndarray) -> float:
    """Ŷ = Σ c_α · Π P_{α_i}(x_i) via stable Legendre recurrence.

    This is the single canonical implementation.  All dashboards
    should use ``PCEEvaluate`` (which calls this) instead of their
    own copies.
    """
    max_ord = int(mi.max()) if mi.size > 0 else 0
    n_p = mi.shape[1]
    P = np.zeros((max_ord + 1, n_p))
    P[0, :] = 1.0
    if max_ord >= 1:
        P[1, :] = x_norm
    for n in range(2, max_ord + 1):
        P[n, :] = ((2 * n - 1) * x_norm * P[n - 1, :] - (n - 1) * P[n - 2, :]) / n
    result = 0.0
    for t in range(len(coeffs)):
        term = float(coeffs[t])
        for p in range(n_p):
            term *= P[mi[t, p], p]
        result += term
    return result
