"""Process-bigraph Process implementations for the UQ pipeline."""

from __future__ import annotations

import json as _json
import logging
from pathlib import Path
from typing import Any

from process_bigraph import Process

logger = logging.getLogger(__name__)


class UQPipeline(Process):
    """Full RFC006 UQ workflow as a single Process node.

    Wraps the entire sample → quantify → report pipeline as a
    time-driven Process. On each tick, advances the pipeline through
    its stages (setup inputs → sample → quantify → report).

    Config:
        sim_data_path: Path to simData.cPickle.
        cache_dir: Cache output directory.
        export_path: Export directory for artifacts.
        n_samples: Number of PCRV samples.
        seed: Random seed.
        generations: Cell generations per simulation.
        n_init_sims: Initial seeds per variant.
        polynomial_order: PCE polynomial order.
        n_bins: Growth-progress bins.
        regression: Regression method ('lsq', 'bcs', 'anl').
        max_duration: vEcoli wall-clock limit (seconds).
        backend: Simulation backend ('vecoli' or 'v2ecoli').
    """

    config_schema = {
        "sim_data_path": {"_type": "string"},
        "cache_dir": {"_type": "string", "_default": "./uq_cache"},
        "export_path": {"_type": "string", "_default": "./uq_results"},
        "n_samples": {"_type": "integer", "_default": 20},
        "seed": {"_type": "integer", "_default": 42},
        "generations": {"_type": "integer", "_default": 1},
        "n_init_sims": {"_type": "integer", "_default": 1},
        "polynomial_order": {"_type": "integer", "_default": 2},
        "n_bins": {"_type": "integer", "_default": 10},
        "regression": {"_type": "string", "_default": "lsq"},
        "max_duration": {"_type": "float", "_default": 10800.0},
        "backend": {"_type": "string", "_default": "vecoli"},
    }

    def __init__(self, config=None, core=None, **kwargs):
        super().__init__(config or {}, core=core, **kwargs)
        self._stage = "idle"
        self._pipeline_complete = False
        self._results_path = ""
        self._report_path = ""

    def inputs(self):
        return {}

    def outputs(self):
        return {
            "pipeline_complete": "boolean",
            "results_path": "string",
            "report_path": "string",
        }

    def update(self, state: dict, interval: float) -> dict:
        """Advance the pipeline by one stage on each tick."""
        cfg = self.config

        if self._stage == "idle":
            self._stage = "setup"
            logger.info("UQPipeline: starting (backend=%s)", cfg.get("backend"))
            return self._output()

        if self._stage == "setup":
            self._stage = "sample"
            logger.info("UQPipeline: setup complete, starting sampling")
            return self._output()

        if self._stage == "sample":
            self._run_sample(cfg)
            self._stage = "quantify"
            return self._output()

        if self._stage == "quantify":
            self._run_quantify(cfg)
            self._stage = "report"

            # Generate HTML report
            from uq.report import generate_html_report

            self._report_path = str(generate_html_report(cfg["export_path"]))
            self._pipeline_complete = True
            logger.info("UQPipeline: complete — report at %s", self._report_path)

    def _output(self) -> dict:
        return {
            "pipeline_complete": self._pipeline_complete,
            "results_path": self._results_path,
            "report_path": self._report_path,
        }

    def _run_sample(self, cfg: dict) -> None:
        """Run the sampling stage."""
        import numpy as np
        from libuq.pipeline.models import SimDataParameter
        from libuq.pipeline.param_loader import ParameterDataset
        from uq.workflow import _setup_input_pc

        sim_data_path = str(Path(cfg["sim_data_path"]).resolve())
        cache_path = Path(cfg["cache_dir"]).resolve()
        n_samples = cfg.get("n_samples", 20)
        seed = cfg.get("seed", 42)
        generations = cfg.get("generations", 1)
        n_init_sims = cfg.get("n_init_sims", 1)
        max_duration = cfg.get("max_duration", 10800.0)
        backend = cfg.get("backend", "vecoli")

        ds = ParameterDataset(sim_data_path=sim_data_path)
        param_space = ds.to_parameter_space(parameters=None)
        bounds = np.array(param_space.parameter_bounds)

        input_pc, _, _ = _setup_input_pc(bounds)
        np.random.seed(seed)
        germ_train = input_pc.sampleGerm(n_samples)
        X_train = input_pc.evalPC(germ_train)

        if backend == "v2ecoli":
            from uq.generators.v2ecoli import V2ecoliGenerator, generate_cache_bundle

            v2e_cache = str(cache_path / "_v2ecoli_cache")
            generate_cache_bundle(sim_data_path, v2e_cache, seed=seed)

            gen = V2ecoliGenerator(
                cache_dir=v2e_cache,
                param_names=param_space.parameter_names,
                seed=seed,
                n_generations=generations,
                max_time=max_duration,
            )
            Y_agg, Y_ts, Y_meta = gen._run_batch(X_train)
        else:
            from libuq.generators.vecoli import TimeseriesGeneratorVecoli

            gen = TimeseriesGeneratorVecoli(
                baseline_sim_data=ds._sim_data,
                param_space=param_space,
                generations=generations,
                n_init_sims=n_init_sims,
                max_duration=max_duration,
            )
            Y_agg, Y_ts, Y_meta = gen._run_batch(X_train)

        from libuq.sampling import PrecomputedCache

        cache_path.mkdir(parents=True, exist_ok=True)
        cache = PrecomputedCache(
            cache_dir=cache_path,
            X=X_train,
            Y=Y_agg,
            parameter_names=param_space.parameter_names,
            metadata={
                "bounds": bounds.tolist(),
                "seed": seed,
                "observable_columns": [],
            },
            Y_timeseries=Y_ts,
            Y_timeseries_meta=Y_meta,
        )
        cache.save()
        np.save(cache_path / "germ_train.npy", germ_train)
        logger.info("UQPipeline: cached %d samples", Y_agg.shape[0])

    def _run_quantify(self, cfg: dict) -> None:
        """Run the quantification stage."""
        from uq.workflow import quantify as wf_quantify

        wf_quantify(
            cache_dir=cfg["cache_dir"],
            sim_data_path=cfg["sim_data_path"],
            polynomial_order=cfg.get("polynomial_order", 2),
            n_bins=cfg.get("n_bins", 10),
            regression=cfg.get("regression", "lsq"),
            export_path=cfg["export_path"],
        )
        self._results_path = str(Path(cfg["export_path"]) / "uq_results.json")
        logger.info("UQPipeline: quantify complete — %s", self._results_path)
