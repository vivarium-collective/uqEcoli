"""
vEcoli simulation function for UQ pipeline: f(x) -> y.

This module provides a parameter-agnostic simulation wrapper that:
1. Accepts x as a numpy array positionally aligned with XSpace parameters
2. Converts LHS samples into a vEcoli workflow config with variants
3. Runs simulations via ``runscripts/workflow.py`` (Nextflow orchestration)
4. Collects Parquet outputs and returns timeseries arrays

**IMPORTANT:** All simulation execution goes through vEcoli's built-in
``runscripts/workflow.py --config <path>``.  We NEVER instantiate
``EcoliSim`` in-process or hand-roll simulation logic.  The variant
system (``ecoli.variants.*``) handles all sim_data mutations.

For generic sim_data parameters (arbitrary dot-path attributes), we use
the ``ecoli.variants.sim_data_setattr`` variant function with ``op: "zip"``
to encode N LHS samples as N variant parameter dicts.
"""

import abc
import copy
import importlib
import json as _json
import logging
import os
import pickle
import shutil
import subprocess
import sys
import tempfile
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
from reconstruction.ecoli.simulation_data import SimulationDataEcoli

from libuq.common import BaseClass

logger = logging.getLogger(__name__)

# -- Helpers ----------------------------------------------------------------


def _get_nested_dict(d: dict, keys: list[str]) -> Any:
    """Traverse a nested dict by a list of keys."""
    for k in keys:
        d = d[k]
    return d


def _fix_new_gene_rel_adj(
    sim_data: SimulationDataEcoli,
    params: dict[str, Any],
) -> dict[str, Any]:
    """Broadcast rel_adj lists to match the actual new gene counts in sim_data."""
    if "rel_adj" not in params:
        return params

    from ecoli.variants.new_gene_internal_shift_variable_strength import (
        get_new_gene_ids_and_indices,
    )

    _, rna_indices, _, monomer_indices = get_new_gene_ids_and_indices(sim_data)
    n_rna = len(rna_indices)
    n_mono = len(monomer_indices)

    rel_adj = params["rel_adj"]
    exp_list = rel_adj.get("rel_exp_adj_list", [1.0])
    trl_list = rel_adj.get("rel_trl_eff_adj_list", [1.0])

    if len(exp_list) == 1 and n_rna > 1:
        exp_list = exp_list * n_rna
    if len(trl_list) == 1 and n_mono > 1:
        trl_list = trl_list * n_mono

    params = {
        **params,
        "rel_adj": {
            "rel_exp_adj_list": exp_list,
            "rel_trl_eff_adj_list": trl_list,
        },
    }
    return params


def _apply_variants(
    sim_data: SimulationDataEcoli,
    variants_config: dict[str, list[dict[str, Any]]],
) -> SimulationDataEcoli:
    """Apply vEcoli variant functions to sim_data."""
    for variant_name, param_dicts in variants_config.items():
        variant_mod = importlib.import_module(f"ecoli.variants.{variant_name}")
        for params in param_dicts:
            if variant_name == "new_gene_internal_shift_variable_strength":
                params = _fix_new_gene_rel_adj(sim_data, params)
            sim_data = variant_mod.apply_variant(sim_data, params)
    return sim_data


def _apply_sim_data_mutations(
    sim_data: SimulationDataEcoli,
    mutations: dict[str, Any],
) -> None:
    """Apply direct attribute mutations to sim_data via dot-path traversal."""
    for attr_path, value in mutations.items():
        parts = attr_path.split(".")
        obj = sim_data
        for part in parts[:-1]:
            obj = getattr(obj, part)
        if isinstance(value, dict) and "__index__" in value:
            arr = getattr(obj, parts[-1])
            arr[value["__index__"]] = value["__value__"]
        else:
            setattr(obj, parts[-1], value)


# -- Parquet column mapping --------------------------------------------------

_SHORT_TO_PARQUET: dict[str, str] = {
    "cell_mass": "listeners__mass__cell_mass",
    "dry_mass": "listeners__mass__dry_mass",
    "volume": "listeners__mass__volume",
    "growth": "listeners__mass__growth",
    "growth_rate": "listeners__mass__growth",
}

_DEFAULT_PARQUET_COLUMNS = [
    "listeners__mass__dry_mass",
    "listeners__mass__cell_mass",
    "listeners__mass__volume",
    "listeners__mass__growth",
]


def _resolve_parquet_columns(output_keys: list[str] | None) -> list[str]:
    """Convert short output key names to full Parquet column names."""
    if output_keys is None:
        return list(_DEFAULT_PARQUET_COLUMNS)
    resolved = []
    for key in output_keys:
        resolved.append(_SHORT_TO_PARQUET.get(key, key))
    return resolved


def _read_parquet_timeseries(
    output_dir: Path,
    experiment_id: str,
    observable_columns: list[str],
) -> tuple[np.ndarray, list[str]]:
    """Read hive-partitioned Parquet output and extract timeseries."""
    history_base = output_dir / experiment_id / "history"
    if not history_base.exists():
        # Try with timestamp suffix — workflow.py appends timestamps
        candidates = list(output_dir.glob(f"{experiment_id}*/history"))
        if candidates:
            history_base = candidates[0]

    pq_files = list(history_base.rglob("*.pq"))
    if not pq_files:
        pq_files = list(history_base.rglob("*.parquet"))
    if not pq_files:
        raise FileNotFoundError(
            f"No Parquet files found under {history_base}. Check that the simulation completed successfully."
        )

    df = pl.read_parquet(
        [str(p) for p in pq_files],
        hive_partitioning=True,
    )

    available = [c for c in observable_columns if c in df.columns]
    if not available:
        raise ValueError(f"None of {observable_columns} found in Parquet columns: {df.columns[:20]}...")

    if "time" in df.columns:
        df = df.sort("time")

    obs_df = df.select(available).fill_null(0.0)
    ts_array = obs_df.to_numpy().astype(np.float64)

    parquet_to_short = {v: k for k, v in _SHORT_TO_PARQUET.items()}
    obs_names = [parquet_to_short.get(c, c) for c in available]

    return ts_array, obs_names


# -- vEcoli workflow.py runner -----------------------------------------------


def _get_vecoli_repo_root() -> str:
    """Find the vEcoli repo root directory.

    vEcoli's ``configs`` package and ``runscripts/workflow.py`` require
    the subprocess to run from the vEcoli repo root.
    """
    try:
        import ecoli

        ecoli_dir = Path(ecoli.__file__).resolve().parent
        repo_root = ecoli_dir.parent
        if (repo_root / "configs" / "__init__.py").exists():
            return str(repo_root)
    except (ImportError, AttributeError):
        pass
    raise RuntimeError(
        "Cannot find vEcoli repo root. Ensure the ecoli package is "
        "installed and its parent directory contains configs/__init__.py"
    )


def _build_workflow_config(
    sim_data_path: str,
    experiment_id: str,
    output_dir: str,
    max_duration: float,
    variants_section: dict[str, Any],
    n_init_sims: int = 1,
    generations: int = 1,
) -> dict[str, Any]:
    """Build a vEcoli workflow config JSON for ``runscripts/workflow.py``.

    This config inherits from ``default.json`` (via vEcoli's config
    merging) and overrides the keys needed for UQ batch execution.

    Args:
        sim_data_path: Absolute path to baseline simData.cPickle.
        experiment_id: Unique experiment identifier.
        output_dir: Absolute path for Parquet output.
        max_duration: Simulation duration in seconds.
        variants_section: The ``"variants"`` dict for the config,
            structured per vEcoli's variant system (see
            ``runscripts/create_variants.parse_variants``).
        n_init_sims: Number of initial seeds per variant.
        generations: Number of cell generations.

    Returns:
        Config dict ready to be written as JSON.
    """
    # Set batch_size small enough that short sims still flush parquet.
    # Default is 400 emits (=400s at 1s timestep). For UQ we want
    # results even from short sims.
    batch_size = max(1, int(max_duration))

    return {
        "sim_data_path": sim_data_path,
        "experiment_id": experiment_id,
        "emitter": "parquet",
        "emitter_arg": {"out_dir": output_dir, "batch_size": batch_size},
        "max_duration": max_duration,
        "n_init_sims": n_init_sims,
        "generations": generations,
        "single_daughters": True,
        "suffix_time": False,
        "variants": variants_section,
    }


def _build_variants_section_generic(
    X: np.ndarray,
    param_specs: list,
) -> dict[str, Any]:
    """Build the ``variants`` config section for generic sim_data params.

    Uses the ``sim_data_setattr`` variant function with ``op: "zip"``
    to encode N LHS samples.  Each sample becomes a variant with a
    ``mutations`` dict mapping dot-paths to values.

    Args:
        X: LHS sample array, shape ``(n_samples, n_params)``.
        param_specs: List of ``SimDataParameter`` specs.

    Returns:
        Dict suitable for the ``"variants"`` key in a workflow config.
    """
    n_samples = X.shape[0]

    # Build one mutations dict per sample
    mutations_list = []
    for i in range(n_samples):
        mutations = {}
        for j, spec in enumerate(param_specs):
            val = float(X[i, j])
            if spec.index is not None:
                mutations[spec.attr_path] = {"__index__": spec.index, "__value__": val}
            else:
                mutations[spec.attr_path] = val
        mutations_list.append(mutations)

    # Single parameter "mutations" with a list of N dicts → N variants
    return {
        "sim_data_setattr": {
            "mutations": {"value": mutations_list},
        }
    }


def _run_workflow(
    config_path: Path,
    timeout: float | None = None,
) -> subprocess.CompletedProcess:
    """Run ``runscripts/workflow.py`` as a subprocess.

    Args:
        config_path: Absolute path to the workflow config JSON.
        timeout: Timeout in seconds (None = no timeout).

    Returns:
        CompletedProcess result.

    Raises:
        RuntimeError: If the subprocess exits with non-zero status.
    """
    vecoli_root = _get_vecoli_repo_root()
    workflow_script = os.path.join(vecoli_root, "runscripts", "workflow.py")

    cmd = [
        sys.executable,
        workflow_script,
        "--config",
        str(config_path),
    ]

    env = os.environ.copy()
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = vecoli_root + (os.pathsep + existing if existing else "")

    logger.info("Running vEcoli workflow: %s", " ".join(cmd))

    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        cwd=vecoli_root,
        env=env,
        timeout=timeout,
    )

    if result.returncode != 0:
        raise RuntimeError(
            f"vEcoli workflow failed (exit code {result.returncode}).\n"
            f"Command: {' '.join(cmd)}\n"
            f"Stderr:\n{result.stderr[-3000:]}"
        )

    return result


# -- Simulation wrapper class -----------------------------------------------


@dataclass
class ITimeseriesBatchProcessor(BaseClass, abc.ABC):
    @abc.abstractmethod
    def _apply_parameter_mutations(self, x: np.ndarray) -> Any:
        pass

    @abc.abstractmethod
    def _run_batch(
        self,
        X: np.ndarray,
        max_workers: int | None = None,
        batch_dir: Path | None = None,
    ) -> tuple[np.ndarray, list[np.ndarray] | None, list[dict[str, np.ndarray]] | None]:
        pass


@dataclass
class TimeseriesGenerator(ITimeseriesBatchProcessor):
    param_space: Any
    max_duration: float
    output_keys: list[str] | None = None
    _obs_names: list[str] | None = field(default=None, init=False, repr=False)

    def _apply_parameter_mutations(self, x: np.ndarray) -> Any:
        pass

    def _run_batch(
        self,
        X: np.ndarray,
        max_workers: int | None = None,
        batch_dir: Path | None = None,
    ) -> tuple[np.ndarray, list[np.ndarray] | None, list[dict[str, np.ndarray]] | None]:
        raise NotImplementedError

    def evaluate_batch(self, X: np.ndarray, max_workers: int | None = None) -> np.ndarray:
        Y_agg, _, _ = self._run_batch(X, max_workers=max_workers)
        return Y_agg


@dataclass
class TimeseriesGeneratorVecoli(ITimeseriesBatchProcessor):
    """Parameter-agnostic vEcoli simulation wrapper for the UQ pipeline.

    .. deprecated::
       Use ``V2ecoliGenerator`` from ``uq.generators.v2ecoli`` instead.
       ``--backend vecoli`` still works but is no longer actively developed.

    All simulations are executed via vEcoli's ``runscripts/workflow.py``
    (Nextflow orchestration).  No ``EcoliSim`` is ever instantiated
    in the UQ process.

    The variant system (``ecoli.variants.sim_data_setattr``) is used
    to encode LHS samples as variant parameter dicts with ``op: "zip"``.

    Implements:
    - ``__call__(x)`` -> raw timeseries ``(n_timesteps, n_obs)``
    - ``evaluate_batch(X)`` -> aggregated ``(n_samples, n_outputs)``
    - ``_run_batch(X)`` -> full batch via single workflow.py invocation
    """

    baseline_sim_data: SimulationDataEcoli
    param_space: Any  # XSpace — avoid circular import
    sim_config_path: str | None = None
    max_duration: float = 10800.0
    generations: int = 1
    n_init_sims: int = 1
    output_keys: list[str] | None = None
    observable_names: list[str] | None = None
    _obs_names: list[str] | None = field(default=None, init=False, repr=False)

    def __post_init__(self):
        warnings.warn(
            "TimeseriesGeneratorVecoli is deprecated. "
            "Use ``uq.generators.v2ecoli.V2ecoliGenerator`` with ``--backend v2ecoli`` instead.",
            DeprecationWarning,
            stacklevel=2,
        )

    @property
    def parameter_names(self) -> list[str]:
        return self.param_space.parameter_names

    @property
    def obs_names(self) -> list[str]:
        if self._obs_names is None:
            raise RuntimeError("Observable names not yet known. Run at least one simulation.")
        return self._obs_names

    def _get_sim_data_path(self) -> str:
        """Get the path to baseline simData.cPickle.

        If we have a path from the ParameterDataset, use it directly.
        Otherwise, write the in-memory sim_data to a temp pickle.
        """
        # Check if param_space or the loader has a path
        # For now, we write to a stable temp location
        from libuq.common import get_repo_root

        cache_dir = get_repo_root() / ".uq_cache"
        cache_dir.mkdir(exist_ok=True)
        pickle_path = cache_dir / "baseline_sim_data.cPickle"
        if not pickle_path.exists():
            with open(pickle_path, "wb") as f:
                pickle.dump(self.baseline_sim_data, f)
        return str(pickle_path)

    def _apply_parameter_mutations(self, x: np.ndarray) -> SimulationDataEcoli:
        """Deep-copy baseline sim_data and apply mutations from x."""
        sd = copy.deepcopy(self.baseline_sim_data)
        uq_params = self.param_space.sample_to_params(x)
        sim_config = uq_params.to_simulation_config()

        mutations = sim_config.get("sim_data_mutations", {})
        if mutations:
            _apply_sim_data_mutations(sd, mutations)

        variants = sim_config.get("variants", {})
        if variants:
            sd = _apply_variants(sd, variants)

        return sd

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """Run simulation for a single parameter vector via workflow.py.

        Returns raw timeseries of shape ``(n_timesteps, n_obs)``.
        """
        X = x.reshape(1, -1)
        Y_agg, Y_ts, _ = self._run_batch(X)
        if Y_ts and len(Y_ts) > 0:
            return Y_ts[0]
        return Y_agg

    def _run_batch(
        self,
        X: np.ndarray,
        max_workers: int | None = None,
        batch_dir: Path | None = None,
    ) -> tuple[np.ndarray, list[np.ndarray] | None, list[dict[str, np.ndarray]] | None]:
        """Run simulations via a single ``workflow.py`` invocation.

        1. Build a workflow config with variants section encoding all
           N LHS samples via ``sim_data_setattr`` + ``op: "zip"``
        2. Run ``runscripts/workflow.py --config <config>``
        3. Collect Parquet outputs from hive-partitioned directory
        4. Return aggregated + per-sample timeseries arrays + metadata

        Args:
            X: Parameter array of shape ``(n_samples, n_params)``.
            max_workers: Unused (Nextflow handles parallelism).
            batch_dir: Working directory. If None, creates temp dir.

        Returns:
            Tuple of (Y_aggregated, Y_timeseries, Y_timeseries_meta).
        """
        cleanup = batch_dir is None
        if batch_dir is None:
            batch_dir = Path(tempfile.mkdtemp(prefix="uq_batch_"))
        batch_dir = batch_dir.resolve()

        n_samples = X.shape[0]
        parquet_cols = _resolve_parquet_columns(self.output_keys)
        experiment_id = "uq_batch"
        if isinstance(batch_dir, str):
            batch_dir = Path(batch_dir)
        output_dir = batch_dir / "output"
        output_dir.mkdir(parents=True, exist_ok=True)

        try:
            # --- Step 1: Write baseline sim_data pickle ---
            # Must be in a kb/ subdirectory so workflow.py's
            # file(kb_dir).copyTo(...) copies only the kb/ dir,
            # not the entire batch_dir.
            kb_dir = batch_dir / "kb"
            kb_dir.mkdir(exist_ok=True)
            sim_data_path = str(kb_dir / "simData.cPickle")
            with open(sim_data_path, "wb") as f:
                pickle.dump(self.baseline_sim_data, f)

            # --- Step 2: Build variants section ---
            variants_section = _build_variants_section_generic(
                X,
                self.param_space._sim_data_parameters,
            )

            # --- Step 3: Build workflow config ---
            config = _build_workflow_config(
                sim_data_path=sim_data_path,
                experiment_id=experiment_id,
                output_dir=str(output_dir),
                max_duration=self.max_duration,
                variants_section=variants_section,
                n_init_sims=self.n_init_sims,
                generations=self.generations,
            )

            config_path = batch_dir / "workflow_config.json"
            config_path.write_text(_json.dumps(config, indent=2))

            logger.info(
                "Running vEcoli workflow with %d variants (samples) in %s",
                n_samples,
                batch_dir,
            )

            # --- Step 4: Run workflow.py ---
            # Nextflow may exit non-zero if some (but not all) variant
            # sims fail.  We tolerate this and collect whatever results
            # are available, warning about missing variants later.
            timeout = max(self.max_duration * n_samples * 2, 600)
            try:
                _run_workflow(config_path, timeout=timeout)
            except RuntimeError as e:
                logger.warning(
                    "Workflow exited with errors (some variants may have "
                    "failed). Collecting available results. Error: %s",
                    str(e)[:500],
                )

            # --- Step 5: Collect Parquet outputs ---
            # workflow.py output structure:
            # {output_dir}/{experiment_id}/history/experiment_id={id}/variant={N}/...
            # Variant 0 = baseline, 1..N = our samples
            history_base = output_dir / experiment_id / "history"
            if not history_base.exists():
                # workflow.py may append timestamp to experiment_id
                candidates = list(output_dir.glob("*/history"))
                if candidates:
                    history_base = candidates[0]

            pq_files = list(history_base.rglob("*.pq"))
            if not pq_files:
                pq_files = list(history_base.rglob("*.parquet"))
            if not pq_files:
                raise FileNotFoundError(
                    f"No Parquet files found under {history_base}. Workflow stderr may have details."
                )

            df = pl.read_parquet(
                [str(p) for p in pq_files],
                hive_partitioning=True,
            )

            available = [c for c in parquet_cols if c in df.columns]
            if not available:
                raise ValueError(f"None of {parquet_cols} found in columns: {df.columns[:20]}...")

            parquet_to_short = {v: k for k, v in _SHORT_TO_PARQUET.items()}
            obs_names = [parquet_to_short.get(c, c) for c in available]
            self._obs_names = obs_names

            sort_cols = []
            if "variant" in df.columns:
                sort_cols.append("variant")
            if "lineage_seed" in df.columns:
                sort_cols.append("lineage_seed")
            if "generation" in df.columns:
                sort_cols.append("generation")
            if "time" in df.columns:
                sort_cols.append("time")
            if sort_cols:
                df = df.sort(sort_cols)

            Y_list: list[np.ndarray] = []
            Y_timeseries: list[np.ndarray] = []
            Y_meta: list[dict[str, np.ndarray]] = []

            # Use baseline (variant=0) as fallback for failed variants
            baseline_df = None
            if "variant" in df.columns:
                baseline_df = df.filter(pl.col("variant") == 0)

            for i in range(n_samples):
                # Variant 0 = baseline, 1..N = our samples
                variant_idx = i + 1
                if "variant" in df.columns:
                    sample_df = df.filter(pl.col("variant") == variant_idx)
                    if sample_df.height == 0:
                        sample_df = df.filter(pl.col("variant") == i)
                    if sample_df.height == 0 and baseline_df is not None:
                        logger.warning(
                            "Variant %d has no data (sim may have crashed). Using baseline data as fallback.",
                            variant_idx,
                        )
                        sample_df = baseline_df
                else:
                    sample_df = df

                obs_df = sample_df.select(available).fill_null(0.0)
                ts_array = obs_df.to_numpy().astype(np.float64)
                Y_timeseries.append(ts_array)
                Y_list.append(ts_array.mean(axis=0))

                # Extract generation/seed metadata for strategies 2-3
                sample_meta: dict[str, np.ndarray] = {}
                if "generation" in sample_df.columns:
                    sample_meta["generation"] = sample_df["generation"].fill_null(0).to_numpy().astype(np.int64)
                if "lineage_seed" in sample_df.columns:
                    sample_meta["lineage_seed"] = sample_df["lineage_seed"].fill_null(0).to_numpy().astype(np.int64)
                Y_meta.append(sample_meta)

            Y_aggregated = np.vstack(Y_list)

            meta = {
                "parameter_names": self.parameter_names,
                "n_samples": n_samples,
                "n_params": int(X.shape[1]),
                "observable_columns": available,
                "obs_names": obs_names,
                "bounds": np.array(self.param_space.parameter_bounds).tolist(),
                "samples": {str(i): {"x": X[i].tolist()} for i in range(n_samples)},
            }
            (batch_dir / "metadata.json").write_text(_json.dumps(meta, indent=2))

            # Return metadata only if any sample had generation/seed labels
            has_meta = Y_meta and any(m for m in Y_meta)
            return Y_aggregated, Y_timeseries, Y_meta if has_meta else None

        finally:
            if cleanup:
                shutil.rmtree(batch_dir, ignore_errors=True)

    def evaluate_batch(
        self,
        X: np.ndarray,
        max_workers: int | None = None,
    ) -> np.ndarray:
        """Evaluate simulation for a batch of parameter vectors."""
        Y_agg, _, _ = self._run_batch(X, max_workers=max_workers)
        return Y_agg


# -- Batch config export / collection --------------------------------------


def export_batch_configs(
    sim_func: "TimeseriesGeneratorVecoli",
    X: np.ndarray,
    batch_dir: str | os.PathLike,
    base_config_path: str | None = None,
    generations: int = 1,
    emitter: str = "parquet",
) -> Path:
    """Convert LHS samples into per-sample vEcoli configs for Nextflow/HPC."""
    batch_dir = Path(batch_dir)
    config_dir = batch_dir / "configs"
    sim_data_dir = batch_dir / "sim_data"
    config_dir.mkdir(parents=True, exist_ok=True)
    sim_data_dir.mkdir(parents=True, exist_ok=True)

    base_config: dict = {}
    if base_config_path is not None:
        base_config = _json.loads(Path(base_config_path).read_text())

    sample_metadata: dict = {
        "parameter_names": sim_func.parameter_names,
        "n_samples": int(X.shape[0]),
        "n_params": int(X.shape[1]),
        "bounds": np.array(sim_func.param_space.parameter_bounds).tolist(),
        "samples": {},
    }

    for i, x in enumerate(X):
        uq_params = sim_func.param_space.sample_to_params(x)
        sim_config = uq_params.to_simulation_config()
        variants = sim_config.get("variants", {})

        sd = copy.deepcopy(sim_func.baseline_sim_data)
        if variants:
            sd = _apply_variants(sd, variants)

        pickle_path = sim_data_dir / f"{i}.cPickle"
        with open(pickle_path, "wb") as f:
            pickle.dump(sd, f)

        sample_config = {**base_config}
        sample_config["experiment_id"] = f"uq_sample_{i:04d}"
        sample_config["sim_data_path"] = str(pickle_path)
        sample_config["generations"] = generations
        sample_config["emitter"] = emitter
        sample_config["variants"] = {}

        config_path = config_dir / f"{i}.json"
        config_path.write_text(_json.dumps(sample_config, indent=2))

        sample_metadata["samples"][str(i)] = {
            "x": x.tolist(),
            "config": str(config_path),
            "sim_data": str(pickle_path),
        }

    (batch_dir / "metadata.json").write_text(_json.dumps(sample_metadata, indent=2))
    return batch_dir


def collect_batch_results(
    batch_dir: str | os.PathLike,
    output_dir: str | os.PathLike,
    observable_columns: list[str] | None = None,
    cache_dir: str | os.PathLike | None = None,
) -> "PrecomputedCache":  # noqa: F821
    """Collect Parquet outputs from a completed batch run into PrecomputedCache."""
    from libuq.sampling import PrecomputedCache

    batch_dir = Path(batch_dir)
    output_dir = Path(output_dir)

    meta = _json.loads((batch_dir / "metadata.json").read_text())
    n_samples = meta["n_samples"]
    parameter_names = meta["parameter_names"]
    bounds = meta["bounds"]

    if observable_columns is None:
        observable_columns = list(_DEFAULT_PARQUET_COLUMNS)

    X_list = []
    Y_list = []
    Y_timeseries = []

    for i in range(n_samples):
        sample_meta = meta["samples"][str(i)]
        x = np.array(sample_meta["x"])
        X_list.append(x)

        experiment_id = f"uq_sample_{i:04d}"
        history_dir = output_dir / experiment_id / "history"

        if not history_dir.exists():
            raise FileNotFoundError(f"No output found for sample {i} at {history_dir}.")

        pq_files = list(history_dir.rglob("*.pq"))
        if not pq_files:
            pq_files = list(history_dir.rglob("*.parquet"))

        df = pl.read_parquet(
            [str(p) for p in pq_files],
            hive_partitioning=True,
        )

        available = [c for c in observable_columns if c in df.columns]
        if not available:
            raise ValueError(f"None of {observable_columns} found in columns: {df.columns}")

        obs_df = df.select(available).fill_null(0.0)
        ts_array = obs_df.to_numpy()
        Y_timeseries.append(ts_array)
        Y_list.append(ts_array.mean(axis=0))

    X = np.vstack(X_list)
    Y = np.vstack(Y_list)

    if cache_dir is None:
        cache_dir = batch_dir / "cache"

    cache = PrecomputedCache(
        cache_dir=Path(cache_dir),
        X=X,
        Y=Y,
        parameter_names=parameter_names,
        metadata={"bounds": bounds, "seed": -1, "source": "batch_collection"},
        Y_timeseries=Y_timeseries,
    )
    cache.save()
    return cache
