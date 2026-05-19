"""In-process v2ecoli simulation generator for the UQ pipeline.

Replaces the subprocess-based vEcoli backend (TimeseriesGeneratorVecoli)
with in-process v2ecoli composites driven by process-bigraph.

Phases 1-3: cache bundle generation, in-process execution, observable extraction.
"""

from __future__ import annotations

import copy
import gc
import logging
import os
import shutil
import tempfile
import time as _time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import dill
import numpy as np
from process_bigraph import Composite

from v2ecoli.cache import save_initial_state, load_initial_state
from v2ecoli.core import build_core, load_cache_bundle, save_cache
from v2ecoli.composites.baseline import baseline as v2ecoli_baseline

from uq.v2ecoli_bridge import (
    OBSERVABLE_PATHS,
    ARRAY_OBSERVABLE_PATHS,
    get_observable_from_state,
    get_array_observable_from_state,
    is_array_preset,
    is_scalar_preset,
)

logger = logging.getLogger(__name__)

_PROCESS_CONFIG_MAP: dict[str, str] = {
    "metabolism": "ecoli-metabolism",
    "transcription": "ecoli-transcript-initiation",
    "translation": "ecoli-polypeptide-elongation",
    "replication": "ecoli-chromosome-replication",
}

_MASS_CONFIG_MAP: dict[str, str] = {
    "cell_dry_mass_fraction": "ecoli-mass-listener",
    "cell_water_mass_fraction": "ecoli-mass-listener",
}


# ═══════════════════════════════════════════════════════════════════
# Phase 1: Cache bundle generation
# ═══════════════════════════════════════════════════════════════════


def generate_cache_bundle(sim_data_path: str, cache_dir: str, seed: int = 0) -> str:
    """Generate v2ecoli cache bundle from simData.cPickle.

    Calls v2ecoli.core.save_cache() to produce the ParCa bundle
    (initial_state.json + sim_data_cache.dill) that all composites
    load at build time.

    Args:
        sim_data_path: Path to simData.cPickle.
        cache_dir: Output directory for the cache bundle.
        seed: RNG seed for stochastic initialisation.

    Returns:
        Absolute path to the cache directory.
    """
    cache_dir = str(Path(cache_dir).resolve())
    save_cache(sim_data_path, cache_dir, seed=seed)
    logger.info("Cache bundle generated at %s", cache_dir)
    return cache_dir


# ═══════════════════════════════════════════════════════════════════
# Phase 1: Parameter mutation
# ═══════════════════════════════════════════════════════════════════


def _apply_mutation_to_configs(configs: dict, dot_path: str, value: float) -> None:
    """Apply a sim_data dot-path mutation to the process configs dict.

    Maps dot-paths like ``process.metabolism.kinetic_objective_weight``
    to the corresponding v2ecoli process config keys.

    Args:
        configs: The ``configs`` dict from the cache bundle (mutated in-place).
        dot_path: Dot-path attribute on ``SimulationDataEcoli``.
        value: New scalar value.
    """
    parts = dot_path.split(".")

    if parts[0] == "process" and len(parts) >= 3:
        process_name = parts[1]
        config_key = _PROCESS_CONFIG_MAP.get(process_name)
        if config_key is None:
            logger.warning("No v2ecoli config mapping for process %r", process_name)
            return

        proc_config = configs.get(config_key)
        if not isinstance(proc_config, dict):
            logger.warning("Config %r not found or not a dict in cache bundle", config_key)
            return

        target_key = parts[-1]
        if len(parts) > 3:
            target = proc_config
            for attr_part in parts[2:-1]:
                if isinstance(target, dict):
                    target = target.get(attr_part, {})
                else:
                    logger.warning("Cannot traverse path %r in config %r", dot_path, config_key)
                    return
            if isinstance(target, dict):
                target[target_key] = float(value)
        else:
            proc_config[target_key] = float(value)

        logger.debug("Mutated %s → %s in %s", dot_path, value, config_key)

    elif parts[0] == "mass" and len(parts) >= 2:
        mass_key = parts[1]
        config_key = _MASS_CONFIG_MAP.get(mass_key)
        if config_key and config_key in configs and isinstance(configs[config_key], dict):
            configs[config_key][mass_key] = float(value)
            logger.debug("Mutated %s → %s in %s", dot_path, value, config_key)
        else:
            logger.warning("No v2ecoli config mapping for mass attribute %r", mass_key)

    else:
        logger.debug("Unrecognized mutation path %s — skipping", dot_path)


def _apply_mutations_to_bundle(
    bundle: dict,
    mutations: dict[str, float],
) -> dict:
    """Deep-copy a cache bundle and apply mutations to its configs.

    Args:
        bundle: Cache bundle dict from ``load_cache_bundle()``.
        mutations: Dict mapping dot-paths to new values.

    Returns:
        New bundle dict with mutated configs (original unchanged).
    """
    configs = copy.deepcopy(bundle.get("configs", {}))
    for dot_path, value in mutations.items():
        _apply_mutation_to_configs(configs, dot_path, value)

    return {
        **bundle,
        "configs": configs,
    }


# ═══════════════════════════════════════════════════════════════════
# Phase 1: Composite building with mutated cache
# ═══════════════════════════════════════════════════════════════════


def _write_mutated_cache(
    source_cache_dir: str,
    target_dir: str,
    mutations: dict[str, float],
) -> str:
    """Write a mutated cache bundle to a target directory.

    Loads the source cache bundle, applies mutations in-memory,
    and writes the mutated configs to a new sim_data_cache.dill
    in the target directory. The initial_state.json is symlinked
    (unchanged) to save space.

    Args:
        source_cache_dir: Baseline cache directory.
        target_dir: Output directory for the mutated cache.
        mutations: Dict mapping dot-paths to new values.

    Returns:
        Absolute path to the mutated cache directory.
    """
    target = Path(target_dir)
    target.mkdir(parents=True, exist_ok=True)

    bundle = load_cache_bundle(source_cache_dir)
    mutated_bundle = _apply_mutations_to_bundle(bundle, mutations)

    unique_names = mutated_bundle.get("unique_names", [])
    dry_mass_inc = mutated_bundle.get("dry_mass_inc_dict", {})

    cache_payload = {
        "configs": mutated_bundle["configs"],
        "unique_names": unique_names,
        "dry_mass_inc_dict": dry_mass_inc,
    }
    cache_path = target / "sim_data_cache.dill"
    with open(cache_path, "wb") as f:
        dill.dump(cache_payload, f)

    # Symlink initial_state.json (unchanged by config mutations)
    src_initial = Path(source_cache_dir) / "initial_state.json"
    dst_initial = target / "initial_state.json"
    if src_initial.exists():
        if dst_initial.exists():
            dst_initial.unlink()
        os.symlink(str(src_initial.resolve()), str(dst_initial))

    logger.debug("Mutated cache written to %s", target)
    return str(target.resolve())


def build_v2ecoli_composite(
    cache_dir: str,
    seed: int = 0,
    mutations: dict[str, float] | None = None,
    core: Any | None = None,
    features: list[str] | None = None,
) -> Composite:
    """Build a v2ecoli composite with optional parameter mutations.

    Applies mutations post-cache, pre-composite (option b from the
    migration plan): loads the cache bundle, mutates configs in-memory,
    writes a temporary mutated cache, and builds the composite from it.

    Args:
        cache_dir: Baseline cache directory (must contain initial_state.json
            and sim_data_cache.dill).
        seed: RNG seed for the composite.
        mutations: Optional dict of dot-path → value mutations.
        core: Optional pre-allocated bigraph-schema core.
        features: Optional list of v2ecoli feature modules.

    Returns:
        Ready-to-run Composite.
    """
    if core is None:
        core = build_core()

    if mutations:
        resolved_cache = _write_mutated_cache(
            cache_dir, _mutated_cache_dir(cache_dir, mutations), mutations
        )
    else:
        resolved_cache = cache_dir

    doc = v2ecoli_baseline(core=core, seed=seed, cache_dir=resolved_cache)
    return Composite(doc, core=core)


def _mutated_cache_dir(cache_dir: str, mutations: dict[str, float]) -> str:
    """Return a deterministic temp dir path for a given mutation set."""
    import hashlib
    mutation_hash = hashlib.md5(
        "&".join(f"{k}={v}" for k, v in sorted(mutations.items())).encode()
    ).hexdigest()[:12]
    return os.path.join(
        os.path.dirname(cache_dir),
        f".mutated_{mutation_hash}",
    )


# ═══════════════════════════════════════════════════════════════════
# Phase 2: In-process composite execution
# ═══════════════════════════════════════════════════════════════════


def run_v2ecoli_composite(
    composite: Composite,
    n_generations: int = 10,
    max_time: float = 10000.0,
    interval: float = 1.0,
    division_callback: Callable | None = None,
    presets: list[str] | None = None,
    array_preset_cache: dict[str, int] | None = None,
) -> dict[str, Any]:
    """Run a v2ecoli composite and extract observable timeseries.

    Advances the composite tick-by-tick, detecting division events
    via the ``divide`` flag in agent state, and collecting listener
    snapshots at each time step.

    Args:
        composite: Ready-to-run v2ecoli Composite.
        n_generations: Maximum number of cell generations to simulate.
        max_time: Maximum simulation time in seconds.
        interval: Tick interval in seconds.
        division_callback: Optional callback ``(agent_id, state)``
            called when a division event is detected.
        presets: Observable presets to extract (e.g. ``["mass"]``).
            If None, extracts all scalar observables.
        array_preset_cache: Mutable dict used on first tick to cache
            the size of each array preset, e.g. ``{"transcriptome": 4345}``.
            Pass a shared dict across calls to avoid re-computing sizes.

    Returns:
        Dict with keys:
            - ``timeseries``: ``{observable_name: [values_per_tick]}``
            - ``metadata``: ``{sample_idx, generation_counts, agent_ids}``
            - ``generations``: Number of completed generations.
            - ``final_time``: Simulation end time.
    """
    logger.info("Starting v2ecoli composite run (%d gens, %.0f s max)",
                n_generations, max_time)

    t = 0.0
    generation_count = 0
    max_generation = 0
    timeseries: dict[str, list[float]] = {}
    agent_ids_seen: set[str] = set()

    if presets is None:
        presets = ["mass"]

    if array_preset_cache is None:
        array_preset_cache = {}

    # Pre-populate timeseries keys
    _init_timeseries_keys(timeseries, presets)

    while t < max_time and generation_count < n_generations:
        try:
            composite.run(interval)
        except Exception as e:
            err_str = str(e).lower()
            if "divide" in err_str or "division" in err_str:
                logger.info("Division detected at t=%.1f", t)
                generation_count += 1
                if division_callback:
                    division_callback("0", composite.state)
                _handle_division(composite, t)
            else:
                logger.warning("Composite run error at t=%.1f: %s", t, e)
                break

        t = composite.state.get("global_time", t)

        agents = composite.state.get("agents", {})
        for agent_id, cell_state in agents.items():
            agent_ids_seen.add(agent_id)

            # ── Extract scalar observables ──
            for preset_name in presets:
                if is_scalar_preset(preset_name):
                    for obs_name in OBSERVABLE_PATHS:
                        val = get_observable_from_state(cell_state, obs_name)
                        if obs_name not in timeseries:
                            timeseries[obs_name] = []
                        timeseries[obs_name].append(val)

            # ── Extract array observables (flattened) ──
            for preset_name in presets:
                if is_array_preset(preset_name):
                    arr = get_array_observable_from_state(cell_state, preset_name)
                    if arr is not None and len(arr) > 0:
                        if preset_name not in array_preset_cache:
                            array_preset_cache[preset_name] = len(arr)
                        for i in range(len(arr)):
                            key = f"{preset_name}_{i}"
                            if key not in timeseries:
                                timeseries[key] = []
                            timeseries[key].append(float(arr[i]))
                    else:
                        # Array not yet populated — pad with zeros
                        n = array_preset_cache.get(preset_name, 0)
                        for i in range(n):
                            key = f"{preset_name}_{i}"
                            if key not in timeseries:
                                timeseries[key] = []
                            timeseries[key].append(0.0)

        gen = agent_id.count("_")
        max_generation = max(max_generation, gen)

        cell = agents.get("0", {})
        bulk = cell.get("bulk", None)
        if bulk is not None and hasattr(bulk, "size") and bulk.size > 0:
            if np.all(bulk == 0):
                logger.info("Bulk molecules exhausted at t=%.1f", t)
                break

    aggregated = {k: np.mean(v) if v else 0.0 for k, v in timeseries.items()}

    return {
        "timeseries": timeseries,
        "aggregated": aggregated,
        "metadata": {
            "generation_count": max_generation,
            "agent_ids": sorted(agent_ids_seen),
        },
        "generations": max_generation,
        "final_time": t,
    }


def _init_timeseries_keys(
    timeseries: dict[str, list[float]],
    presets: list[str] | None = None,
) -> None:
    """Initialize timeseries keys from the observable path map."""
    if presets is None:
        presets = ["mass"]
    for preset_name in presets:
        if is_scalar_preset(preset_name):
            for obs_name in OBSERVABLE_PATHS:
                if obs_name not in timeseries:
                    timeseries[obs_name] = []
        # Array preset keys are created lazily on first extraction


def _handle_division(composite: Composite, current_time: float) -> None:
    """Reset composite state after a division event.

    After division, the daughter cell takes over as agent ``0``.
    """
    agents = composite.state.get("agents", {})
    # After division, the original agent may have been replaced
    # or new agents added. We re-base on the first available agent.
    if "0" not in agents and agents:
        first_id = sorted(agents.keys())[0]
        composite.state["agents"]["0"] = agents[first_id]


def extract_timeseries(
    result: dict[str, Any],
    observable_names: list[str] | None = None,
) -> np.ndarray:
    """Convert a run result to the UQ pipeline's Y_timeseries format.

    Args:
        result: Output of ``run_v2ecoli_composite()``.
        observable_names: Subset of observables to extract.
            If None, uses all available.

    Returns:
        ``(n_timesteps, n_observables)`` numpy array matching the
        current Parquet-based timeseries format.
    """
    ts = result["timeseries"]
    if observable_names is None:
        observable_names = sorted(ts.keys())

    n_steps = max((len(ts.get(n, [])) for n in observable_names), default=0)
    if n_steps == 0:
        return np.zeros((1, len(observable_names)))

    arrays = []
    for name in observable_names:
        vals = ts.get(name, [])
        if len(vals) < n_steps:
            vals = list(vals) + [0.0] * (n_steps - len(vals))
        arrays.append(np.array(vals[:n_steps], dtype=np.float64))

    return np.column_stack(arrays)


# ═══════════════════════════════════════════════════════════════════
# Phase 2: Multiprocessing support
# ═══════════════════════════════════════════════════════════════════


def _run_single_sample(args: tuple) -> tuple[int, np.ndarray, dict[str, Any], list[str]]:
    """Run one parameter sample through v2ecoli.

    Standalone function for multiprocessing support. Builds the
    composite, runs it, and returns aggregated observables.

    Args:
        args: ``(idx, x_vec, cache_dir, param_names, seed, n_generations,
            max_time, presets)``

    Returns:
        ``(idx, y_aggregated, metadata, obs_names)``
    """
    idx, x_vec, cache_dir, param_names, seed, n_generations, max_time, presets = args

    mutations = {}
    for j, name in enumerate(param_names):
        mutations[name] = float(x_vec[j])

    array_cache: dict[str, int] = {}
    sample_seed = seed + idx
    composite = build_v2ecoli_composite(
        cache_dir=cache_dir,
        seed=sample_seed,
        mutations=mutations,
    )

    result = run_v2ecoli_composite(
        composite,
        n_generations=n_generations,
        max_time=max_time,
        presets=presets,
        array_preset_cache=array_cache,
    )

    del composite
    gc.collect()

    obs_names = sorted(result["timeseries"].keys())
    y = np.array([result["aggregated"].get(n, 0.0) for n in obs_names], dtype=np.float64)

    return idx, y, result["metadata"], obs_names


# ═══════════════════════════════════════════════════════════════════
# Phase 1-2: V2ecoliGenerator class
# ═══════════════════════════════════════════════════════════════════


@dataclass
class V2ecoliGenerator:
    """In-process v2ecoli simulation generator for the UQ pipeline.

    Matches the ``ITimeseriesBatchProcessor`` interface from
    ``libuq.generators.vecoli`` so it can be used as a drop-in
    replacement for ``TimeseriesGeneratorVecoli``.

    Internally builds and runs v2ecoli composites in-process,
    avoiding subprocess + Nextflow overhead.

    Usage::

        gen = V2ecoliGenerator(
            cache_dir="./uq_cache",
            param_names=["process.metabolism.kinetic_objective_weight"],
            seed=42,
        )
        Y = gen.evaluate_batch(X)  # (n_samples, n_observables)
    """

    cache_dir: str
    param_names: list[str]
    seed: int = 0
    n_generations: int = 10
    max_time: float = 10000.0
    max_workers: int | None = None
    presets: list[str] | None = None
    _obs_names: list[str] | None = field(default=None, init=False, repr=False)

    @property
    def obs_names(self) -> list[str]:
        if self._obs_names is None:
            raise RuntimeError("Observable names not yet known. Run at least one simulation.")
        return self._obs_names

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """Run simulation for a single parameter vector.

        Args:
            x: Parameter vector of shape ``(n_params,)``.

        Returns:
            Raw timeseries of shape ``(n_timesteps, n_obs)``.
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
        """Run a batch of parameter vectors through in-process v2ecoli.

        Args:
            X: Parameter array of shape ``(n_samples, n_params)``.
            max_workers: Max parallel workers. If None, uses
                ``self.max_workers`` (defaults to 1).
            batch_dir: Ignored (kept for interface compatibility).

        Returns:
            Tuple of ``(Y_aggregated, Y_timeseries, Y_meta)`` matching
            ``TimeseriesGeneratorVecoli._run_batch()``.
        """
        n_samples = X.shape[0]
        n_workers = max_workers or self.max_workers or 1

        if n_workers > 1 and n_samples > 1:
            return self._run_batch_parallel(X, n_workers)
        return self._run_batch_sequential(X)

    def _run_batch_sequential(
        self,
        X: np.ndarray,
    ) -> tuple[np.ndarray, list[np.ndarray] | None, list[dict[str, np.ndarray]] | None]:
        """Run samples sequentially (single-process)."""
        n_samples = X.shape[0]
        Y_list: list[np.ndarray] = []
        Y_ts_list: list[np.ndarray] = []
        Y_meta_list: list[dict[str, np.ndarray]] = []

        for i in range(n_samples):
            idx, y, meta, obs_names = _run_single_sample((
                i, X[i], self.cache_dir, self.param_names,
                self.seed, self.n_generations, self.max_time, self.presets,
            ))
            ts = y.reshape(1, -1)
            Y_list.append(ts.mean(axis=0))
            Y_ts_list.append(ts)
            Y_meta_list.append({
                "generation": np.array([meta.get("generation_count", 0)], dtype=np.int64),
                "lineage_seed": np.array([self.seed + i], dtype=np.int64),
            })
            if self._obs_names is None:
                self._obs_names = obs_names

        Y_all = np.vstack(Y_list)
        return Y_all, Y_ts_list, Y_meta_list

    def _run_batch_parallel(
        self,
        X: np.ndarray,
        n_workers: int,
    ) -> tuple[np.ndarray, list[np.ndarray] | None, list[dict[str, np.ndarray]] | None]:
        """Run samples in parallel using ``ProcessPoolExecutor``."""
        n_samples = X.shape[0]
        args_list = [
            (i, X[i], self.cache_dir, self.param_names, self.seed,
             self.n_generations, self.max_time, self.presets)
            for i in range(n_samples)
        ]

        Y_list: list[np.ndarray] = [None] * n_samples  # type: ignore[list-item]
        Y_meta_list: list[dict] = [None] * n_samples  # type: ignore[list-item]

        with ProcessPoolExecutor(max_workers=n_workers) as pool:
            futures = {
                pool.submit(_run_single_sample, args): args[0]
                for args in args_list
            }
            for future in as_completed(futures):
                idx, y, meta, obs_names = future.result()
                Y_list[idx] = y
                Y_meta_list[idx] = meta
                if self._obs_names is None:
                    self._obs_names = obs_names

        if self._obs_names is None and Y_list:
            self._obs_names = sorted(OBSERVABLE_PATHS.keys())

        Y_all = np.vstack([y for y in Y_list if y is not None])
        Y_ts_list = [y.reshape(1, -1) for y in Y_list if y is not None]
        Y_meta_clean = [
            {
                "generation": np.array([m.get("generation_count", 0)], dtype=np.int64),
                "lineage_seed": np.array([self.seed + i], dtype=np.int64),
            }
            for i, m in enumerate(Y_meta_list) if m is not None
        ]

        return Y_all, Y_ts_list, Y_meta_clean

    def evaluate_batch(
        self,
        X: np.ndarray,
        max_workers: int | None = None,
    ) -> np.ndarray:
        """Evaluate simulation for a batch of parameter vectors.

        Args:
            X: Parameter array of shape ``(n_samples, n_params)``.
            max_workers: Max parallel workers.

        Returns:
            Aggregated output of shape ``(n_samples, n_observables)``.
        """
        Y_agg, _, _ = self._run_batch(X, max_workers=max_workers)
        return Y_agg
