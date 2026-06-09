"""Top-level worker entry for `uq sample --design-config`'s concurrent
condition runs.

``ProcessPoolExecutor`` requires a picklable, importable callable as the
worker target. Closures and nested functions don't work because the spawn
start method (default on macOS, recommended on Linux for NumPy safety)
re-imports the worker module fresh in each child process.

This module is intentionally small and dependency-light at the top level —
heavyweight imports happen inside ``run_one_condition`` so the parent
process doesn't pay for them and so workers don't inherit any
spawn-incompatible state from the parent's CLI session.

§25-F Phase 3 / Tier 1A — concurrent design-condition orchestration.
"""

from __future__ import annotations

import dataclasses
from typing import Any, Literal


@dataclasses.dataclass
class CondResult:
    """Outcome of one condition's pre-bake + UQ workflow run.

    Attributes:
        j: Index of the condition in the original design entry list
            (preserves ordering across out-of-order completion).
        condition_id: Deterministic ID from ``_design_condition_id``.
        status: ``"ok"`` if the workflow completed and a cache was
            written; ``"failed"`` otherwise.
        n_samples_cached: Rows in the resulting cache's X matrix
            (0 when status="failed").
        wall_clock_seconds: Wall-clock duration for this worker, including
            pre-baking, subprocess execution, and Parquet collection.
        log_tail: Last 50 lines of the per-condition log file (for
            failure diagnosis without leaving the parent's console).
        error: Traceback string if the worker itself raised; ``None`` on
            ``"ok"`` and on workflow-level (non-Python) failures.
    """

    j: int
    condition_id: str
    status: Literal["ok", "failed"]
    n_samples_cached: int
    wall_clock_seconds: float
    log_tail: str
    error: str | None


def _read_log_tail(log_path: Any, n_lines: int = 50) -> str:
    from pathlib import Path
    p = Path(log_path)
    if not p.exists():
        return ""
    try:
        with p.open("r", errors="replace") as f:
            lines = f.readlines()
    except OSError:
        return ""
    return "".join(lines[-n_lines:])


def run_one_condition(args: dict[str, Any]) -> CondResult:
    """Run one design-condition's pre-bake + UQ workflow in a worker process.

    Args:
        args: A dict of all the parameters needed for the worker.
            Using a single dict argument (rather than many keyword args)
            keeps the ``ProcessPoolExecutor.submit`` call clean and makes
            the worker easy to mock in tests. Required keys mirror the
            parameters of ``uq.cli._sample_local`` plus a few extras:

            j (int): condition index for ordering
            condition_id (str)
            cond_dir_str (str): cache_dir/condition_<id>/ as string
            sim_data_path (str): baseline simData.cPickle
            design_module_name (str)
            design_params (dict)
            X_train, X_test, germ_train, germ_test (np.ndarray | None)
            param_specs (list[dict]): serialized SimDataParameter specs
            bounds (np.ndarray)
            n_samples, n_test, n_init_sims, generations (int)
            max_duration (float)
            observables (list[str])
            generation_lower_bound (int)
            seed (int)

    Returns:
        ``CondResult`` describing the outcome.
    """
    import copy as _copy
    import pickle as _pickle
    import time as _time
    import traceback
    from pathlib import Path

    import numpy as np

    j: int = args["j"]
    condition_id: str = args["condition_id"]
    cond_dir = Path(args["cond_dir_str"])
    cond_dir.mkdir(parents=True, exist_ok=True)
    log_path = cond_dir / "_log.txt"

    start = _time.monotonic()
    try:
        # Lazy imports inside the worker — heavy modules, only paid when the
        # worker actually runs (not at parent import time).
        from libuq.pipeline.models import SimDataParameter
        from libuq.pipeline.param_loader import ParameterDataset
        from uq.cli import _sample_local
        from uq.vecoli_config import _apply_design_to_baseline

        # 1. Load baseline, deep-copy, apply design's apply_variant, pickle.
        ds = ParameterDataset(sim_data_path=args["sim_data_path"])
        baseline = ds.sim_data
        designed = _apply_design_to_baseline(
            _copy.deepcopy(baseline),
            args["design_module_name"],
            args["design_params"],
        )
        designed_pickle = cond_dir / "sim_data.cPickle"
        with designed_pickle.open("wb") as f:
            _pickle.dump(designed, f)

        # 2. Rebuild param_space pointing at the pre-baked designed pickle so
        #    parameter_bounds + parameter_names match what's about to be
        #    perturbed.
        param_specs = [SimDataParameter.from_dict(s) for s in args["param_specs"]]
        ds_designed = ParameterDataset(sim_data_path=str(designed_pickle))
        param_space = ds_designed.to_parameter_space(parameters=param_specs)

        # 3. Run the standard local UQ workflow against the pre-baked pickle.
        #    quiet=True suppresses Rich UI; output is captured to log_path.
        #    experiment_id=condition_id gives Nextflow a per-condition scratch
        #    directory so concurrent workers don't collide on nextflow_temp/.
        X_train = args["X_train"]
        X_test = args["X_test"]
        X_all = (
            np.vstack([X_train, X_test]) if X_test is not None else X_train
        )

        _sample_local(
            sim_data_path=str(designed_pickle),
            cache_path=cond_dir,
            X_all=X_all,
            param_space=param_space,
            n_samples=args["n_samples"],
            n_test=args["n_test"],
            n_init_sims=args["n_init_sims"],
            generations=args["generations"],
            max_duration=args["max_duration"],
            observables=args["observables"],
            generation_lower_bound=args["generation_lower_bound"],
            base_config=None,
            conditions=[],
            X_train=X_train,
            X_test=X_test,
            germ_train=args["germ_train"],
            germ_test=args["germ_test"],
            bounds=args["bounds"],
            seed=args["seed"],
            experiment_id=condition_id,
            quiet=True,
            log_path=str(log_path),
        )

        # 4. Count cached samples.
        x_npy = cond_dir / "X.npy"
        n_cached = int(np.load(x_npy).shape[0]) if x_npy.exists() else args["n_samples"]

        # 5. Mark as done so a future --resume can skip this condition cleanly.
        (cond_dir / "_DONE").write_text(condition_id + "\n")

        return CondResult(
            j=j,
            condition_id=condition_id,
            status="ok",
            n_samples_cached=n_cached,
            wall_clock_seconds=_time.monotonic() - start,
            log_tail=_read_log_tail(log_path),
            error=None,
        )

    except Exception:
        tb = traceback.format_exc()
        # Persist the failure marker for post-mortem and to keep failed
        # conditions visible during --resume.
        (cond_dir / "_FAILED").write_text(tb)
        return CondResult(
            j=j,
            condition_id=condition_id,
            status="failed",
            n_samples_cached=0,
            wall_clock_seconds=_time.monotonic() - start,
            log_tail=_read_log_tail(log_path),
            error=tb,
        )
