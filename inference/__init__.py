"""§5 UQ-backed inference for the E. coli whole-cell model.

A self-contained layer on top of a ``uq sample`` cache that turns the UQ
surrogate into a validated statistical-inference engine: ``data → posterior over
simData parameters``. Decoupled from ``libuq``/``uq`` (reads cache artifacts
only). See ``inference/README.md`` and ``.todo/5.md`` for the full design.

Layers:
  - **B** trajectory surrogate — :class:`~inference.parametric_dmd.ParametricTrajectorySurrogate`
  - **C** inference — :class:`~inference.sbi_engine.NPEInferenceEngine` (amortized NPE)
    over the Kennedy–O'Hagan :class:`~inference.observation.ObservationModel`, with
    :class:`~inference.mcmc_engine.MCMCInferenceEngine` as the exact cross-check.
"""

from inference._cache import TrajectoryCache, load_trajectory_cache
from inference.observation import ObservationModel, TrajectoryForwardMap
from inference.parametric_dmd import ParametricTrajectorySurrogate

__all__ = [
    "TrajectoryCache",
    "load_trajectory_cache",
    "ParametricTrajectorySurrogate",
    "TrajectoryForwardMap",
    "ObservationModel",
    # Engines are imported lazily (torch/sbi are heavy); import directly:
    #   from inference.sbi_engine import NPEInferenceEngine
    #   from inference.mcmc_engine import MCMCInferenceEngine
]
