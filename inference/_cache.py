"""Shared cache loading for the §5 inference package.

Read-only consumer of a ``uq sample`` cache directory (the layout written by
``uq/sampling.py`` / ``libuq``): ``X.npy`` (parameter design points), per-sample
``timeseries/sample_*.npy`` trajectories, and ``metadata.json`` (parameter
names, bounds, observable columns). **This module never imports or mutates
``libuq``/``uq``** — it only reads files those packages produced, so the
inference layer stays fully decoupled from the existing pipeline.

The one non-trivial transform here is **phase resampling**. WCM trajectories
have parameter-dependent *length* (faster growth ⇒ shorter cell cycle), so the
raw cache holds ``T``-varying matrices (1118–2158 steps in ``.cache/uq_cache``).
``ParametricDMD`` and every downstream comparison need a common time axis, so we
warp each trajectory onto a canonical cell-cycle **phase** grid ``θ ∈ [0, 1]``
with a fixed number of points. The original duration is preserved separately in
``durations`` — it is itself a parameter-dependent scalar observable (a natural
PCE target), not something to silently discard.
"""

from __future__ import annotations

import glob
import json
import os
from dataclasses import dataclass

import numpy as np


@dataclass
class TrajectoryCache:
    """A phase-aligned view of a ``uq sample`` trajectory cache.

    Attributes:
        X: ``(n, d)`` physical parameter design points.
        Y: ``(n, P, F)`` trajectories resampled onto a common ``P``-point phase
           grid (``θ ∈ [0, 1]``), in physical observable units.
        durations: ``(n,)`` original trajectory length ``T`` per sample — the
           parameter-dependent cell-cycle duration, retained for honesty.
        bounds: ``(d, 2)`` ``[lb, ub]`` per parameter.
        param_names: length-``d`` parameter names.
        obs_names: length-``F`` observable column names.
        phase: ``(P,)`` the canonical phase grid (``linspace(0, 1, P)``).
    """

    X: np.ndarray
    Y: np.ndarray
    durations: np.ndarray
    bounds: np.ndarray
    param_names: list[str]
    obs_names: list[str]
    phase: np.ndarray

    @property
    def n_samples(self) -> int:
        return self.X.shape[0]

    @property
    def n_params(self) -> int:
        return self.X.shape[1]

    @property
    def n_features(self) -> int:
        return self.Y.shape[2]

    @property
    def n_phase(self) -> int:
        return self.Y.shape[1]

    def x_unit(self, X: np.ndarray | None = None) -> np.ndarray:
        """Min-max normalize parameters to ``[0, 1]^d`` via ``bounds``.

        Parameters span ~10 orders of magnitude (e.g. ``kinetic_objective_weight``
        ``~5e-8`` vs ``basal_elongation_rate ~20``); the RBF/POD interpolation in
        ``ParametricDMD`` is distance-based, so unit-cube normalization is
        mandatory for it to weight every axis fairly. Test points are normalized
        with the *same* bounds, so this introduces no train/test leakage.
        """
        X = self.X if X is None else np.atleast_2d(X)
        lb, ub = self.bounds[:, 0], self.bounds[:, 1]
        span = np.where((ub - lb) < 1e-300, 1.0, ub - lb)
        return (X - lb) / span


def _resample_to_phase(Y: np.ndarray, n_phase: int) -> np.ndarray:
    """Linearly warp a ``(T, F)`` trajectory onto an ``(n_phase, F)`` phase grid.

    Maps each trajectory's own time axis to normalized phase ``[0, 1]`` and
    interpolates every observable column onto a shared grid. This is the
    cell-cycle-phase alignment that makes trajectories of different durations
    comparable in a single POD/DMD basis.
    """
    T = Y.shape[0]
    src = np.linspace(0.0, 1.0, T)
    dst = np.linspace(0.0, 1.0, n_phase)
    return np.column_stack([np.interp(dst, src, Y[:, f]) for f in range(Y.shape[1])])


def load_trajectory_cache(cache: str, n_phase: int = 256) -> TrajectoryCache:
    """Load a ``uq sample`` cache and phase-align its trajectories.

    Args:
        cache: path to a cache directory containing ``X.npy``, ``metadata.json``,
            and ``timeseries/sample_*.npy``.
        n_phase: number of points on the canonical phase grid (default 256 —
            ample for smooth mass growth; raise for oscillatory observables).
    """
    with open(os.path.join(cache, "metadata.json")) as f:
        meta = json.load(f)
    X = np.load(os.path.join(cache, "X.npy"))
    paths = sorted(glob.glob(os.path.join(cache, "timeseries", "sample_*.npy")))
    if not paths:
        paths = sorted(glob.glob(os.path.join(cache, "timeseries", "*.npy")))
    raw = [np.load(p) for p in paths]

    if X.shape[0] != len(raw):
        raise ValueError(
            f"{cache}: X has {X.shape[0]} rows but found {len(raw)} trajectories"
        )

    durations = np.array([t.shape[0] for t in raw], dtype=float)
    Y = np.stack([_resample_to_phase(t, n_phase) for t in raw])  # (n, P, F)

    obs_names = meta.get("observable_columns") or [
        f"obs_{i}" for i in range(Y.shape[2])
    ]
    param_names = meta.get("parameter_names") or [
        f"x{i}" for i in range(X.shape[1])
    ]
    bounds = np.array(meta["bounds"], dtype=float)

    return TrajectoryCache(
        X=X,
        Y=Y,
        durations=durations,
        bounds=bounds,
        param_names=param_names,
        obs_names=obs_names,
        phase=np.linspace(0.0, 1.0, n_phase),
    )
