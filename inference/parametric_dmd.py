"""§5 Layer B — ParametricDMD trajectory surrogate.

B0 (``examples/pipeline/koopman/b0_reconstruction_gate.py``) proved a rank-2
Koopman/DMD operator reconstructs a *single, already-simulated* trajectory
(in-sample capacity). This module builds the next thing B0 explicitly deferred:
a surrogate that predicts the **full trajectory ``Y(t; X)`` at parameter
settings never simulated**, by interpolating the DMD dynamics across the
parameter axis — the real Layer-B prize.

Engine: PyDMD's :class:`~pydmd.ParametricDMD` (the plan's named tool), with
``ezyrb`` POD for spatial reduction and RBF for parameter-space interpolation,
wrapping an **eigenvalue-constrained** :class:`~pydmd.BOPDMD`. The eigenvalue
constraint (``real_eig_limit``) is the direct fix for the B0 finding that
unconstrained bagging emits eigenvalues with large positive real part that blow
up over the forecast horizon (reconstruction error → 1e6). Capping the real part
keeps every fit forecastable.

Design choices, all to keep the surrogate honest:
  - **Train-only z-scoring.** Per-observable mean/std are computed from the
    training trajectories only and reused on held-out points — no leakage. DMD's
    SVD is scale-sensitive across the heterogeneous mass/volume/growth columns.
  - **Unit-cube parameters.** Interpolation is distance-based; parameters are
    normalized to ``[0, 1]^d`` upstream (:meth:`TrajectoryCache.x_unit`).
  - **Divergence guard.** A prediction that goes non-finite or explodes past a
    sane multiple of the training scale is reported as such rather than silently
    averaged in (mirrors the B0 gate's guard).

Nothing here imports ``libuq``/``uq``; the surrogate consumes only numpy arrays
from :mod:`inference._cache`.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field

import numpy as np


@dataclass
class ParametricTrajectorySurrogate:
    """Predicts ``Y(θ; X)`` for unseen ``X`` by interpolating DMD dynamics.

    Args:
        pod_rank: spatial POD rank (≤ n_features). For the 4-column ``mass``
            preset, B0's rank-2 finding suggests 2–3.
        dmd_rank: SVD rank of the per-reduced-space DMD fit (``svd_rank``).
        real_eig_limit: cap on the real part of BOPDMD eigenvalues — the B0 fix.
            ``None`` disables it; a finite value (e.g. 10.0 in normalized phase)
            keeps forecasts bounded.
        num_trials: BOPDMD bagging trials for eigenvalue UQ. ``0`` = deterministic
            optimized DMD (trustworthy default per B0); ``>0`` enables bagging
            (needs the eigenvalue constraint to avoid the B0 divergence).
        interpolator: ``"rbf"`` (default) or ``"linear"`` — the ezyrb
            parameter-space approximation.
    """

    pod_rank: int = 2
    dmd_rank: int = 2
    real_eig_limit: float | None = 10.0
    num_trials: int = 0
    interpolator: str = "rbf"

    # Fitted state.
    _mu: np.ndarray | None = field(default=None, repr=False)
    _sd: np.ndarray | None = field(default=None, repr=False)
    _pdmd: object = field(default=None, repr=False)
    _n_phase: int = field(default=0, repr=False)
    _n_features: int = field(default=0, repr=False)

    def _make_interpolator(self):
        from ezyrb import RBF, Linear

        return Linear() if self.interpolator == "linear" else RBF()

    def fit(self, X_unit: np.ndarray, Y: np.ndarray) -> ParametricTrajectorySurrogate:
        """Fit on training trajectories.

        Args:
            X_unit: ``(n, d)`` unit-cube parameters.
            Y: ``(n, P, F)`` trajectories in physical units.
        """
        from ezyrb import POD
        from pydmd import BOPDMD, ParametricDMD

        X_unit = np.atleast_2d(X_unit)
        _, P, F = Y.shape
        self._n_phase, self._n_features = P, F

        # Train-only standardization (per observable, across samples & phase).
        flat = Y.reshape(-1, F)
        self._mu = flat.mean(axis=0)
        self._sd = np.where(flat.std(axis=0) < 1e-12, 1.0, flat.std(axis=0))
        Yz = (Y - self._mu) / self._sd  # (n, P, F)

        # ParametricDMD wants (n_params, n_features, n_time).
        training = np.transpose(Yz, (0, 2, 1))  # (n, F, P)
        t = np.linspace(0.0, 1.0, P)

        dmd = BOPDMD(
            svd_rank=min(self.dmd_rank, F),
            num_trials=self.num_trials,
            trial_size=0.6,
            real_eig_limit=self.real_eig_limit,
        )
        self._pdmd = ParametricDMD(
            dmd,
            POD(rank=min(self.pod_rank, F)),
            self._make_interpolator(),
            dmd_fit_kwargs={"t": t},
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # BOPDMD conditioning/convergence chatter
            self._pdmd.fit(training, X_unit)
        return self

    def predict(self, X_unit: np.ndarray) -> np.ndarray:
        """Predict trajectories at unit-cube parameters.

        Args:
            X_unit: ``(m, d)`` unit-cube parameters.

        Returns:
            ``(m, P, F)`` predicted trajectories in physical units. Rows that
            diverge (non-finite or exploding past the training scale) are filled
            with NaN so the failure is visible, not silently averaged.
        """
        if self._pdmd is None:
            raise RuntimeError("predict() before fit()")
        X_unit = np.atleast_2d(X_unit)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self._pdmd.parameters = X_unit
            recon = np.real(self._pdmd.reconstructed_data)  # (m, F, P)

        Yz = np.transpose(recon, (0, 2, 1))  # (m, P, F)
        Y = Yz * self._sd + self._mu

        # Divergence guard: anything > 50x the training z-scale is a blow-up.
        bad = ~np.isfinite(Yz).all(axis=(1, 2)) | (np.abs(Yz).max(axis=(1, 2)) > 50.0)
        Y[bad] = np.nan
        return Y

    def eigenvalues(self) -> np.ndarray | None:
        """Fitted DMD eigenvalues (Koopman rates), if accessible."""
        try:
            return self._pdmd._reference_dmd.eigs
        except Exception:
            return None
