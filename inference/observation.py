"""§5 Layer C — the observation model: forward map + Kennedy–O'Hagan discrepancy.

This is the shared core both inference engines (SBI/NPE in :mod:`sbi_engine`,
emcee MCMC in :mod:`mcmc_engine`) build on. It turns the Layer-B trajectory
surrogate into a generative model for *observed data* and encodes, from the
start, the three distinct sources of uncertainty between a parameter setting and
a measurement — keeping the eventual inference honest:

    data(θ)  =  η(θ)              # surrogate forward map  (Layer B)
              + δ                  # model discrepancy  (Kennedy & O'Hagan 2001)
              + ε                  # observation noise

  - **η(θ)** — the ParametricDMD surrogate's predicted trajectory, summarized to
    a fixed feature vector (state channels sampled at ``K`` phase points). This
    is the "UQ result as the parameterization of the next phase" — Layer B *is*
    the forward map the inference loop calls.
  - **σ_surrogate** — the surrogate's own predictive error, taken from the **B1
    LOO-conformal residuals** (``inference/b1_uq_cache_mass.json``). Folding it
    in stops the posterior from pretending the emulator is exact (§4 caveat #2).
  - **δ (discrepancy)** — the systematic, *correlated* simulator-vs-reality
    mismatch. Modeled as a smooth Gaussian-process-style bias across phase with
    magnitude ``τ``. This is the term that separates "inference under the model"
    from "inference about the cell": without it, the parameters distort to
    explain the simulator's own structural errors and the posterior is biased &
    overconfident.
  - **ε** — independent observation noise.

Because δ, ε and surrogate error are all Gaussian, the engines can either
*sample* them (SBI: marginalize by simulation) or *add their covariances*
(MCMC: marginalize analytically) — both are provided here so the two engines
share one definition.

No ``libuq``/``uq`` imports; consumes only the numpy surrogate from
:mod:`inference.parametric_dmd`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def state_channel_indices(obs_names: list[str]) -> list[int]:
    """State (extensive) channels, excluding rate/derivative channels.

    Rate channels (``growth`` etc.) are noisy instantaneous derivatives that B0
    and B1 both showed lack parametric structure — they carry little inferential
    signal and only inflate the feature vector. Default to all channels if none
    look like rates.
    """
    rate = ("growth", "rate", "deriv")
    idx = [i for i, c in enumerate(obs_names) if not any(r in c.lower() for r in rate)]
    return idx or list(range(len(obs_names)))


class TrajectoryForwardMap:
    """θ (physical) → summary feature vector, via a fitted ParametricDMD surrogate.

    The feature vector is the state-channel trajectory sampled at ``K`` evenly
    spaced phase points — a compact, identifiable observation representation
    (full 256-pt trajectories are redundant given the rank-2/3 dynamics).
    """

    def __init__(self, surrogate, bounds: np.ndarray, obs_names: list[str],
                 n_phase_points: int = 8):
        self.surrogate = surrogate
        self.bounds = np.asarray(bounds, dtype=float)
        self.state_idx = state_channel_indices(obs_names)
        self.n_phase_points = n_phase_points
        self._phase_sel: np.ndarray | None = None

    def _to_unit(self, X: np.ndarray) -> np.ndarray:
        lb, ub = self.bounds[:, 0], self.bounds[:, 1]
        span = np.where((ub - lb) < 1e-300, 1.0, ub - lb)
        return (np.atleast_2d(X) - lb) / span

    def featurize(self, Y: np.ndarray) -> np.ndarray:
        """(n, P, F) trajectories → (n, K*len(state_idx)) feature matrix."""
        if Y.ndim == 2:
            Y = Y[None]
        P = Y.shape[1]
        if self._phase_sel is None:
            self._phase_sel = np.linspace(0, P - 1, self.n_phase_points).astype(int)
        sub = Y[:, self._phase_sel][:, :, self.state_idx]   # (n, K, S)
        return sub.reshape(Y.shape[0], -1)

    def __call__(self, X: np.ndarray) -> np.ndarray:
        """Physical θ → feature vector. NaN rows (surrogate divergence) propagate."""
        Y = self.surrogate.predict(self._to_unit(X))        # (n, P, F) physical
        return self.featurize(Y)

    @property
    def n_features(self) -> int:
        return self.n_phase_points * len(self.state_idx)


@dataclass
class ObservationModel:
    """Kennedy–O'Hagan observation model: noise + surrogate error + discrepancy.

    All scales are *relative* to the per-feature data scale (estimated from the
    training feature matrix), so a single set of defaults is dimensionless and
    portable across observable presets.

    Args:
        feature_scale: ``(n_feat,)`` per-feature scale (std of training features).
        sigma_obs: relative observation-noise std (default 2%).
        surrogate_sd: ``(n_feat,)`` relative surrogate-error std from B1 conformal
            residuals; ``None`` falls back to ``sigma_surrogate_default``.
        sigma_surrogate_default: used if ``surrogate_sd`` is unavailable.
        tau: relative magnitude of the systematic discrepancy δ (default 5%).
        corr_len: correlation length of δ across the feature/phase index
            (fraction of the vector length) — makes δ a smooth bias, not noise.
    """

    feature_scale: np.ndarray
    sigma_obs: float = 0.02
    surrogate_sd: np.ndarray | None = None
    sigma_surrogate_default: float = 0.03
    tau: float = 0.10
    corr_len: float = 0.4

    def __post_init__(self):
        s = np.asarray(self.feature_scale, dtype=float)
        s = np.where(s < 1e-12, 1.0, s)
        self.feature_scale = s
        n = s.shape[0]

        # surrogate-error std per feature (absolute), from B1 conformal residuals.
        if self.surrogate_sd is not None:
            self._sur = np.broadcast_to(np.asarray(self.surrogate_sd, float), (n,)) * s
        else:
            self._sur = self.sigma_surrogate_default * s

        # discrepancy covariance: squared-exponential over the feature index.
        i = np.arange(n)
        d2 = (i[:, None] - i[None, :]) ** 2
        ell = max(self.corr_len * n, 1.0)
        K = np.exp(-0.5 * d2 / ell ** 2)
        self._cov_delta = (self.tau ** 2) * np.outer(s, s) * K

        # diagonal noise + surrogate covariance.
        self._cov_diag = (self.sigma_obs * s) ** 2 + self._sur ** 2

        # total covariance and its Cholesky (for sampling) / inverse (for MCMC).
        self._cov_total = self._cov_delta + np.diag(self._cov_diag)
        self._chol = np.linalg.cholesky(self._cov_total +
                                        1e-9 * np.eye(n) * np.trace(self._cov_total) / n)

    # --- SBI path: sample the perturbations (marginalize by simulation) ---
    def sample(self, mean: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        """Add a δ+ε+surrogate draw to mean features ``(n, n_feat)``."""
        mean = np.atleast_2d(mean)
        z = rng.standard_normal(mean.shape)
        return mean + z @ self._chol.T

    def sample_systematic(self, rng: np.random.Generator) -> np.ndarray:
        """One *structural* discrepancy realization δ ``(n_feat,)`` (from Σ_δ only).

        Unlike :meth:`sample`, this is drawn once and held fixed across a dataset
        — the defining property of Kennedy–O'Hagan discrepancy (a systematic
        simulator-vs-reality bias, not per-observation noise).
        """
        n = self.feature_scale.shape[0]
        chol_d = np.linalg.cholesky(
            self._cov_delta + 1e-9 * np.eye(n) * (np.trace(self._cov_delta) / n + 1e-12))
        return chol_d @ rng.standard_normal(n)

    def sample_noise_only(self, mean: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        """Add only ε+surrogate (diagonal) noise — no discrepancy."""
        mean = np.atleast_2d(mean)
        return mean + rng.standard_normal(mean.shape) * np.sqrt(self._cov_diag)

    # --- MCMC path: analytic Gaussian log-likelihood (marginalize δ,ε) ---
    def log_likelihood(self, pred: np.ndarray, data: np.ndarray) -> float:
        """Gaussian log-likelihood ``log N(data; pred, Σ_total)``."""
        if not np.isfinite(pred).all():
            return -np.inf
        r = np.ravel(data) - np.ravel(pred)
        alpha = np.linalg.solve(self._cov_total, r)
        logdet = 2.0 * np.log(np.diag(self._chol)).sum()
        return float(-0.5 * (r @ alpha + logdet + r.size * np.log(2 * np.pi)))

    @property
    def cov_total(self) -> np.ndarray:
        return self._cov_total
