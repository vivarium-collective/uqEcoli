"""§5 Layer C — the ML inference engine: amortized Neural Posterior Estimation.

This is the "real ML inference" piece. NPE trains a neural conditional density
estimator ``q(θ | x)`` on simulated ``(θ, x)`` pairs, so that after a single
training pass it returns the **posterior over simData parameters for any
observed data instantly** (amortized, likelihood-free). It is the principled
choice for the WCM because the simulator is a *stochastic* generator whose
likelihood is intractable — NPE never needs the likelihood, only the ability to
simulate, which the Layer-B surrogate makes cheap.

Why this beats the alternatives *here* (honest, per §5 §2a/§4):
  - vs. **raw-WCM MCMC**: a posterior needs ~10⁵–10⁶ forward evaluations;
    against the minutes-per-run WCM that is infeasible. NPE trains on cheap
    *surrogate* simulations instead.
  - vs. **PCE-MCMC on scalars**: that discards the trajectory (the phenotype) and
    assumes a clean Gaussian likelihood the stochastic simulator does not have.
    NPE consumes the trajectory features directly and is amortized.
  - vs. **ABC**: NPE is vastly more sample-efficient and gives a parametric
    posterior, not an acceptance-thresholded cloud.

The generative model simulated here is the full Kennedy–O'Hagan observation
model from :mod:`inference.observation` (surrogate η + discrepancy δ + noise ε),
so the trained posterior already accounts for model discrepancy — the engine is
discrepancy-aware from the first training pass, not as a bolt-on.

Validation is SBI-native: simulation-based calibration / coverage in
:mod:`inference.c_inference_prototype`. Nothing here imports ``libuq``/``uq``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class NPEInferenceEngine:
    """Amortized NPE over physical parameters with a K–O'H observation model.

    Args:
        forward_map: a :class:`~inference.observation.TrajectoryForwardMap`.
        obs_model: a :class:`~inference.observation.ObservationModel`.
        bounds: ``(d, 2)`` physical parameter bounds (the uniform prior support).
        n_train: number of simulated training pairs.
        seed: torch/numpy seed for reproducibility.
        density_estimator: sbi density-estimator key ("nsf", "maf", ...).
    """

    forward_map: object
    obs_model: object
    bounds: np.ndarray
    n_train: int = 4000
    seed: int = 0
    density_estimator: str = "nsf"

    _posterior: object = field(default=None, repr=False)
    _prior: object = field(default=None, repr=False)

    def _torch(self):
        import torch
        return torch

    def _build_prior(self):
        import torch
        from sbi.utils import BoxUniform
        lb = torch.as_tensor(self.bounds[:, 0], dtype=torch.float32)
        ub = torch.as_tensor(self.bounds[:, 1], dtype=torch.float32)
        return BoxUniform(low=lb, high=ub)

    def simulate(self, theta: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        """θ → noisy observation features via η(θ) + δ + ε (the K–O'H model)."""
        mean = self.forward_map(theta)            # (n, n_feat), NaN if diverged
        return self.obs_model.sample(mean, rng)

    def train(self) -> NPEInferenceEngine:
        import torch
        from sbi.inference import NPE

        torch.manual_seed(self.seed)
        rng = np.random.default_rng(self.seed)
        self._prior = self._build_prior()

        # Draw θ from the prior, simulate observations, drop diverged rows.
        lb, ub = self.bounds[:, 0], self.bounds[:, 1]
        theta = rng.uniform(lb, ub, size=(self.n_train, len(lb)))
        x = self.simulate(theta, rng)
        ok = np.isfinite(x).all(axis=1)
        theta, x = theta[ok], x[ok]
        if ok.sum() < 0.5 * self.n_train:
            raise RuntimeError(
                f"surrogate diverged on {(~ok).sum()}/{self.n_train} prior draws; "
                "tighten real_eig_limit or restrict bounds")

        theta_t = torch.as_tensor(theta, dtype=torch.float32)
        x_t = torch.as_tensor(x, dtype=torch.float32)
        inference = NPE(prior=self._prior, density_estimator=self.density_estimator)
        inference.append_simulations(theta_t, x_t)
        # Silence sbi's per-epoch stdout chatter; keep the API call itself intact.
        import contextlib
        import io
        with contextlib.redirect_stdout(io.StringIO()):
            inference.train()
            self._posterior = inference.build_posterior()
        self._n_kept = int(ok.sum())
        return self

    def sample_posterior(self, x_obs: np.ndarray, n: int = 4000) -> np.ndarray:
        """Draw ``n`` posterior samples of θ given observed features ``x_obs``."""
        import torch
        if self._posterior is None:
            raise RuntimeError("sample_posterior() before train()")
        x = torch.as_tensor(np.ravel(x_obs), dtype=torch.float32)
        with torch.no_grad():
            s = self._posterior.sample((n,), x=x, show_progress_bars=False)
        return s.cpu().numpy()
