"""§5 Layer C — emcee MCMC engine: the exact-posterior cross-check.

For the *deterministic* surrogate prototype, the likelihood is writable (the
Kennedy–O'Hagan Gaussian in :mod:`inference.observation`), so an ensemble MCMC
gives an (asymptotically) **exact** posterior. That makes it the reference the
amortized NPE must agree with — if NPE and MCMC posteriors match on the same
observation, the neural engine is trustworthy; where they diverge, NPE is
under-trained. This is a correctness gate, not a competitor: once the data is a
genuinely *stochastic* timeseries (intractable likelihood), MCMC drops out and
NPE carries the inference.

Same observation model as NPE (η + δ + ε), so the two engines target the
identical posterior. No ``libuq``/``uq`` imports.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class MCMCInferenceEngine:
    """Affine-invariant ensemble MCMC over physical θ with a K–O'H likelihood.

    Args:
        forward_map: a :class:`~inference.observation.TrajectoryForwardMap`.
        obs_model: a :class:`~inference.observation.ObservationModel`.
        bounds: ``(d, 2)`` physical bounds (uniform prior support).
    """

    forward_map: object
    obs_model: object
    bounds: np.ndarray

    def _log_prob(self, theta: np.ndarray, data: np.ndarray) -> float:
        lb, ub = self.bounds[:, 0], self.bounds[:, 1]
        if np.any(theta < lb) or np.any(theta > ub):
            return -np.inf
        pred = self.forward_map(theta.reshape(1, -1))
        return self.obs_model.log_likelihood(pred, data)

    def sample_posterior(self, data: np.ndarray, n: int = 4000,
                         n_walkers: int = 32, burn: int = 400,
                         seed: int = 0) -> np.ndarray:
        """Return ``~n`` posterior samples of θ given observed features ``data``."""
        import emcee

        lb, ub = self.bounds[:, 0], self.bounds[:, 1]
        d = len(lb)
        rng = np.random.default_rng(seed)
        p0 = rng.uniform(lb, ub, size=(n_walkers, d))

        sampler = emcee.EnsembleSampler(
            n_walkers, d, self._log_prob, args=(np.ravel(data),))
        steps = burn + int(np.ceil(n / n_walkers))
        sampler.run_mcmc(p0, steps, progress=False)
        chain = sampler.get_chain(discard=burn, flat=True)
        return chain
