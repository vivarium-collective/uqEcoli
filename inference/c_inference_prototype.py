"""§5 Layer C — end-to-end inference prototype, validated on synthetic truth.

Closes the loop the whole §5 stack was built for: **data → posterior over
simData parameters**, using the Layer-B ParametricDMD surrogate as the forward
map and amortized NPE (with Kennedy–O'Hagan discrepancy) as the inference
engine. No new simulation — it runs entirely off an existing ``uq sample``
cache, so the inference *machinery* is proven before any stochastic/multi-gen
data exists. Re-point ``--cache`` at a richer cache later and the same code runs.

Pipeline:
  1. Fit the ParametricDMD surrogate on the whole cache → the forward map η(θ).
  2. LOO pass → per-feature surrogate error σ_surrogate (the B1 UQ result reused
     as the parameterization of this phase) → the ObservationModel.
  3. Train NPE once (amortized) on η(θ)+δ+ε draws from the prior.
  4. **Validation A — calibration (SBC/coverage):** draw many synthetic truths
     θ*, generate their data, infer, check that q% credible intervals contain θ*
     ~q% of the time. This is what makes the claim "real inference" rather than
     "confident point estimates."
  5. **Validation B — recovery on real WCM trajectories:** infer θ for each of
     the cache's actual simulated trajectories and compare to the known X.
  6. **Discrepancy ablation:** show that on δ-contaminated data, modeling δ keeps
     coverage calibrated while ignoring it makes the posterior overconfident —
     the concrete payoff of folding K–O'H in from the start.
  7. **MCMC cross-check:** on one observation, confirm the amortized NPE
     posterior matches the exact emcee posterior (engine correctness).

Usage:
    uv run python inference/c_inference_prototype.py \\
        --cache .cache/uq_cache --n-train 4000 --n-sbc 200
"""

from __future__ import annotations

import argparse
import json

import numpy as np

from inference._cache import load_trajectory_cache
from inference.mcmc_engine import MCMCInferenceEngine
from inference.observation import ObservationModel, TrajectoryForwardMap
from inference.parametric_dmd import ParametricTrajectorySurrogate
from inference.sbi_engine import NPEInferenceEngine

NOMINAL_LEVELS = [0.5, 0.8, 0.9, 0.95]


def _fit_surrogate(Xu, Y, pod_rank, dmd_rank):
    return ParametricTrajectorySurrogate(
        pod_rank=pod_rank, dmd_rank=dmd_rank, real_eig_limit=10.0,
        interpolator="rbf").fit(Xu, Y)


def _surrogate_error(cache, fmap_template, pod_rank, dmd_rank) -> np.ndarray:
    """Per-feature relative surrogate error from a LOO pass (the B1 UQ result)."""
    n = cache.n_samples
    Xu = cache.x_unit()
    true_feats, pred_feats = [], []
    for i in range(n):
        tr = [j for j in range(n) if j != i]
        try:
            surr = _fit_surrogate(Xu[tr], cache.Y[tr], pod_rank, dmd_rank)
            pred = surr.predict(Xu[i:i + 1])          # (1, P, F)
            pf = fmap_template.featurize(pred)[0]
            if not np.isfinite(pf).all():
                continue
            pred_feats.append(pf)
            true_feats.append(fmap_template.featurize(cache.Y[i])[0])
        except Exception:
            continue
    true_feats = np.array(true_feats)
    pred_feats = np.array(pred_feats)
    scale = np.where(true_feats.std(0) < 1e-12, 1.0, true_feats.std(0))
    resid_std = (pred_feats - true_feats).std(0)
    return resid_std / scale                          # relative per-feature


def _coverage(samples: np.ndarray, truth: np.ndarray, levels) -> dict:
    """Per-level: fraction of params whose central interval contains the truth."""
    out = {}
    for q in levels:
        lo = np.quantile(samples, (1 - q) / 2, axis=0)
        hi = np.quantile(samples, 1 - (1 - q) / 2, axis=0)
        out[q] = (truth >= lo) & (truth <= hi)        # per-param bool
    return out


def _aggregate_coverage(per_truth: list[dict], levels, idx=None) -> dict:
    def sub(d, q):
        v = d[q]
        return float(np.mean(v[idx])) if idx else float(np.mean(v))
    return {q: float(np.mean([sub(d, q) for d in per_truth])) for q in levels}


def run(cache_dir: str, pod_rank: int, dmd_rank: int, n_phase: int,
        n_train: int, n_sbc: int, n_post: int, seed: int) -> dict:
    cache = load_trajectory_cache(cache_dir, n_phase=n_phase)
    bounds, names = cache.bounds, cache.param_names
    Xu = cache.x_unit()

    # 1. forward map = surrogate fit on the whole cache.
    surrogate = _fit_surrogate(Xu, cache.Y, pod_rank, dmd_rank)
    fmap = TrajectoryForwardMap(surrogate, bounds, cache.obs_names)

    # 2. surrogate error from LOO (B1 UQ result → parameterization).
    sur_sd = _surrogate_error(cache, fmap, pod_rank, dmd_rank)
    train_feats = fmap.featurize(cache.Y)
    feature_scale = np.where(train_feats.std(0) < 1e-12, 1.0, train_feats.std(0))
    obs = ObservationModel(feature_scale=feature_scale, surrogate_sd=sur_sd)

    # 3. train NPE once (amortized, discrepancy-aware).
    engine = NPEInferenceEngine(fmap, obs, bounds, n_train=n_train, seed=seed).train()

    rng = np.random.default_rng(seed + 1)
    lb, ub = bounds[:, 0], bounds[:, 1]
    prior_std = (ub - lb) / np.sqrt(12.0)

    # 4. Validation A — SBC / coverage on synthetic truths.
    cov_A, contraction = [], []
    for _ in range(n_sbc):
        theta_star = rng.uniform(lb, ub)
        data = obs.sample(fmap(theta_star.reshape(1, -1)), rng)
        if not np.isfinite(data).all():
            continue
        post = engine.sample_posterior(data, n=n_post)
        cov_A.append(_coverage(post, theta_star, NOMINAL_LEVELS))
        contraction.append(post.std(0) / prior_std)
    coverage_A = _aggregate_coverage(cov_A, NOMINAL_LEVELS)
    contraction = np.mean(contraction, axis=0)         # per-param posterior/prior std

    # 5. Validation B — recovery on the real cached WCM trajectories.
    cov_B, err_B = [], []
    for i in range(cache.n_samples):
        data = fmap.featurize(cache.Y[i])              # real trajectory, real discrepancy
        post = engine.sample_posterior(data, n=n_post)
        cov_B.append(_coverage(post, cache.X[i], NOMINAL_LEVELS))
        err_B.append(np.abs(post.mean(0) - cache.X[i]) / (ub - lb))  # normalized
    coverage_B = _aggregate_coverage(cov_B, NOMINAL_LEVELS)
    recovery_err = np.mean(err_B, axis=0)              # per-param normalized |bias|

    # 6. Discrepancy ablation — a FIXED structural δ (the K–O'H bias), modeled vs
    # ignored, scored on the parameters the data actually identifies (where a bias
    # can bite; unidentified params stay prior-wide and mask the effect).
    informed = [j for j, c in enumerate(contraction) if c < 0.7] or list(range(len(lb)))
    obs_nodisc = ObservationModel(feature_scale=feature_scale, surrogate_sd=sur_sd, tau=0.0)
    eng_nodisc = NPEInferenceEngine(fmap, obs_nodisc, bounds, n_train=n_train,
                                    seed=seed).train()
    # Average over several structural-bias realizations — a single δ draw is
    # high-variance, so the demonstration must marginalize over which bias the
    # "reality" happens to have.
    n_delta = 8
    cov_with, cov_without = [], []
    for _ in range(n_delta):
        delta_sys = obs.sample_systematic(rng)        # one structural bias, held fixed
        for _ in range(max(n_sbc // n_delta, 5)):
            theta_star = rng.uniform(lb, ub)
            mean = fmap(theta_star.reshape(1, -1))
            if not np.isfinite(mean).all():
                continue
            data = obs.sample_noise_only(mean + delta_sys, rng)   # structural δ + noise
            cov_with.append(_coverage(engine.sample_posterior(data, n_post),
                                      theta_star, NOMINAL_LEVELS))
            cov_without.append(_coverage(eng_nodisc.sample_posterior(data, n_post),
                                         theta_star, NOMINAL_LEVELS))
    ablation = {
        "informed_params": [names[j] for j in informed],
        "with_discrepancy": _aggregate_coverage(cov_with, NOMINAL_LEVELS, informed),
        "without_discrepancy": _aggregate_coverage(cov_without, NOMINAL_LEVELS, informed),
    }

    # 7. MCMC cross-check on one real observation.
    mcmc = MCMCInferenceEngine(fmap, obs, bounds)
    data0 = fmap.featurize(cache.Y[0])
    npe0 = engine.sample_posterior(data0, n_post)
    mc0 = mcmc.sample_posterior(data0, n=6000, n_walkers=48, burn=800, seed=seed)
    # Agreement is measured in units of the posterior's OWN width: on a wide
    # (weakly-identified) marginal a mean differing by prior-range fractions is
    # just Monte-Carlo noise, not engine disagreement. |Δmean| / posterior σ is
    # the metric that asks "do the engines agree relative to the uncertainty?".
    pooled_sd = 0.5 * (npe0.std(0) + mc0.std(0))
    pooled_sd = np.where(pooled_sd < 1e-12, 1.0, pooled_sd)
    crosscheck = {
        "npe_mean": npe0.mean(0).tolist(),
        "mcmc_mean": mc0.mean(0).tolist(),
        "disagreement_in_posterior_sd": (np.abs(npe0.mean(0) - mc0.mean(0))
                                         / pooled_sd).tolist(),
    }

    return {
        "param_names": names,
        "n_train_kept": engine._n_kept,
        "n_features": fmap.n_features,
        "surrogate_error_rel": sur_sd.tolist(),
        "coverage_synthetic": coverage_A,
        "coverage_real_trajectories": coverage_B,
        "posterior_contraction": dict(zip(names, contraction.round(3).tolist())),
        "recovery_abs_bias_norm": dict(zip(names, recovery_err.round(3).tolist())),
        "discrepancy_ablation": ablation,
        "mcmc_crosscheck": crosscheck,
        "_levels": NOMINAL_LEVELS,
        "_config": {"pod_rank": pod_rank, "dmd_rank": dmd_rank, "n_phase": n_phase,
                    "n_train": n_train, "n_sbc": n_sbc, "seed": seed},
    }


def print_report(r: dict) -> None:
    L = r["_levels"]
    print(f"\n§5 Layer C — inference prototype  ({r['n_train_kept']} training sims, "
          f"{r['n_features']} obs features)\n")

    def cov_row(name, d):
        cells = " ".join(f"{q:.2f}:{d[q]:>5.2f}" for q in L)
        print(f"  {name:<26} {cells}")

    print("CALIBRATION — realized coverage vs nominal level (want realized ≈ nominal):")
    cov_row("synthetic truths", r["coverage_synthetic"])
    cov_row("real WCM trajectories", r["coverage_real_trajectories"])

    print("\nDISCREPANCY ABLATION (δ-contaminated data):")
    cov_row("  δ modeled (K–O'H)", r["discrepancy_ablation"]["with_discrepancy"])
    cov_row("  δ ignored", r["discrepancy_ablation"]["without_discrepancy"])

    print("\nWHAT THE DATA CONSTRAINS (posterior std / prior std; <1 = informed):")
    for k, v in r["posterior_contraction"].items():
        bar = "informed" if v < 0.7 else ("weak" if v < 0.95 else "≈prior (unidentified)")
        print(f"  {k:<32} {v:>5.2f}  {bar}")

    print("\nNPE vs MCMC cross-check (|Δposterior-mean| in posterior σ, want <~0.3):")
    for k, v in zip(r["param_names"], r["mcmc_crosscheck"]["disagreement_in_posterior_sd"]):
        print(f"  {k:<32} {v:>5.2f}σ")

    # Verdict: coverage calibrated + NPE≈MCMC ⇒ the engine is real inference.
    cov = r["coverage_synthetic"]
    miscal = max(abs(cov[q] - q) for q in L)
    disagree = max(r["mcmc_crosscheck"]["disagreement_in_posterior_sd"])
    print("\n" + "=" * 66)
    ok_cal = miscal <= 0.12
    ok_xc = disagree <= 0.30
    if ok_cal and ok_xc:
        verdict = ("PASS — calibrated posterior + NPE matches exact MCMC. This is "
                   "validated UQ-backed inference of simData parameters.")
    elif ok_cal:
        verdict = ("MARGINAL — calibrated, but NPE/MCMC disagree; raise --n-train.")
    else:
        verdict = (f"MISCALIBRATED — worst coverage gap {miscal:.2f}; "
                   "raise --n-train or revisit the observation model.")
    print(f"VERDICT: {verdict}")
    print(f"  worst calibration gap={miscal:.2f}, worst NPE/MCMC disagreement={disagree:.2f}σ")
    print("=" * 66)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cache", default=".cache/uq_cache")
    ap.add_argument("--pod-rank", type=int, default=3)
    ap.add_argument("--dmd-rank", type=int, default=3)
    ap.add_argument("--n-phase", type=int, default=256)
    ap.add_argument("--n-train", type=int, default=4000)
    ap.add_argument("--n-sbc", type=int, default=150)
    ap.add_argument("--n-post", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--json-out", default=None)
    args = ap.parse_args()

    r = run(args.cache, args.pod_rank, args.dmd_rank, args.n_phase,
            args.n_train, args.n_sbc, args.n_post, args.seed)
    print_report(r)
    if args.json_out:
        with open(args.json_out, "w") as f:
            json.dump(r, f, indent=2)
        print(f"\nwrote {args.json_out}")


if __name__ == "__main__":
    main()
