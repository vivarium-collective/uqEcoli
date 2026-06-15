"""§5 Layer-B / Step B1 — the ParametricDMD held-out generalization gate.

B0 answered "can DMD rebuild a trajectory it was *fit on*?" (in-sample capacity).
B1 answers the question that actually decides whether Layer B is a *surrogate*:

    Can ParametricDMD predict the full trajectory ``Y(θ; X)`` at parameter
    settings it has **never seen** — and does it beat the naive parametric
    baselines (predicting the training mean, or the nearest training neighbor)?

Protocol (leave-one-out over the cache's ``n`` samples, **no new simulation**):
  1. Phase-align trajectories onto a common ``θ ∈ [0, 1]`` grid (durations vary
     because doubling time is parameter-dependent — see :mod:`inference._cache`).
  2. For each sample ``i``: fit the surrogate on the other ``n−1`` samples,
     predict ``i`` at its own ``X_i``, score against the truth.
  3. Score with the B0-consistent variance-normalized error
     ``e = ||Y − Ŷ|| / ||Y − mean_θ(Y)||`` per observable (``e = 1`` ties the
     trajectory's own flat-mean; ``e → 0`` perfect; ``e² ≈ 1 − R²``).

Baselines ParametricDMD must beat to justify itself:
  - **global_mean** — predict the training-set mean trajectory (uses *no* ``X``).
    Beating this proves the surrogate uses the parameters at all.
  - **nearest_X** — predict the training trajectory whose ``X`` is closest
    (0th-order parametric interpolation). Beating this justifies the DMD +
    interpolation machinery over a lookup table.

Honest UQ: LOO-conformal coverage. Because each prediction is genuinely
out-of-sample, the empirical (1−α) quantile of the LOO residuals defines a
distribution-free band; we report realized vs nominal coverage as a calibration
check (the temporal analog of the Layer-A conformal wrap).

Usage:
    uv run python inference/b1_generalization.py \\
        --cache .cache/uq_cache --pod-rank 2 --dmd-rank 2 --n-phase 256
"""

from __future__ import annotations

import argparse
import json

import numpy as np

from inference._cache import load_trajectory_cache
from inference.parametric_dmd import ParametricTrajectorySurrogate

# Decision-rule thresholds (documented constants).
GOOD_GEN_E = 0.50      # held-out e < 0.50  ⇒  >75% of trajectory variance predicted
BEAT_MARGIN = 0.90     # must reach ≤ 90% of the nearest-neighbor error to "win"
NOMINAL_COVERAGE = 0.90


def _temporal_rel_err(Y: np.ndarray, Yhat: np.ndarray) -> np.ndarray:
    """Per-column ``||Y−Ŷ|| / ||Y−mean_θ(Y)||`` (B0-consistent, 1.0 == flat-mean)."""
    num = np.linalg.norm(Y - Yhat, axis=0)
    den = np.linalg.norm(Y - Y.mean(axis=0), axis=0)
    den = np.where(den < 1e-12, 1.0, den)
    return num / den


def run_gate(cache_dir: str, pod_rank: int, dmd_rank: int, n_phase: int,
             real_eig_limit: float | None, num_trials: int, interpolator: str) -> dict:
    cache = load_trajectory_cache(cache_dir, n_phase=n_phase)
    n, P, F = cache.Y.shape
    Xu = cache.x_unit()

    dmd_pred = np.full_like(cache.Y, np.nan)
    dmd_err, gmean_err, nn_err = [], [], []

    for i in range(n):
        train = [j for j in range(n) if j != i]
        Ytrain = cache.Y[train]
        truth = cache.Y[i]

        # global-mean baseline (no X).
        gmean = Ytrain.mean(axis=0)
        gmean_err.append(_temporal_rel_err(truth, gmean))

        # nearest-X-neighbor baseline (0th-order parametric interpolation).
        d = np.linalg.norm(Xu[train] - Xu[i], axis=1)
        nn_err.append(_temporal_rel_err(truth, Ytrain[int(np.argmin(d))]))

        # ParametricDMD surrogate.
        surr = ParametricTrajectorySurrogate(
            pod_rank=pod_rank, dmd_rank=dmd_rank,
            real_eig_limit=real_eig_limit, num_trials=num_trials,
            interpolator=interpolator,
        )
        try:
            surr.fit(Xu[train], Ytrain)
            pred = surr.predict(Xu[i:i + 1])[0]
            dmd_pred[i] = pred
            e = _temporal_rel_err(truth, pred)
            if not np.isfinite(pred).all() or np.nanmax(e) > 5.0:
                e = np.full(F, np.nan)
            dmd_err.append(e)
        except Exception:
            dmd_err.append(np.full(F, np.nan))

    dmd_err = np.array(dmd_err)
    gmean_err = np.array(gmean_err)
    nn_err = np.array(nn_err)

    # LOO-conformal coverage on z-scored residuals (out-of-sample by construction).
    sd = cache.Y.reshape(-1, F).std(axis=0)
    sd = np.where(sd < 1e-12, 1.0, sd)
    resid = np.abs((cache.Y - dmd_pred) / sd)            # (n, P, F)
    finite = np.isfinite(resid).all(axis=(1, 2))
    coverage = None
    if finite.sum() >= 3:
        r = resid[finite]
        band = np.quantile(r, NOMINAL_COVERAGE, axis=0)  # (P, F) pointwise band
        coverage = float((r <= band).mean())

    return {
        "global_mean": {"per_col": np.nanmean(gmean_err, 0).tolist(),
                        "overall": float(np.nanmean(gmean_err))},
        "nearest_X": {"per_col": np.nanmean(nn_err, 0).tolist(),
                      "overall": float(np.nanmean(nn_err))},
        "parametric_dmd": {"per_col": np.nanmean(dmd_err, 0).tolist(),
                           "overall": float(np.nanmean(dmd_err)),
                           "n_diverged": int((~finite).sum())},
        "_coverage": {"nominal": NOMINAL_COVERAGE, "realized": coverage},
        "_cols": cache.obs_names,
        "_n_samples": n, "_n_phase": P, "_n_features": F,
        "_duration_range": [int(cache.durations.min()), int(cache.durations.max())],
        "_config": {"pod_rank": pod_rank, "dmd_rank": dmd_rank,
                    "real_eig_limit": real_eig_limit, "num_trials": num_trials,
                    "interpolator": interpolator},
    }


def print_report(res: dict) -> None:
    cols = res["_cols"]
    print(f"\nB1 ParametricDMD generalization gate — {res['_n_samples']} samples "
          f"(LOO), {res['_n_features']} observables, {res['_n_phase']}-pt phase grid")
    print(f"trajectory durations span {res['_duration_range'][0]}–{res['_duration_range'][1]} steps "
          f"(parameter-dependent doubling time); cfg={res['_config']}\n")

    header = f"{'method':<16} " + " ".join(f"{c.split('__')[-1][:10]:>11}" for c in cols) + f" {'OVERALL':>9}"
    print(header)
    print("-" * len(header))
    for name in ("global_mean", "nearest_X", "parametric_dmd"):
        e = res[name]
        cells = " ".join(f"{v:>11.3f}" for v in e["per_col"])
        print(f"{name:<16} {cells} {e['overall']:>9.3f}")

    dmd = np.array(res["parametric_dmd"]["per_col"])
    nn = np.array(res["nearest_X"]["per_col"])
    gm = np.array(res["global_mean"]["per_col"])
    best_base = np.minimum(nn, gm)            # the best a naive method achieves, per channel
    cov = res["_coverage"]["realized"]
    nd = res["parametric_dmd"]["n_diverged"]

    # Per-channel verdict: DMD must beat the best naive baseline by BEAT_MARGIN
    # AND reach GOOD_GEN_E. Aggregate on the MEDIAN channel so one structureless
    # observable (e.g. the noisy growth derivative — see B0) can't drag the call.
    wins = (dmd <= BEAT_MARGIN * best_base) & (dmd < GOOD_GEN_E)
    med = float(np.nanmedian(dmd))
    short = [c.split("__")[-1] for c in cols]

    print("\n" + "=" * 70)
    print("DECISION (§5 Layer-B B1 — held-out parametric generalization)")
    print(f"  median channel e={med:.3f}  (~{100*(1-med**2):.1f}% trajectory variance)")
    for j, c in enumerate(short):
        mark = "WIN " if wins[j] else "----"
        note = ""
        if not wins[j]:
            # is ANY method good here? if best baseline is also poor, the channel
            # lacks parametric structure (not a DMD failure).
            note = (" (no method beats flat-mean → observable lacks parametric "
                    "structure)" if best_base[j] > GOOD_GEN_E else " (DMD underperforms baseline)")
        print(f"    [{mark}] {c:<12} DMD e={dmd[j]:.3f}  best-baseline e={best_base[j]:.3f}{note}")
    if cov is not None:
        print(f"  LOO-conformal coverage: {cov:.2f} realized vs {res['_coverage']['nominal']:.2f} nominal")
    if nd:
        print(f"  WARNING: {nd}/{res['_n_samples']} folds diverged (NaN, excluded)")

    n_struct = int((best_base < GOOD_GEN_E).sum())  # channels with parametric structure
    n_win = int(wins.sum())
    if nd > res["_n_samples"] // 2:
        verdict = ("FAIL — majority of folds diverged; tighten real_eig_limit or "
                   "escalate to Hankel/SINDy.")
    elif n_win >= n_struct and n_struct > 0 and med < GOOD_GEN_E:
        verdict = (f"PASS — ParametricDMD generalizes on all {n_struct} structured "
                   "channels and beats naive interpolation. Proceed to "
                   "multi-gen/stochastic cache + Layer C.")
    elif n_win > 0:
        verdict = (f"MARGINAL — generalizes on {n_win}/{n_struct} structured channels; "
                   "raise pod/dmd rank or add samples before Layer C.")
    else:
        verdict = ("FAIL — does not beat naive baselines on any structured channel; "
                   "investigate before proceeding.")
    print(f"  VERDICT: {verdict}")
    print("=" * 70)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cache", default=".cache/uq_cache")
    ap.add_argument("--pod-rank", type=int, default=3,
                    help="spatial POD rank (parametric generalization needs more "
                         "modes than B0's single-trajectory rank-2)")
    ap.add_argument("--dmd-rank", type=int, default=3)
    ap.add_argument("--n-phase", type=int, default=256)
    ap.add_argument("--real-eig-limit", type=float, default=10.0,
                    help="cap on Re(eigenvalue) (B0 divergence fix); set <0 to disable")
    ap.add_argument("--num-trials", type=int, default=0,
                    help="BOPDMD bagging trials (0 = deterministic optimized DMD)")
    ap.add_argument("--interpolator", choices=["rbf", "linear"], default="rbf")
    ap.add_argument("--json-out", default=None)
    args = ap.parse_args()

    real_eig_limit = None if args.real_eig_limit < 0 else args.real_eig_limit
    res = run_gate(args.cache, args.pod_rank, args.dmd_rank, args.n_phase,
                   real_eig_limit, args.num_trials, args.interpolator)
    print_report(res)
    if args.json_out:
        with open(args.json_out, "w") as f:
            json.dump(res, f, indent=2)
        print(f"\nwrote {args.json_out}")


if __name__ == "__main__":
    main()
