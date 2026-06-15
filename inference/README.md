# `inference/` — the §5 UQ-backed inference pipeline

This package turns the UQ surrogate work (PCE / ParametricDMD) into a genuine
**statistical inference engine** for the *E. coli* whole-cell model (vEcoli /
v2ecoli): given observed data, it returns a **validated posterior over `simData`
parameters**. It implements Layers B and C of the plan in `.todo/5.md`.

It is **fully decoupled** from `libuq`/`uq`: every module here reads only the
numpy artifacts a `uq sample` cache produces (`X.npy`, `timeseries/*.npy`,
`metadata.json`) and never imports or mutates the existing pipeline code.

## The stack

```
cached (X, Y(t))  ←  `uq sample`   (the only step that touches the WCM)
    │
    ├─ Layer A — PCE                 →  scalar Sobol attribution        [in libuq, upstream]
    ├─ Layer B — ParametricDMD       →  trajectory Y(θ; X) surrogate    [parametric_dmd.py]
    └─ Layer C — NPE (+ K–O'H δ)     →  posterior over θ given data     [sbi_engine.py]
                 / emcee cross-check                                    [mcmc_engine.py]
```

Layer B is the forward map Layer C calls: the data being matched is a *trajectory*,
so the surrogate that produces trajectories (ParametricDMD), not scalar PCE, is the
right forward model for a stochastic timeseries generator.

## Modules

| file | role |
|---|---|
| `_cache.py` | Read-only loader; phase-aligns variable-length trajectories onto a canonical θ∈[0,1] grid (duration is parameter-dependent and kept separately). |
| `parametric_dmd.py` | `ParametricTrajectorySurrogate` — PyDMD `ParametricDMD` + ezyrb POD/RBF over **eigenvalue-constrained `BOPDMD`** (the B0 divergence fix). Predicts `Y(θ; X)` at unseen X. |
| `b1_generalization.py` | **Step B1 gate** — leave-one-out held-out parametric generalization vs naive baselines + LOO-conformal coverage. |
| `observation.py` | The Kennedy–O'Hagan observation model `data = η(θ) + δ + ε`: surrogate forward map + **model discrepancy** + noise; σ_surrogate sourced from the B1 conformal residuals. |
| `sbi_engine.py` | `NPEInferenceEngine` — amortized Neural Posterior Estimation (the ML inference method). Discrepancy-aware from the first training pass. |
| `mcmc_engine.py` | `MCMCInferenceEngine` — emcee exact-posterior cross-check (correctness gate for NPE while the likelihood is still tractable). |
| `c_inference_prototype.py` | **End-to-end Layer C runner** — trains NPE on the surrogate and validates by SBC/coverage, real-trajectory recovery, a discrepancy ablation, and an NPE↔MCMC cross-check. |

## Why NPE (the "real ML inference" choice)

The WCM is a **stochastic** simulator: its likelihood is implicitly marginalized
over latent seed randomness and is intractable. NPE is built for exactly this —
it learns `q(θ|x)` from simulated `(θ, x)` pairs without ever needing the
likelihood, is **amortized** (train once, infer on any new data instantly), and
consumes the trajectory directly rather than collapsing it to a scalar. The
surrogate makes the millions of forward evaluations affordable. For the current
*deterministic-surrogate* prototype the likelihood is still writable, so emcee
MCMC serves as the exact reference NPE is checked against; once genuinely
stochastic data arrives, MCMC drops out and NPE carries the inference.

Model discrepancy δ (Kennedy & O'Hagan 2001) is folded in **from the start** so
the engine supports inference *about the cell*, not merely *under the model* —
without δ the parameters distort to explain the simulator's own structural error.

## Results (on `.cache/uq_cache`, 20 single-gen `mass` samples)

**B1 — ParametricDMD generalization: PASS.** Rank-3, leave-one-out:
- state channels (dry_mass/cell_mass/volume): **~99.5 % held-out variance**, ~3.6× better than nearest-neighbor or global-mean baselines;
- `growth` (instantaneous-derivative channel) lacks parametric structure — no method beats the flat mean (B0-consistent);
- LOO-conformal coverage 0.90 realized vs 0.90 nominal.
- Artifact: `b1_uq_cache_mass.json`.

**Layer C — inference: validated.** Amortized NPE with K–O'H discrepancy:
- **Calibrated** — synthetic-truth coverage matches nominal across 50/80/90/95 % (worst gap ≈ 0.02); real-trajectory recovery likewise.
- **Discrepancy matters** — on δ-contaminated data (averaged over 8 structural-bias realizations), the identified parameter stays calibrated when δ is modeled (90 % interval → 0.86 coverage) but collapses to overconfidence when ignored (90 % interval → 0.72 coverage).
- **Honest identifiability** — `cell_dry_mass_fraction` is strongly informed by mass trajectories (posterior ~16× tighter than prior); the RNAP/elongation/metabolism parameters are not identifiable from `mass` alone and the posterior correctly returns ≈ prior.
- **Engine correctness** — amortized NPE agrees with exact emcee MCMC to within ≤0.27 posterior σ on every parameter.
- Artifact: `c_inference_uq_cache_mass.json`.

## Run

```bash
# B1 generalization gate (no sims)
uv run python inference/b1_generalization.py --cache .cache/uq_cache

# Layer C end-to-end inference + validation (no sims)
uv run python inference/c_inference_prototype.py --cache .cache/uq_cache \
    --n-train 8000 --n-sbc 150 --json-out inference/c_inference_uq_cache_mass.json
```

## Scope / honesty

- This is validated on **single-generation** trajectories with **no seed axis**;
  it proves the inference *machinery*. The division-reset (multi-gen) and
  intrinsic-stochastic (multi-seed) paths require a richer cache (new sims) and
  are the next step — re-point `--cache` and the same code runs.
- The claim earned here is **inference of `simData` parameters**: a validated,
  discrepancy-aware posterior. The stronger claim of inferring the *true
  biological* parameters additionally requires real cd1 data fitted with the δ
  term (the machinery is already in place).
