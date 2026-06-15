Inference Tutorial
==================

This page documents **D** — a complete, runnable walkthrough of the
UQ-backed inference functionality in the ``inference/`` package, from a cached
``uq sample`` run to a validated posterior over ``simData`` parameters. It
assumes the concepts from :doc:`inference_methodology`; no statistics background
is required to *run* it.

.. contents::
   :local:
   :depth: 2

Prerequisites
-------------

* A ``uq sample`` cache with **per-sample trajectories** on disk
  (``timeseries/sample_*.npy``). The bundled example is
  ``.cache/uq_cache`` (20 single-generation ``mass`` samples, 6 parameters).
* The inference extras (installed with the project): ``pydmd``, ``ezyrb``
  (Layer B) and ``sbi``, ``emcee`` (Layer C).

To produce your own cache, see :doc:`getting_started`:

.. code-block:: bash

   uv run uq sample /path/to/simData.cPickle \
       --cache-dir ./uq_cache --n-samples 50 --generations 1 \
       --observables mass --live

Step 1 — Validate the trajectory surrogate (Layer B / B1)
---------------------------------------------------------

Before inference, confirm the trajectory surrogate **generalizes to unseen
parameters**. The B1 gate runs leave-one-out: it fits ParametricDMD on
:math:`n-1` samples and predicts the held-out one, comparing against naive
baselines.

.. code-block:: bash

   uv run python inference/b1_generalization.py --cache .cache/uq_cache

Expected output (abridged):

.. code-block:: text

   method              dry_mass   cell_mass      volume      growth   OVERALL
   global_mean            0.337       0.351       0.351       0.545     0.396
   nearest_X              0.248       0.260       0.260       0.633     0.350
   parametric_dmd         0.069       0.070       0.070       0.634     0.211
   ...
     [WIN ] dry_mass     DMD e=0.069  best-baseline e=0.248
     [WIN ] cell_mass    DMD e=0.070  best-baseline e=0.260
     [WIN ] volume       DMD e=0.070  best-baseline e=0.260
     [----] growth       ... (no method beats flat-mean → lacks parametric structure)
   LOO-conformal coverage: 0.90 realized vs 0.90 nominal
   VERDICT: PASS — ParametricDMD generalizes on all 3 structured channels ...

How to read it:

* ``e`` is the variance-normalized error (``e = 1`` ties the flat-mean,
  ``e → 0`` perfect, :math:`e^2 \approx 1 - R^2`). ``e = 0.07`` ≈ 99.5 % variance.
* ParametricDMD must beat both baselines on the channels that *have* parametric
  structure. ``growth`` (a noisy instantaneous derivative) lacks structure for
  *all* methods and is reported, not hidden.
* The LOO-conformal coverage line is the calibrated trajectory-uncertainty check.

Key flags: ``--pod-rank``/``--dmd-rank`` (default 3 — the parametric problem
needs more modes than B0's single-trajectory rank-2), ``--n-phase`` (phase-grid
resolution), ``--real-eig-limit`` (the eigenvalue cap that keeps forecasts
bounded), ``--json-out``.

Step 2 — Run end-to-end inference (Layer C)
-------------------------------------------

The Layer-C runner fits the surrogate, derives its error budget from a LOO pass,
trains amortized NPE once, and validates the resulting posterior four ways.

.. code-block:: bash

   uv run python inference/c_inference_prototype.py \
       --cache .cache/uq_cache --n-train 8000 --n-sbc 150 \
       --json-out inference/c_inference_uq_cache_mass.json

Abridged output:

.. code-block:: text

   CALIBRATION — realized coverage vs nominal level (want realized ≈ nominal):
     synthetic truths           0.50: 0.48 0.80: 0.79 0.90: 0.90 0.95: 0.95
     real WCM trajectories      0.50: 0.47 0.80: 0.85 0.90: 0.91 0.95: 0.96

   DISCREPANCY ABLATION (δ-contaminated data):
       δ modeled (K–O'H)        0.50: 0.49 0.80: 0.76 0.90: 0.86 0.95: 0.92
       δ ignored                0.50: 0.35 0.80: 0.58 0.90: 0.72 0.95: 0.80

   WHAT THE DATA CONSTRAINS (posterior std / prior std; <1 = informed):
     cell_dry_mass_fraction            0.06  informed
     fraction_active_rnap_free         0.90  weak
     ...

   NPE vs MCMC cross-check (|Δposterior-mean| in posterior σ, want <~0.3):
     cell_dry_mass_fraction            0.12σ ...

   VERDICT: PASS — calibrated posterior + NPE matches exact MCMC.

How to read each block:

#. **Calibration.** Realized coverage should track the nominal level on both
   synthetic truths and the real cached trajectories. A 90 % interval that
   contains the truth ≈ 90 % of the time is what makes this *inference*.
#. **Discrepancy ablation.** On data carrying a structural bias, modeling
   :math:`\delta` keeps coverage near nominal; ignoring it under-covers
   (overconfident). This is the payoff of the Kennedy–O'Hagan term.
#. **What the data constrains.** Posterior-std / prior-std per parameter: ``<1``
   means the data informed it. Here ``mass`` trajectories strongly identify
   ``cell_dry_mass_fraction`` and correctly leave the RNAP/elongation parameters
   near their prior.
#. **NPE vs MCMC.** Disagreement of the amortized posterior from the exact MCMC
   posterior, in units of the posterior's own width. Small ⇒ the neural engine is
   trustworthy.

Useful flags: ``--n-train`` (more pairs ⇒ tighter NPE, the principled fix if the
cross-check is marginal), ``--n-sbc`` (number of calibration truths),
``--n-post`` (posterior samples per inference), ``--pod-rank``/``--dmd-rank``,
``--seed``.

Step 3 — Use the API programmatically
-------------------------------------

The same pipeline as a library. Every object is documented in
:doc:`api/inference`.

.. code-block:: python

   import numpy as np
   from inference import (
       load_trajectory_cache, ParametricTrajectorySurrogate,
       TrajectoryForwardMap, ObservationModel,
   )
   from inference.sbi_engine import NPEInferenceEngine

   # 1. Load + phase-align a uq sample cache (read-only).
   cache = load_trajectory_cache(".cache/uq_cache", n_phase=256)
   Xu = cache.x_unit()                       # parameters normalized to [0,1]^d

   # 2. Layer B: fit the trajectory surrogate = the forward map η(θ).
   surrogate = ParametricTrajectorySurrogate(
       pod_rank=3, dmd_rank=3, real_eig_limit=10.0).fit(Xu, cache.Y)
   fmap = TrajectoryForwardMap(surrogate, cache.bounds, cache.obs_names)

   # 3. Kennedy–O'Hagan observation model (η + δ + ε).
   feats = fmap.featurize(cache.Y)
   feature_scale = feats.std(0)
   obs = ObservationModel(feature_scale=feature_scale, tau=0.10)

   # 4. Layer C: train amortized NPE once.
   engine = NPEInferenceEngine(fmap, obs, cache.bounds, n_train=8000).train()

   # 5. Infer: data (a trajectory's features) → posterior over θ.
   data = fmap.featurize(cache.Y[0])         # any observed trajectory
   posterior = engine.sample_posterior(data, n=4000)   # (4000, d) samples

   # 6. Summarize.
   for name, lo, mid, hi in zip(
           cache.param_names,
           np.quantile(posterior, 0.05, axis=0),
           np.quantile(posterior, 0.50, axis=0),
           np.quantile(posterior, 0.95, axis=0)):
       print(f"{name:<32} {mid:.4g}  [90% CI {lo:.4g}, {hi:.4g}]")

For an *exact* posterior (deterministic-surrogate regime) or to cross-check NPE,
swap in the MCMC engine:

.. code-block:: python

   from inference.mcmc_engine import MCMCInferenceEngine
   mcmc = MCMCInferenceEngine(fmap, obs, cache.bounds)
   chain = mcmc.sample_posterior(data, n=6000)         # (≈6000, d) samples

Step 4 — Inferring against your own data
----------------------------------------

To infer parameters from a *measured* trajectory (rather than a cached
simulation), featurize it the same way and pass it to ``sample_posterior``:

.. code-block:: python

   measured = np.load("my_measured_trajectory.npy")     # (T, F), same observables
   from inference._cache import _resample_to_phase
   measured_phase = _resample_to_phase(measured, cache.n_phase)
   data = fmap.featurize(measured_phase)
   posterior = engine.sample_posterior(data, n=4000)

.. important::

   The posterior you get is **inference of the ``simData`` parameters under the
   model**. To make a defensible claim about the *true biological* parameters,
   fit real measurement data with the discrepancy term active and validate
   coverage on held-out real data — the :math:`\delta` machinery is already in
   place (raise ``tau`` to your estimated model-error magnitude). See the scope
   note in :doc:`inference_methodology`.

Re-pointing at richer caches
----------------------------

Nothing above assumes the single-generation prototype cache. To run on a
multi-generation, multi-seed, or different-observable cache, change ``--cache``:

.. code-block:: bash

   uv run python inference/c_inference_prototype.py --cache ./uq_cache_higher_order \
       --pod-rank 4 --n-train 12000

For multi-generation trajectories with division resets, the Layer-B surrogate
escalation path (windowing / Hankel embedding) is documented in
:doc:`inference_design_rationale`; the Layer-C engine code is unchanged.

Artifacts produced
------------------

* ``inference/b1_uq_cache_mass.json`` — B1 per-channel errors, baselines, coverage.
* ``inference/c_inference_uq_cache_mass.json`` — full Layer-C report: coverage
  tables, discrepancy ablation, per-parameter contraction and recovery, and the
  NPE↔MCMC cross-check.

Both are plain JSON, suitable for downstream plotting or inclusion in a report.

Troubleshooting
---------------

* **"surrogate diverged on N/M prior draws"** — the eigenvalue-constrained fit
  produced unstable operators for part of the prior. Tighten ``--real-eig-limit``
  or narrow the parameter bounds in the cache's design.
* **NPE/MCMC cross-check marginal** — raise ``--n-train`` (NPE is amortized; more
  training pairs cost a one-time pass and tighten agreement).
* **A channel never beats the baseline in B1** — that observable likely lacks
  parametric structure (e.g. a noisy derivative). Drop it from the observable set
  or rely on the structured channels; the gate reports which is which.
