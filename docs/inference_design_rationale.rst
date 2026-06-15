Design Rationale
================

This page documents **C** — *why* the methodology of :doc:`inference_methodology`
and the UQ integration of :doc:`inference_uq_integration` are the **ideal choices
for this domain**, weighed honestly against the alternatives we could have
picked. The domain has three defining features that drive every decision:

#. the WCM is **expensive** (minutes-to-hours per run; ParCa ~70 min),
#. it is a **stochastic** generator (seed-to-seed variation; intractable
   likelihood),
#. its phenotype is a **trajectory** (the cell-cycle time-course), not a scalar.

.. contents::
   :local:
   :depth: 2

Decision 1 — Surrogate-based inference (vs. inference on the raw WCM)
---------------------------------------------------------------------------

**Choice:** do inference against a cheap surrogate fit to a cached
:math:`(X, Y)` design, never against the simulator directly.

**Why it is forced by the domain.** A posterior needs :math:`10^5\!-\!10^6`
forward evaluations. At minutes-to-hours each, raw-WCM inference is not "slow,"
it is *infeasible* — geological wall-clock. The surrogate evaluates in
microseconds, making the inference loop affordable. This is not an optimization;
it is the only way the problem is tractable at all.

**Alternative rejected:** direct ABC/MCMC on the WCM. Correct in principle,
impossible in practice at WCM cost.

Decision 2 — A *trajectory* forward map (vs. inferring on scalar PCE)
---------------------------------------------------------------------------

**Choice:** Layer B (ParametricDMD) predicts the full trajectory
:math:`Y(t; X)`, and inference matches trajectory data.

**Why.** The phenotype of the WCM is its time-course — the growth curve, the
cell-cycle shape, expression dynamics. Inverting against a single scalar endpoint
(e.g. final doubling time) discards exactly the structure that makes the data
informative. Matching the trajectory uses all of it.

**Alternative rejected:** MCMC on a scalar PCE surrogate. It is the *quick* path,
but for a timeseries generator it is the wrong forward model — it throws away the
temporal signal and, as Decision 4 explains, assumes a likelihood the stochastic
simulator does not have. The PCE remains valuable for *sensitivity* (Layer A);
it is simply not the forward map for trajectory inference.

Decision 3 — DMD/Koopman for the trajectory surrogate (vs. neural emulator or GP)
---------------------------------------------------------------------------------

**Choice:** ParametricDMD over an eigenvalue-constrained optimized DMD.

**Why it fits.** The B0 gate measured that single-generation mass dynamics are
**rank-2** and almost perfectly linear in a Koopman sense (≈99.7 % variance on the
mass/volume channels). When the dynamics *are* low-rank linear, DMD is the
maximally data-efficient, interpretable choice: it needs a few dozen trajectories
(not thousands), the eigenvalues *are* the growth/oscillation rates (mechanistic
meaning for free), and ParametricDMD extends it to unseen parameters with a
principled POD-plus-interpolation construction.

**Alternatives rejected (for now):**

* **Neural ODE / RNN emulator** — handles arbitrary nonlinearity but is
  data-hungry and opaque; unjustified when the dynamics are demonstrably rank-2.
* **Gaussian-process trajectory model** — excellent native uncertainty but scales
  :math:`O(n^3)` and gives no dynamical (eigenvalue) interpretation.
* **SINDy** — the right escalation *if* linearity fails (e.g. the multi-generation
  division sawtooth defeats time-delay embedding). It is held in reserve behind
  the B0/B1 decision rule, not adopted pre-emptively.

The honest guardrail: the B0 result is for *smooth single-generation* traces. The
division-reset discontinuity is deferred to a multi-generation cache, where the
documented escalation path (Hankel/time-delay embedding, then SINDy) applies.

Decision 4 — Simulation-Based Inference / NPE (vs. likelihood-based MCMC)
-------------------------------------------------------------------------------

**Choice:** amortized Neural Posterior Estimation as the headline inference
engine; ``emcee`` MCMC only as an exact cross-check.

**Why it is the right tool for a stochastic simulator.** The WCM's likelihood is
implicitly marginalized over latent seed randomness — it is **intractable**. A
clean Gaussian likelihood is an approximation that pretends the simulator is
deterministic. SBI is built precisely for intractable-likelihood simulators: it
needs only the ability to *simulate* :math:`(\theta, x)` pairs, which the
surrogate provides cheaply, and NPE additionally gives an **amortized** posterior
— train once, infer on any new dataset instantly. Its training set is *already*
the cached :math:`(X, Y)`.

.. list-table:: Inference engines weighed for this domain
   :header-rows: 1
   :widths: 22 50 28

   * - Method
     - Verdict for the WCM
     - Role here
   * - **NPE (SBI)**
     - Likelihood-free, amortized, consumes the cache directly, handles
       stochasticity natively. **Best fit.**
     - Headline engine.
   * - MCMC on a likelihood
     - Exact when the likelihood is writable — true only for the *deterministic*
       prototype, not the stochastic target.
     - Exact cross-check; retires once data is stochastic.
   * - Scalar-PCE MCMC
     - Discards the trajectory; assumes a Gaussian likelihood the simulator lacks.
     - Not used for trajectory inference.
   * - ABC
     - Assumption-light but sample-inefficient; thresholded cloud, not a
       parametric posterior.
     - Sanity baseline only.

.. note::

   This revises the original plan ordering. The plan first ranked MCMC above SBI
   for the small (≤10-parameter) regime — correct *if* the observable is scalar
   with a writable Gaussian likelihood. Once the target is explicitly a
   *stochastic timeseries*, that premise no longer holds and SBI is the
   principled headline choice. MCMC is retained for what it is genuinely best at
   here: an exact reference to validate the neural engine while the prototype
   likelihood is still tractable.

Decision 5 — Model discrepancy from the start (vs. noise-only, or bolt-on later)
--------------------------------------------------------------------------------

**Choice:** include the Kennedy–O'Hagan discrepancy term :math:`\delta` in the
observation model from the first training pass.

**Why.** A surrogate (and the WCM itself) is a *model of* the cell, not the cell.
With a noise-only likelihood, the parameters absorb the model's systematic error
and the posterior becomes biased and overconfident. The discrepancy ablation
demonstrates this empirically: on δ-contaminated data, ignoring :math:`\delta`
makes a nominal 90 % credible interval cover only ~72 % of the time, while
modeling it restores ~86 %. Adding :math:`\delta` later would mean re-deriving
and re-validating the whole engine; building it in is both cheaper and the
difference between *inference under the model* and *inference about the cell*.

**Alternative rejected:** independent Gaussian noise only. Simpler, but produces
confidently wrong posteriors exactly where the surrogate is structurally biased.

Decision 6 — Calibration/coverage as the acceptance test (vs. point accuracy)
-----------------------------------------------------------------------------

**Choice:** the engine is accepted only when its credible intervals are
*calibrated* (coverage ≈ nominal on held-out truth, via SBC), plus an NPE↔MCMC
agreement check.

**Why.** A posterior that is sharp but miscalibrated is worse than no posterior —
it is confidently wrong. Coverage is the property that makes the word "inference"
verifiable rather than aspirational, and it is the same empirical discipline the
UQ pipeline already applies to the PCE (train/test relative error, k-fold CV).

**Alternative rejected:** reporting a point estimate / MAP without coverage.
Cheaper to produce, impossible to trust.

Summary
-------

Every choice traces back to the three domain features. *Expensive* forces a
surrogate (D1) and rules out raw-WCM inference. *Trajectory phenotype* forces a
trajectory forward map (D2) and a low-rank dynamical surrogate that fits the
measured structure (D3). *Stochastic* forces a likelihood-free, amortized engine
(D4) and an explicit discrepancy term (D5). And *scientific credibility* forces
calibration, not point accuracy, as the bar (D6). The result is the minimal stack
that is simultaneously affordable, faithful to the data's structure, and honest
about its own uncertainty.

Continue to :doc:`inference_tutorial` to run it end to end.
