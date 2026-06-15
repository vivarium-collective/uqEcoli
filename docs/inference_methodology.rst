Inference Methodology
=====================

This page documents **A** — the actual machine-learning / statistical-inference
methodology the ``inference/`` package implements to do *verifiable inference*
on the *E. coli* whole-cell model (WCM). It is the theoretical companion to
:doc:`inference_uq_integration` (how the UQ pipeline feeds it),
:doc:`inference_design_rationale` (why these choices), and
:doc:`inference_tutorial` (how to run it).

.. contents::
   :local:
   :depth: 2

Two meanings of "inference"
---------------------------

The word *inference* is used in two distinct senses, and this package is precise
about which it delivers.

* **Runtime / ML inference** — running a *trained* model forward to produce
  outputs (as in "neural-network inference"). A fitted surrogate that stands in
  for the expensive simulator qualifies. The framework already had this.
* **Statistical / scientific inference** — reasoning *from observed data back to
  unknown parameters*, with quantified uncertainty: a **posterior**
  :math:`p(\theta \mid \text{data})`. This requires a likelihood (or a
  likelihood-free substitute), a prior, and a posterior-estimation engine.

The ``inference/`` package implements the **second** sense. It closes the loop

.. math::

   \text{data} \;\longrightarrow\; p(\theta \mid \text{data})

where :math:`\theta` are ``simData`` parameters of the WCM, and validates that
the resulting posterior is *calibrated* — the property that makes it real
inference rather than confident-sounding point estimates.

The one idea underneath everything
-----------------------------------

The WCM is an **expensive, stochastic function** :math:`Y = f(X)`: turn a handful
of ``simData`` knobs :math:`X` (RNAP active fraction, ribosome elongation rate,
dry-mass fraction, …), run the simulation, read out observables :math:`Y`
(a growth-curve trajectory, gene expression, fluxes). One evaluation costs
minutes-to-hours of CPU; ParCa alone is ~70 min. Every analysis worth doing —
sensitivity, calibration, inference — needs :math:`f` evaluated thousands to
millions of times, which is infeasible against the raw simulator.

The strategy is therefore always the same: **run the real** :math:`f` **a few
dozen times, cache the** :math:`(X, Y)` **pairs, fit a cheap surrogate, and do
everything downstream against the surrogate.** Inference is one such downstream
task; the UQ pipeline produces the cache and the surrogate.

The three-layer architecture
----------------------------

Inference is built as a stack, each layer reusing the *same* cached
:math:`(X, Y)` so no layer pays the WCM cost twice:

.. math::

   \underbrace{(X,\; Y(t))}_{\texttt{uq sample}}
   \;\longrightarrow\;
   \begin{cases}
     \text{Layer A: PCE} & \to\ \text{scalar Sobol attribution} \\[2pt]
     \text{Layer B: ParametricDMD} & \to\ \text{trajectory } Y(t;X) \\[2pt]
     \text{Layer C: NPE} + \delta & \to\ p(\theta \mid \text{data})
   \end{cases}

* **Layer A — PCE** (in ``libuq``, upstream): a polynomial-chaos surrogate of
  *scalar* outputs that yields Sobol sensitivity indices analytically. See
  :doc:`sensitivity_analysis`.
* **Layer B — ParametricDMD** (``inference/parametric_dmd.py``): a *trajectory*
  surrogate that predicts the full time-course :math:`Y(t; X)` at unseen
  parameters. This is the forward map inference calls, because WCM data is a
  timeseries.
* **Layer C — Neural Posterior Estimation + discrepancy**
  (``inference/sbi_engine.py``, ``inference/observation.py``): the inference
  engine proper. Given observed data it returns a posterior over :math:`\theta`.

Layer B — the trajectory surrogate (Koopman / DMD)
--------------------------------------------------

A trajectory is a **snapshot matrix** :math:`Y \in \mathbb{R}^{T \times F}`
(time :math:`\times` observables). A *dynamical* model is a rule that steps the
state forward; the simplest is **linear**,

.. math::

   x_{t+1} = A\, x_t ,

which is the multi-variable generalization of exponential growth. A linear
operator :math:`A` decomposes into **modes** (eigenvectors — patterns across
observables) with **rates** (eigenvalues): real part :math:`>0` grows,
:math:`<0` decays, complex oscillates. **Dynamic Mode Decomposition (DMD)**
recovers the best :math:`A` directly from the snapshot matrix, with no equations.
The **Koopman** viewpoint justifies linearity even for nonlinear biology: in a
rich enough set of observable functions, the dynamics become linear and DMD
applies.

**ParametricDMD** makes the dynamics parameter-dependent. Each :math:`X` has its
own operator :math:`A(X)`; we fit :math:`A` at the simulated design points,
reduce the spatial dimension by POD, and **interpolate** the reduced dynamics
across parameter space (RBF) so a brand-new :math:`X` yields a constructed
:math:`A(X)` and hence a full predicted trajectory :math:`Y(t; X)` — *without
simulating*. The implementation wraps PyDMD's ``ParametricDMD`` over an
**eigenvalue-constrained** optimized-DMD fit (``BOPDMD`` with ``real_eig_limit``)
so that interpolated operators cannot produce eigenvalues whose real part blows
the forecast up over long horizons.

.. note::

   Trajectories are first warped onto a canonical cell-cycle **phase** grid
   :math:`\theta \in [0, 1]` because doubling time — hence trajectory length — is
   itself parameter-dependent. The duration is retained separately as a scalar
   observable.

Layer C — the inference engine (SBI / NPE)
------------------------------------------

The inverse problem is Bayesian. Given observed data :math:`d`,

.. math::

   p(\theta \mid d) \;\propto\; p(d \mid \theta)\, p(\theta) ,

with **prior** :math:`p(\theta)` (the parameter bounds already encoded by the UQ
sampler) and **likelihood** :math:`p(d \mid \theta)`. The posterior cannot be
written down, so we *sample* it.

The WCM is a **stochastic** simulator: a single :math:`\theta` maps to a
*distribution* of trajectories (seed-to-seed variation), so the likelihood is
implicitly marginalized over latent randomness and is **intractable**.
**Simulation-Based Inference (SBI)** sidesteps the likelihood entirely. In its
**Neural Posterior Estimation (NPE)** form, a neural conditional density
estimator :math:`q_\phi(\theta \mid x)` is trained on simulated pairs
:math:`(\theta_i, x_i)` so that, after one training pass, it returns the
posterior for *any* observed :math:`x` instantly (**amortized**). The training
pairs are exactly the :math:`(X, Y)` the UQ surrogate produces cheaply.

.. math::

   \theta_i \sim p(\theta), \quad
   x_i = g(\theta_i) \;\;\Longrightarrow\;\;
   q_\phi(\theta \mid x) \xrightarrow{\text{train}} p(\theta \mid x)

Here :math:`g` is the **generative observation model** below, not the raw
surrogate — discrepancy and noise are part of what the network learns to invert.

The Kennedy–O'Hagan observation model (model discrepancy)
---------------------------------------------------------

A surrogate (or even the WCM) is a *model of* the cell, not the cell. Ignoring
that gap biases inference: the parameters distort to "explain" the model's own
structural error, producing an overconfident, wrong posterior. The
**Kennedy & O'Hagan (2001)** framework makes the gap explicit:

.. math::

   d \;=\; \underbrace{\eta(\theta)}_{\text{surrogate forward map}}
        \;+\; \underbrace{\delta}_{\text{model discrepancy}}
        \;+\; \underbrace{\varepsilon}_{\text{observation noise}} .

* :math:`\eta(\theta)` — the Layer-B trajectory surrogate, summarized to a fixed
  feature vector.
* :math:`\delta` — a **systematic, correlated** simulator-vs-reality bias, modeled
  as a smooth Gaussian process across the feature/phase index with magnitude
  :math:`\tau`. It is drawn once and held fixed across a dataset (it is
  structural, not per-observation noise).
* :math:`\varepsilon` — independent observation noise, plus the surrogate's own
  predictive error :math:`\sigma_{\text{surrogate}}`, sourced from the Layer-B
  conformal residuals.

This package folds :math:`\delta` in **from the first training pass**, not as a
later correction. Because :math:`\delta, \varepsilon` and surrogate error are
Gaussian, the two engines marginalize them consistently: NPE by *simulating*
them into the training data, MCMC by *adding their covariances* into an analytic
Gaussian likelihood.

The exact-MCMC cross-check
--------------------------

For the current *deterministic-surrogate* prototype the likelihood is still
writable, so an affine-invariant ensemble MCMC (``emcee``) gives an
asymptotically **exact** posterior. It is used as a correctness reference: if the
amortized NPE posterior matches the exact MCMC posterior on the same observation,
the neural engine is trustworthy. Once genuinely stochastic data arrives and the
likelihood becomes intractable, MCMC drops out and NPE carries the inference.

Validation: what makes it *verifiable*
--------------------------------------

Two distribution-free checks turn "we produced a posterior" into "we produced a
*trustworthy* posterior":

* **Coverage / Simulation-Based Calibration (SBC).** On held-out cases where the
  truth :math:`\theta^\*` is known, a claimed :math:`q`-credible interval must
  contain :math:`\theta^\*` about :math:`q` of the time. Reporting coverage — not
  just a point estimate — is the bar for "real inference."
* **Conformal prediction (Layer B).** Distribution-free prediction bands from
  held-out residuals, used to report calibrated trajectory uncertainty and to
  source :math:`\sigma_{\text{surrogate}}`.

Honest scope of the claim
-------------------------

After validation passes, the defensible statement is:

   *Validated, discrepancy-aware statistical inference of* ``simData``
   *parameters* — a coverage-checked posterior, using a UQ surrogate as the fast
   forward map.

What this does **not** yet claim, and why:

#. **Interpolation, not extrapolation.** The surrogate is trustworthy only inside
   the sampled envelope :math:`[\text{lb}, \text{ub}]^d`.
#. **Inference of the true biological parameters.** That stronger claim requires
   real measurement data fit through the :math:`\delta` term; the machinery is in
   place, so it is a *data* step, not a *code* step.
#. **Scope is what was validated** — the parameters and observables coverage was
   actually measured on, not arbitrary ones.

See :doc:`inference_design_rationale` for why each methodological choice is the
right one for this domain.

References
----------

* Kennedy, M. C. & O'Hagan, A. (2001). *Bayesian calibration of computer models.*
  J. R. Statist. Soc. B 63(3), 425–464.
* Cranmer, K., Brehmer, J. & Louppe, G. (2020). *The frontier of simulation-based
  inference.* PNAS 117(48), 30055–30062.
* Papamakarios, G. & Murray, I. (2016). *Fast ε-free inference of simulation
  models with Bayesian conditional density estimation.* NeurIPS.
* Talts, S. *et al.* (2018). *Validating Bayesian inference algorithms with
  simulation-based calibration.* arXiv:1804.06788.
* Schmid, P. J. (2010). *Dynamic mode decomposition of numerical and experimental
  data.* J. Fluid Mech. 656, 5–28.
* Sako, T., Brunton, S. L. *et al.* — PyDMD: https://github.com/PyDMD/PyDMD
* Tejero-Cantero, A. *et al.* (2020). *sbi: A toolkit for simulation-based
  inference.* JOSS 5(52), 2505.
