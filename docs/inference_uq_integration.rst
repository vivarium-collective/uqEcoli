How UQ Feeds Inference
======================

This page documents **B** — how the uncertainty-quantification pipeline exposed
by the ``uq`` CLI/package fits into inference *in general*, and concretely how
the inference methodology (:doc:`inference_methodology`) **consumes the outputs**
of the ``uq`` pipeline. The short version: **UQ produces the artifacts inference
cannot afford to recompute**, and inference is a downstream consumer that never
touches the simulator itself.

.. contents::
   :local:
   :depth: 2

The division of labor
----------------------

.. list-table::
   :header-rows: 1
   :widths: 18 42 40

   * - Stage
     - What it does
     - Cost
   * - ``uq sample``
     - UQPC steps 1–3: design the parameter sweep, **run the WCM** at each point,
       cache :math:`(X, Y, Y(t))` to disk.
     - **Expensive** — the only step that touches the simulator.
   * - ``uq quantify``
     - UQPC steps 4–5: fit the PCE surrogate, read out Sobol indices.
     - Cheap, repeatable, simulator-free.
   * - ``inference/`` (this package)
     - Layer B trajectory surrogate + Layer C posterior inference.
     - Cheap — consumes the cache; **never re-simulates**.

UQ is the *forward* discipline (propagate parameter uncertainty out to
observables; attribute variance). Inference is the *inverse* discipline (reason
from observations back to parameters). The forward artifacts are exactly the
raw material the inverse problem needs — which is why building UQ first was the
prerequisite, not a detour.

The contract: what ``uq sample`` writes
---------------------------------------

The inference package is **fully decoupled** from ``libuq``/``uq``: it imports
none of their code and reads only the on-disk cache a ``uq sample`` run produces.
That cache directory is the entire interface:

.. code-block:: text

   uq_cache/
   ├── X.npy                     # (n, d)  parameter design points  (physical units)
   ├── Y.npy                     # (n, F)  aggregated scalar outputs
   ├── germ_train.npy            # (n, d)  the same design in PyTUQ germ space [-1,1]
   ├── timeseries/
   │   └── sample_XXXX.npy        # (T, F)  per-sample WCM trajectory  ← Layer B input
   └── metadata.json             # parameter_names, bounds, observable_columns, ...

Every field inference needs is here:

* ``X.npy`` + ``metadata.json["bounds"]`` → the parameter design and the **prior
  support** for the posterior.
* ``metadata.json["parameter_names"]`` / ``["observable_columns"]`` → labels for
  posterior marginals and observation features.
* ``timeseries/sample_XXXX.npy`` → the **snapshot matrices** that train the
  Layer-B ParametricDMD surrogate. The per-sample trajectory *is* DMD's input,
  so no new simulation is needed to build the trajectory surrogate.

This is loaded read-only by
:func:`inference._cache.load_trajectory_cache`, which also phase-aligns the
variable-length trajectories onto a common grid.

How methodology A uses each UQ output
-------------------------------------

The inference methodology reuses UQ outputs at three distinct points:

**1. The sampling design becomes the prior and the training distribution.**
The bounds the UQ user chose (``--params-file`` or the default six
``simData`` parameters) define the uniform prior :math:`p(\theta)` and the range
over which NPE draws its training :math:`\theta`. The germ-space design
(``germ_train.npy``) is the same space PyTUQ's PCE uses, keeping the two layers
consistent.

**2. The cached trajectories become the forward map** :math:`\eta(\theta)`.
Layer B fits ParametricDMD on the cached ``timeseries/`` snapshots and becomes
the cheap, differentiable trajectory predictor that Layer C calls millions of
times during inference. Without this surrogate, an inference loop against the
raw WCM would need :math:`10^5\!-\!10^6` minutes-long simulations — impossible.

**3. The UQ surrogate's own uncertainty parameterizes the likelihood.**
This is the tightest coupling. The Layer-B leave-one-out **conformal residuals**
(the held-out prediction errors, the same UQ diagnostic the framework reports
for the PCE) directly supply :math:`\sigma_{\text{surrogate}}`, the surrogate
error term in the Kennedy–O'Hagan observation model:

.. math::

   d = \eta(\theta) + \delta + \varepsilon, \qquad
   \operatorname{Var}(\varepsilon) = \sigma_{\text{obs}}^2 + \sigma_{\text{surrogate}}^2 .

In other words, **the UQ result is the parameterization of the inference phase**:
the emulator's measured error budget is folded into the posterior so it cannot
pretend the surrogate is exact. This is the concrete realization of §4 caveat #2
("a surrogate inherits, not reduces, the model's uncertainty").

Sensitivity (Sobol) informs identifiability
--------------------------------------------

The UQ pipeline's Sobol indices (Layer A, :doc:`sensitivity_analysis`) and the
inference posterior answer mirror-image questions and corroborate each other:

* **Sobol (forward):** *which parameters drive the variance of this observable?*
* **Posterior contraction (inverse):** *which parameters does this observable
  identify?*

A parameter with near-zero Sobol index on the observed quantity has little
forward influence — and the inference posterior for it will (correctly) stay
close to the prior, because the data carries no information to constrain it. The
inference report makes this explicit (posterior-std / prior-std per parameter),
so the two analyses agree on what the data can and cannot tell you.

The soundness constraint inherited from UQ
-------------------------------------------

The UQ pipeline enforces that the design axis :math:`P_{\text{design}}` (vEcoli
variants) and the sampling axis :math:`P_{\text{uq}}` (PCRV-sampled parameters)
stay **disjoint**, or the PCE variance decomposition double-counts. Inference
inherits the same constraint: a parameter cannot be both a fixed design knob and
an inferred dimension. Any cache used for inference must therefore come from a
sampling design in which :math:`P_{\text{design}} \perp P_{\text{uq}}`.

Re-pointing at richer caches
----------------------------

Because the interface is just the cache directory, the inference code is
agnostic to *how* the cache was produced. The same engine runs on:

* a local subprocess ``uq sample`` cache (default),
* an SMS-API remote ``uq sample`` cache (``--api-url``, see :doc:`cli_reference`),
* a future multi-generation, multi-seed cache (for the stochastic/division-reset
  paths) — produced by the same command with more ``--generations`` and seeds.

Re-point ``--cache`` and the identical inference code runs; nothing in the
methodology assumes the single-generation prototype cache.

See :doc:`inference_design_rationale` for *why* this UQ-first, surrogate-fed
design is the right one for the WCM domain, and :doc:`inference_tutorial` to run
the full path.
