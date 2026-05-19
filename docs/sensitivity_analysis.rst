Sensitivity Analysis
====================

The UQ framework implements global sensitivity analysis (GSA) using
Polynomial Chaos Expansion (PCE) surrogates and Sobol indices, as
specified in the Milestone 08.4.2 requirements.

Overview
--------

Sensitivity analysis answers: **Which input parameters have the greatest
influence on simulation outputs?**

The framework uses:

* **PCE (Polynomial Chaos Expansion)** — builds a polynomial surrogate
  from simulation data
* **Sobol indices** — closed-form variance decomposition from PCE
  coefficients (Sudret 2008)
* **PyTUQ** — Sandia National Labs' UQ library for the PCE math

Sobol Sensitivity Indices
-------------------------

Sobol indices decompose output variance into contributions from each input.

First-Order Index (:math:`S_i`)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Measures the **main effect** — variance explained by varying :math:`X_i`
alone:

.. math::

   S_i = \frac{V[E[Y|X_i]]}{V[Y]}

Total-Order Index (:math:`S_{Ti}`)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Measures the **total effect** including all interactions involving
:math:`X_i`:

.. math::

   S_{Ti} = 1 - \frac{V[E[Y|X_{\sim i}]]}{V[Y]}

**Interpretation:**

* :math:`S_i \approx S_{Ti}` — parameter acts mainly independently
* :math:`S_{Ti} \gg S_i` — parameter has significant interactions
* :math:`\sum S_i < 1` — interaction effects exist in the model

PCE Surrogate Method
--------------------

PCE replaces the expensive simulator with a polynomial approximation:

.. math::

   \hat Y(\boldsymbol\xi) = \sum_{|\alpha|\le p} c_\alpha\,\Phi_\alpha(\boldsymbol\xi)

where :math:`\Phi_\alpha` are multivariate Legendre polynomials (the
orthogonal basis for uniform priors on :math:`[-1,1]^d`) and
:math:`c_\alpha` are coefficients fit by regression.

Advantages
^^^^^^^^^^

* **Analytic Sobol** — indices computed in closed form from coefficients
  (no additional Monte Carlo sampling)
* **Nonlinear** — polynomial basis captures interactions up to order *p*
* **Fast** — surrogate evaluates in microseconds once fitted

Polynomial Order Selection
^^^^^^^^^^^^^^^^^^^^^^^^^^

.. list-table::
   :header-rows: 1
   :widths: 20 40 40

   * - Order
     - Pros
     - Cons
   * - 1-2
     - Few samples needed, fast
     - May miss nonlinear effects
   * - 3
     - Good balance (default)
     - Moderate sample requirements
   * - 4+
     - Captures complex behaviour
     - Many samples, overfitting risk

**Rule of thumb**: start at order 3; increase if training error > 10%.

Sample Size
^^^^^^^^^^^

The number of terms in the PCE basis is:

.. math::

   P = \binom{d + p}{p}

For *d* = 6 parameters and *p* = 2: *P* = 28.  For *p* = 3: *P* = 84.
The rule :math:`N \ge 2P` is a safe starting point.

Regression Backends
^^^^^^^^^^^^^^^^^^^

Three PyTUQ regression methods are available via ``--regression``:

* **lsq** (default) — ordinary least squares: :math:`c = (A^TA)^{-1}A^T y`
* **bcs** — Bayesian Compressed Sensing: sparse, good when :math:`N < P`
* **anl** — analytical Bayesian projection

Usage
-----

Via CLI (typical):

.. code-block:: bash

   # Stage 1: sample + evaluate
   uv run uq sample /path/to/simData.cPickle \
       --n-samples 50 --n-test 10 \
       --cache-dir ./cache

   # Stage 2: fit PCE + compute Sobol
   uv run uq quantify /path/to/simData.cPickle \
       --cache-dir ./cache \
       --export-path ./results \
       --polynomial-order 3 \
       --regression lsq

Programmatic (advanced):

.. code-block:: python

   from uq.workflow import run_uqpc_from_cache, UQPCResult

   result: UQPCResult = run_uqpc_from_cache(
       cache_dir="./cache",
       sim_data_path="/path/to/simData.cPickle",
       polynomial_order=3,
       regression="lsq",
       n_bins=10,
   )

   # Per-strategy Sobol indices
   for name, si in result.sobol_indices[0].items():
       print(f"{name}: S_Ti = {si.total_order:.4f}")

Multi-Output Variance Weighting
-------------------------------

When multiple observables exist, per-output Sobol indices are combined by
variance-weighting:

.. math::

   S_i = \sum_{j=1}^{m} \frac{\text{Var}[Y_{:,j}]}{\sum_k \text{Var}[Y_{:,k}]}\,S_i^{(j)}

This produces a single :math:`S_i` / :math:`S_{Ti}` per parameter that
represents importance *across all outputs*.

Interpretation
--------------

* **Top driver** (highest :math:`S_{Ti}`): change this parameter to get the
  biggest change in the output.  For the default parameter set,
  ``basal_elongation_rate`` (ribosome speed) is typically the top driver.
* **Negligible** (:math:`S_{Ti} \approx 0`): fixing this parameter to a
  constant does not meaningfully reduce output variance.
* **Interactive** (:math:`S_{Ti} \gg S_i`): the effect of this parameter
  depends on the value of other parameters — look at joint effects.

Cross-Condition Comparison
--------------------------

When ``uq sample`` is run with ``--conditions`` (multi-parca), ``uq quantify``
automatically produces a cross-condition comparison with rank stability
metrics.  See :doc:`cli_reference` for the ``--conditions`` flag.

See Also
--------

* :doc:`tutorial_workflow` — full mathematical walkthrough
* :doc:`cli_reference` — ``--regression``, ``--polynomial-order`` flags
* `PyTUQ UQPC Workflow <https://sandialabs.github.io/pytuq/apps/uqpc.html>`_
* Sudret, B. (2008). *Global sensitivity analysis using polynomial chaos
  expansions*. Reliability Engineering & System Safety 93(7), 964-979.
