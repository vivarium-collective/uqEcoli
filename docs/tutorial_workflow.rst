Tutorial — the UQPC workflow, end to end
========================================

This tutorial walks through the full PyTUQ UQPC pipeline as it is
implemented in ``uqEcoli``, explaining *why* each step is there and
*what* it is computing.  It is intentionally long: read it once, then
use the :doc:`cli_reference` to drive runs.

The upstream reference is
<https://sandialabs.github.io/pytuq/apps/uqpc.html>.  Every step below
maps to exactly one block in ``uq/workflow.py``; function names are
quoted in brackets.

Notation
--------

* :math:`\mathbf{x}\in\mathbb{R}^d` — physical-space parameters.
* :math:`\mathbf{\xi}\in[-1,1]^d` — **germ** variables (the input
  PC variables; Legendre basis for uniform priors).
* :math:`Y(\mathbf{x})\in\mathbb{R}^m` — time-averaged observables
  returned by one vEcoli run.
* :math:`\hat Y(\boldsymbol\xi) = \sum_{\alpha} c_\alpha\,\Phi_\alpha(\boldsymbol\xi)`
  — the PCE surrogate.
* :math:`\Phi_\alpha` — multivariate Legendre polynomial indexed by the
  multi-index :math:`\alpha=(\alpha_1,\dots,\alpha_d)` with
  :math:`|\alpha|\le p`.

Step 1 — setup inputs  [``_setup_input_pc``]
--------------------------------------------

.. math::

   \mathbf{x}(\boldsymbol\xi) = \underbrace{\frac{1}{2}(\mathbf{lb}+\mathbf{ub})}_{\text{midpoints}}
                             + \underbrace{\tfrac{1}{2}(\mathbf{ub}-\mathbf{lb})}_{\text{half-ranges}}\,\boldsymbol\xi

This is an affine map from the Legendre germ :math:`\boldsymbol\xi\in[-1,1]^d`
to the physical parameter box :math:`[\mathbf{lb},\mathbf{ub}]`.  Encoded
as a PCRV, the first row of the coefficient matrix holds the midpoints
and the remaining rows form ``diag(half_ranges)``.  PyTUQ does the rest.

Why Legendre?  Uniform priors on a bounded interval are exactly the
orthogonality measure of the Legendre polynomials; we get Sobol indices
in closed form from the PC coefficients (Sudret 2008).

Step 2 — generate samples  [``_generate_training_samples``]
-----------------------------------------------------------

.. math::

   \boldsymbol\xi^{(i)} \sim U\!\left([-1,1]^d\right),
   \qquad i=1,\dots,N

   \mathbf{x}^{(i)} = \mathbf{x}\!\left(\boldsymbol\xi^{(i)}\right)

implemented as ``input_pc.sampleGerm(N)`` followed by
``input_pc.evalPC``.

When ``--n-test`` is nonzero, :math:`N_{\text{test}}` additional samples
are drawn from the *same* measure and **appended** to the training
batch before we run vEcoli — this is the PyTUQ UQPC ``--ntst`` mode.  The
validation samples are split back off after collection, so there is one
and only one vEcoli workflow invocation per ``uq sample`` call.

.. note::

   PyTUQ is the single source of truth for sampling.  There is no
   hand-rolled LHS, Monte Carlo, or quadrature code inside ``uqEcoli``.

Step 3 — evaluate the simulation  [``uq/cli.py::sample``]
----------------------------------------------------------

Each row of :math:`X` is translated into one *variant* using vEcoli's
``sim_data_setattr`` variant function:

.. code-block:: json

   {
     "variants": {
       "sim_data_setattr": {
         "mutations": {
           "value": [
             {"process.translation.basal_elongation_rate": 21.3},
             {"process.translation.basal_elongation_rate": 19.8},
             ...
           ]
         }
       }
     }
   }

The ``{"value": [...]}`` form is the canonical variant-expansion
grammar documented at
<https://covertlab.github.io/vEcoli/workflows.html#variants>
and parsed by ``runscripts/create_variants.py::parse_variants``.  Each
list entry becomes one pickled ``variant_sim_data_NNNN.cPickle`` that
vEcoli then runs for the requested number of generations and lineage
seeds.

vEcoli emits hive-partitioned Parquet:

.. code-block:: text

   history/experiment_id=.../variant=.../lineage_seed=.../generation=.../agent_id=.../000.pq

``uqEcoli`` reads this tree with
``polars.read_parquet(..., hive_partitioning=True)`` and extracts
observables according to the ``--observables`` presets.  Each preset
mirrors a cd1 analysis module from the Vegas/Bermuda CD1 deliverables:

================  ================================================  ========
Preset            cd1 module                                         Features
================  ================================================  ========
``mass``          cd1_higher_order_properties (raw)                  5
``higher_order``  cd1_higher_order_properties (derived)              6
``exchange_fluxes`` cd1_exchange_fluxes                              ~87
``transcriptome`` cd1_transcriptomics                                ~4,300
``proteome``      cd1_proteomics                                     ~4,300
``fluxome``       cd1_fluxomics (dry-mass normalized)                ~2,800
================  ================================================  ========

``--generation-lower-bound N`` filters early generations before
aggregation (same SQL logic as the cd1 ``generation_lower_bound``
parameter).

The time-averaged observable vector becomes one row of :math:`Y`; the
raw timeseries is kept under ``timeseries/`` for strategies 2-4.

**Bring-your-own-variants (``--variants-source base-config``)**

The default contract — one PCRV sample row ↔ one ``sim_data_setattr``
variant — can be inverted.  If you already have a vEcoli config JSON
with a fully-spec'd ``variants`` block (e.g. a previously-curated sweep,
a multi-condition design, a hand-picked kinetic-feature scan), pass it
with ``--variants-source base-config --base-config <PATH>`` and:

1. PCRV sampling is skipped entirely (no ``sampleGerm`` / ``evalPC``).
2. ``uq.vecoli_config._x_from_variants()`` reverse-maps mutation values
   into :math:`X` row-by-row, column-ordered by ``--params-file``'s
   ``attr_path`` list.  The germ matrix :math:`\Xi` is then derived via
   the affine inverse of step 1's map.
3. :math:`N_{\text{samples}}` is **forced** to the length of the
   variants list.  ``--n-test`` is ignored (no validation surface
   without PCRV draws).

Because :math:`N` is now constrained by the user's variants count
rather than chosen for PCE basis-size needs, ``uq sample`` prints a
**PCE adequacy diagnostic** (red/yellow/dim) based on
:math:`N_{\text{samples}} / \binom{p+d}{d}` at the default
``--polynomial-order 2`` quantify setting.  The advisory tells you
exactly how many more variants to add, what order to drop to, or
whether ``--regression bcs`` is the better fit.

BYO mode is local-only: forbidden with ``--backend v2ecoli`` and with
``--api-url`` (remote mutation pushdown is unfinished).  The variants
block must use ``sim_data_setattr`` — other modules cannot be
reverse-mapped to scalar :math:`X` values and the cache will be
incompatible with ``uq quantify``.

**v2ecoli backend (``--backend v2ecoli``)**

An alternative in-process backend uses the ``v2ecoli`` process-bigraph
composite instead of the subprocess + Nextflow orchestration.  The steps
differ in execution but produce the same ``PrecomputedCache`` format:

1. **Cache bundle generation** — ``generate_cache_bundle()`` extracts the
   initial state and sim_data from ``simData.cPickle`` into
   ``cache_dir/{initial_state.json, sim_data_cache.dill}``.
2. **Mutation** — each sample's parameter values are applied in-memory
   via ``apply_mutations_to_configs()``, which deep-copies the baseline
   ``configs`` dict and updates the specified sim_data dot-paths.
3. **Composite build** — ``build_v2ecoli_composite()`` constructs a
   v2ecoli ``Composite`` with listeners for the requested observable
   presets (MassListener, RNACounts, MonomerCounts, FBA results, etc.).
4. **Tick loop** — ``run_v2ecoli_composite()`` advances the composite
   for ``N_generations * ticks_per_generation`` steps.
5. **Observable extraction** — ``extract_timeseries()`` traverses
   ``composite.state['agents']`` directly using the path maps in
   :py:mod:`uq.v2ecoli_bridge` (``OBSERVABLE_PATHS`` for scalars,
   ``ARRAY_OBSERVABLE_PATHS`` for arrays like transcriptome/proteome).

Parallelism is managed via ``ProcessPoolExecutor`` (controlled by
``--max-workers``), unlike the Nextflow-managed vEcoli backend.

The `v2ecoli` preset table maps v2ecoli listeners to UQ observables:

.. code-block:: text

   mass            → MassListener (cell mass, volume, DNA/RNA fractions)
   higher_order    → derived from mass (doubling time, growth rate)
   transcriptome   → RNACounts listener
   proteome        → MonomerCounts listener
   fluxome         → FBA results (dry-mass normalized)
   exchange_fluxes → FBA results (external metabolite fluxes)

Array observables (transcriptome, proteome, fluxome, exchange_fluxes)
are flattened to ``{preset}_{index}`` column names at extraction time.

Step 4 — PCE surrogate fit  [``_fit_surrogate``]
------------------------------------------------

Given germ samples :math:`\Xi\in\mathbb{R}^{N\times d}` and outputs
:math:`Y\in\mathbb{R}^{N\times m}`, we want

.. math::

   Y_j(\boldsymbol\xi) \;\approx\; \hat Y_j(\boldsymbol\xi)
     \;=\; \sum_{|\alpha|\le p} c_{j,\alpha}\,\Phi_\alpha(\boldsymbol\xi)
     \qquad j = 1,\dots,m

The multi-index set comes from PyTUQ's ``get_mi(p, d)``; the design
matrix :math:`A\in\mathbb{R}^{N\times|\alpha|}` comes from
``pcrv.evalBases(Ξ, 0)``.  Coefficients are fit per output column using
one of the PyTUQ regression backends:

* **lsq** — ordinary least squares: :math:`c = (A^TA)^{-1}A^T y`.
* **bcs** — Bayesian Compressed Sensing (sparse, regularized):
  solves the :math:`\ell_1`-regularized problem with tolerance
  ``--tol``.  Preferred when :math:`N < |\alpha|`.
* **anl** — analytical Bayesian projection.

The fitted coefficients are pushed back into the PCRV via
``setMiCfs`` so that subsequent Sobol queries act on the surrogate.

Step 5 — relative errors  [``_compute_relative_errors``]
--------------------------------------------------------

For each output column :math:`j`:

.. math::

   \varepsilon_j = \frac{\| Y_{:,j} - \hat Y_{:,j} \|_2}{\| Y_{:,j} \|_2}

Reported on the training set (always) and the held-out test set (when
``uq sample --n-test > 0``).  These are the numbers the ``uq quantify``
Rich report prints under the **SURROGATE QUALITY** panel.

A large training error means the PCE order is too low or the model has
a discontinuity the smooth basis cannot capture; a large test error
with small training error is the classic over-fit signature and is why
``--n-test`` matters.

Step 6 — Sobol decomposition  [``_compute_sobol``]
---------------------------------------------------

Because the PCE basis is orthogonal under the input measure, variance
is a diagonal sum over coefficients:

.. math::

   \mathrm{Var}[\hat Y_j] = \sum_{|\alpha|\ge 1} c_{j,\alpha}^2\,\|\Phi_\alpha\|^2

The first-order Sobol index for parameter :math:`i` uses only the
multi-indices that touch dimension :math:`i` alone:

.. math::

   S_{i}^{(j)} = \frac{1}{\mathrm{Var}[\hat Y_j]}
     \sum_{\alpha\in\mathcal{A}_i} c_{j,\alpha}^2\,\|\Phi_\alpha\|^2,
     \qquad
     \mathcal{A}_i = \{\alpha : \alpha_i>0,\ \alpha_{k\ne i}=0\}

The total-order index sums over every multi-index that has a nonzero
entry at position :math:`i`:

.. math::

   S_{T,i}^{(j)} = \frac{1}{\mathrm{Var}[\hat Y_j]}
     \sum_{\alpha:\alpha_i>0} c_{j,\alpha}^2\,\|\Phi_\alpha\|^2

PyTUQ implements the book-keeping in
``PCRV.computeSens``/``computeTotSens``/``computeJointSens``.  For
multi-output models we variance-weight across outputs:

.. math::

   S_i \;=\; \sum_{j=1}^{m} \frac{\mathrm{Var}[Y_{:,j}]}{\sum_k \mathrm{Var}[Y_{:,k}]}\,S_i^{(j)}

so the final ``first_order`` / ``total_order`` vectors are single
numbers per parameter that can be rendered as bar charts.

Step 6b — bootstrap Sobol CIs  [``_bootstrap_sobol_cis``]
---------------------------------------------------------

Single-fit Sobol indices are point estimates with no uncertainty
quantification.  ``S_{T_i} = 0.42`` for one parameter and
``S_{T_i} = 0.38`` for another could be a real ranking or it could be
fit noise — without CIs the user cannot tell.

``--bootstrap N`` on ``uq quantify`` runs the standard non-parametric
bootstrap (Archer/Iooss/Saltelli convention) on top of steps 4–6:

.. code-block:: text

   for b in range(N):
       idx = sample n_samples row indices with replacement
       refit PCE on (germ[idx], Y[idx]) via the same regression backend
       recompute Sobol via PCRV.computeSens / computeTotSens
   report per-parameter 2.5–97.5 percentile interval

This is method-agnostic — works identically with ``lsq``, ``bcs``, and
``anl`` regression.  Threaded through all four RFC006 strategies via
the ``n_bootstrap`` parameter on ``run_uqpc``.  Sobol tables grow from
``S_Ti`` to ``S_Ti  [low, high]``.  Typical N = 200 (sub-second for
moderate output dimensions).

Step 6c — surrogate quality and noise floor diagnostics
--------------------------------------------------------

Three additional panels surface what raw Sobol numbers can hide.

**Design quality at the actual order** —
``_print_quantify_adequacy_panel`` re-runs the count adequacy +
condition-number checks at the ``--polynomial-order`` actually used in
the current ``quantify`` run.  The sample-time check assumed order 2;
if the user passes ``--polynomial-order 3`` the basis size jumps from
:math:`\binom{8}{6} = 28` to :math:`\binom{9}{6} = 84` and an ok design
at p=2 can become underdetermined silently.  Status:
``ok`` / ``marginal`` / ``underdetermined`` for the count check;
``well-conditioned`` / ``marginal`` / ``ill-conditioned`` / ``singular``
for the κ(A) check.

**Cross-validation when no held-out test set is available** —
``_k_fold_cv_error`` runs 5-fold CV using the same PyTUQ regression
backends as the main fit.  Replaces ``--n-test`` validation in BYO
mode or under ``--n-test 0``.  Cell ``CV (5-fold)`` appears in the
``SURROGATE QUALITY`` panel.  Skip threshold:
:math:`N \ge 2.5 \cdot \text{basis\_size}` — under that, each fold's
training set is itself under-fit and the CV error is just noise.

**Aleatoric vs epistemic decomposition** —
``_aleatoric_noise_decomposition`` uses vEcoli's ``lineage_seed``
replicate structure already in the cache to split total :math:`Y`
variance into:

.. math::

   \mathrm{Var}[Y]_{\text{total}}
     = \underbrace{\mathrm{Var}_i\!\left[\bar Y_i\right]}_{\text{between-variant: epistemic}}
     + \underbrace{\mathrm{E}_i\!\left[\mathrm{Var}_k\!\left[Y_{i,k}\right]\right]}_{\text{within-variant: aleatoric}}

where :math:`Y_{i,k}` is the per-seed mean for variant :math:`i`,
replicate :math:`k`, and :math:`\bar Y_i` is the variant grand mean.
Signal fraction :math:`\eta^2_{\text{between}} = V_{\text{between}} /
V_{\text{total}}` reports the share of variance the PCE could possibly
explain.  Color-coded verdict:

* ≥ 80%: signal dominates — interpret Sobol straightforwardly
* 50–80%: moderate aleatoric — Sobol undershoots the per-parameter
  share of *explainable* variance by ~ :math:`1/\eta^2\times`
* < 50%: aleatoric exceeds parameter signal

When ``n_init_sims = 1`` (the default), the panel reports
``not estimable`` with an advisory to rerun ``uq sample`` with
``--n-init-sims 4`` (the recommended minimum for noise-floor
estimation).

Aggregation strategies (RFC006 §3)
----------------------------------

``quantify`` runs the whole steps-4-through-6 pipeline **four times**,
on four different aggregations of the cached timeseries.  They share
:math:`X` but differ in how :math:`Y` is computed.

1. **Uniform / bulk** — :math:`Y_{i,k} = \overline{y}_{i,k}` (mean over
   all timesteps of all cells).  Baseline global GSA.

2. **By generation** — for each unique generation :math:`g` present in
   *all* samples, :math:`Y^{(g)}_{i,k}` is the mean over only the rows
   labelled ``generation == g``.  Tells you which parameters matter
   *after* the population has converged to steady-state growth.

3. **By lineage seed** — same idea with ``lineage_seed``.  Isolates
   which parameters drive stochastic lineage-to-lineage variance.

4. **Growth-stratified** — timesteps are binned into
   :math:`n_{\text{bins}}` cell-cycle stages by the dimensionless
   growth progress variable

   .. math::

      \theta(t) = \frac{\log m(t) - \log m_\text{birth}}{\log m_\text{div} - \log m_\text{birth}}

   (monotonic in :math:`[0,1]` — no spectral decomposition required).
   The per-stage mean observables are stacked into one big PCE output
   and a Sobol index is extracted per stage.  This is the heatmap you
   see in the dashboard's "Sensitivity Spectrogram" panel.

Why the cache?
--------------

Steps 1-3 are expensive (one vEcoli simulation per variant).  Steps 4-6
are essentially free (matrix math on a cached :math:`(X, Y)` pair).
Splitting the workflow at the cache boundary means you can:

* Re-fit with a higher ``--polynomial-order`` without re-simulating.
* Swap between ``lsq``/``bcs``/``anl`` regression backends.
* Change ``--n-bins`` for the growth-stratified strategy.
* Rebuild the dashboard/TUI artifacts from the same cache.

And you can ship the cache to another machine that does not have vEcoli
or v2ecoli installed — ``quantify`` only needs ``simData.cPickle`` to
reconstruct the parameter space metadata, and is **backend-agnostic**
(it reads the same ``PrecomputedCache`` format regardless of whether
samples were generated by ``--backend vecoli`` or ``--backend v2ecoli``).

Reading the export directory
----------------------------

``QuantifyResult.export(path)`` writes:

.. code-block:: text

   uq_results/
   ├── uq_results.json              # dashboard schema — all strategies
   ├── population_sobol/            # strategy 1
   │   ├── first_order.npy
   │   └── total_order.npy
   ├── population_surrogate/        # strategy 1 PCE coefficients
   ├── generation_{g}_sobol/        # strategy 2 — one dir per generation
   ├── seed_{s}_sobol/              # strategy 3 — one dir per lineage
   ├── growth_stage_{i}_sobol/      # strategy 4 — one dir per bin
   └── growth_stratified_surrogate/ # strategy 4 combined PCE

Every ``.npy`` is a 1-D float array indexed by parameter name (which is
also in ``uq_results.json``).  The two surrogate directories each
contain ``coefficients.npy``, ``multi_indices.npy``, and the metadata
needed to re-evaluate :math:`\hat Y(\boldsymbol\xi)` without ``pytuq``
installed.

Further reading
---------------

* :doc:`sensitivity_analysis` — deeper coverage of the PCE algebra.
* :doc:`aggregation_strategies` — worked examples for strategies 2-4.
* :doc:`cell_cycle` — background on the growth-progress variable
  :math:`\theta(t)`.
