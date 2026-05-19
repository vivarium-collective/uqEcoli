Aggregation Strategies
======================

The UQ framework implements four aggregation strategies that determine how
the per-timestep simulation output is reduced to a single response vector
:math:`Y` per sample before PCE fitting.  All four share the same input
samples :math:`X` but produce different :math:`Y` vectors, revealing which
parameters drive variance under different grouping assumptions.

Overview
--------

.. list-table:: Aggregation Strategies
   :header-rows: 1
   :widths: 20 30 50

   * - Strategy
     - Grouping
     - Purpose
   * - Uniform (Strategy 1)
     - None (all data)
     - Baseline population-level GSA
   * - By Generation (Strategy 2)
     - Cell generation number
     - Does parameter importance converge with cell age?
   * - By Lineage Seed (Strategy 3)
     - Stochastic lineage seed
     - Does sensitivity depend on random initial conditions?
   * - Growth-Stratified (Strategy 4)
     - Cell cycle stage (θ)
     - Does parameter importance shift within a cell cycle?

Strategy 1: Uniform / Population
--------------------------------

All timesteps across all cells are averaged into a single vector:

.. math::

   Y_{i,k} = \frac{1}{T_i}\sum_{t=1}^{T_i} y_{i,k}(t)

where :math:`y_{i,k}(t)` is the :math:`k`-th observable of sample
:math:`i` at timestep :math:`t`.  This is the "bulk" view — it answers
"which parameters drive the most variance across the whole population?"

The HTML report shows :math:`S_i` (first-order) and :math:`S_{Ti}`
(total-order) side by side for each parameter.  A large gap
(:math:`S_{Ti} \gg S_i`) signals interaction effects.

Strategy 2: By Generation
-------------------------

Timesteps are grouped by the ``generation`` metadata column before
averaging.  For each generation :math:`g` present across **all** samples:

.. math::

   Y^{(g)}_{i,k} = \frac{1}{T_{i,g}}\sum_{t: \text{gen}=g} y_{i,k}(t)

A separate PCE + Sobol decomposition is fit for each :math:`g`.  This
reveals whether parameter importance changes as the population converges
to steady-state growth.

Requires ``uq sample --generations N`` with :math:`N \ge 2`.  If only one
generation is available the report shows a note explaining why the strategy
was skipped.

Strategy 3: By Lineage Seed
---------------------------

Timesteps are grouped by ``lineage_seed``:

.. math::

   Y^{(s)}_{i,k} = \frac{1}{T_{i,s}}\sum_{t: \text{seed}=s} y_{i,k}(t)

Requires ``uq sample --n-init-sims N`` with :math:`N \ge 2`.

Strategy 4: Growth-Stratified
-----------------------------

Timesteps are binned by the growth-progress variable :math:`\theta \in [0,1]`
(:math:`\theta = 0` at birth, :math:`\theta = 1` at division):

.. math::

   \theta(t) = \frac{\log m(t) - \log m_{\text{birth}}}{\log m_{\text{div}} - \log m_{\text{birth}}}

:math:`n_{\text{bins}}` uniformly-spaced θ-stages are created (default 10),
and per-stage mean observables are stacked into one PCE output:

.. math::

   Y_{i}^{\text{gs}} = \left[Y_{i}^{(1)}, Y_{i}^{(2)}, \dots, Y_{i}^{(n_{\text{bins}})}\right]^\top

A single PCE is fit on this stacked vector, then :math:`S_{Ti}` is extracted
per stage.  The result is the **parameter × stage heatmap** shown in the
HTML report and dashboard — revealing how parameter importance evolves
within a single cell cycle.

See :doc:`cell_cycle` for the biological rationale and θ formula details.

Implementation
--------------

The four strategies are implemented in ``uq/workflow.py``:

.. code-block:: python

   from uq.workflow import (
       run_strategy1_uniform,
       run_strategy2_by_generation,
       run_strategy3_by_seed,
       run_strategy4_growth_stratified,
       run_uqpc,
   )

   # Each returns (SobolIndices, PCESurrogate)
   s1, p1 = run_strategy1_uniform(X, Y1, param_names, pc, order, regression)
   s4, p4 = run_strategy4_growth_stratified(X, timeseries, param_names,
                                            observables, pc, order, regression,
                                            n_bins=10)

They are called together by :py:func:`~uq.workflow.run_uqpc`, which is the
entry point invoked by ``uq quantify``.

Choosing a Strategy
-------------------

.. list-table:: Strategy Selection Guide
   :header-rows: 1
   :widths: 40 60

   * - If you want to...
     - Use this strategy
   * - Compare with bulk experimental data
     - Uniform (1)
   * - Check simulation convergence
     - By Generation (2)
   * - Quantify stochastic variability
     - By Lineage Seed (3)
   * - Analyze cell cycle dynamics
     - Growth-Stratified (4)
   * - See all four together
     - ``uq quantify`` (runs all 4 by default)

See Also
--------

* :doc:`cell_cycle` — growth-stratified θ details
* :doc:`tutorial_workflow` — mathematical walkthrough
* :doc:`cli_reference` — ``--n-bins`` flag
