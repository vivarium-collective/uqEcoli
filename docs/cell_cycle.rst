Growth-Stratified Sensitivity Analysis
=======================================

``uq quantify`` implements **Strategy 4** of the RFC006 pipeline: stratifying
simulation data by growth progress (θ, "theta") to reveal whether parameter
importance changes as a cell grows from birth toward division.

Biological Background
---------------------

*E. coli* cells grow approximately exponentially between birth and division
(Schaechter et al., 1958; Taheri-Araghi et al., 2015). In the vEcoli model
(Macklin et al., 2020; Ahn-Horst et al., 2022), dry mass is a primary
integrated output reflecting the net balance of biosynthetic processes and
is directly comparable to experimental measurements (OD, buoyant mass).

The Growth-Progress Variable θ
------------------------------

The growth progress variable is defined as the normalized log-mass ratio:

.. math::

   \theta(t) = \frac{\log M(t) - \log M_{\min}}{\log M_{\max} - \log M_{\min}}

where :math:`M(t)` is the dry mass at time :math:`t`, and the min/max are
taken over each cell's observation window.  θ ranges monotonically from 0
(birth / start of window) to 1 (division / end of window).

This is **not** a full cell cycle variable — it does not identify B/C/D-period
phases or make claims about replication timing.  It simply bins the growth
trajectory by mass accumulation so that sensitivity analysis can reveal
whether parameter importance changes as the cell enlarges.

Implementation
--------------

The computation lives in ``uq/growth.py``:

.. code-block:: python

   from uq.growth import compute_growth_fraction, bin_by_growth_stage
   import numpy as np

   # timeseries: (n_timesteps, n_observables), column 0 = dry_mass
   theta = compute_growth_fraction(timeseries)
   # θ in [0, 1]

   # Bin into 10 uniform stages
   stage_labels, binned = bin_by_growth_stage(theta, n_bins=10)

The :py:func:`~uq.workflow.run_strategy4_growth_stratified` function in
``uq/workflow.py`` applies this to every sample in the cache, then fits an
independent PCE + Sobol decomposition per stage.

Interpreting the Heatmap
------------------------

The HTML report and dashboard render per-stage :math:`S_{Ti}` as a
**parameter × stage heatmap** (viridis colour scale).  A parameter whose
bar darkens from left to right matters more late in the cell cycle;
lightening means it matters more early.

Typical patterns seen with the 6 default parameters:

* **dry_mass_frac** — dominance decreases from θ≈0.29 to θ≈0.18
  (composition matters most at birth)
* **basal_elongation_rate** — dominance increases from θ≈0.26 to θ≈0.30
  (ribosome speed matters more as the cell grows)
* **kinetic_objective_weight** — dominance increases sharply near division

``n_bins`` defaults to 10 (configurable via ``--n-bins``).

Comparison with Earlier Approaches
----------------------------------

Earlier iterations of this codebase included a ``CellCycleAggregator`` class
with multiple cell cycle variable implementations (mass-based, DNA-based,
cell angle).  That framework has been removed from the public codebase in
favour of the simpler growth-stratified approach above.  See the private
``spectral-music-apollo`` branch for the original cell cycle variable
implementation.

API Reference
-------------

* :py:func:`uq.growth.compute_growth_fraction` — compute θ from dry mass
* :py:func:`uq.growth.bin_by_growth_stage` — discretize θ into stage bins
* :py:func:`uq.workflow.run_strategy4_growth_stratified` — full per-stage
  PCE + Sobol pipeline

See Also
--------

* :doc:`cli_reference` — ``--n-bins`` flag
* :doc:`aggregation_strategies` — overview of all four RFC006 strategies
