Design Document
===============

This page documents the design rationale and implementation status of the
vEcoli UQ Framework.

.. note::
   This package implements **RFC006**, the authoritative specification for
   uncertainty quantification in vEcoli.

   For compliance status, see ``readmes/RFC006-VERIFICATION.md``.

Vision
------

The UQ framework satisfies **Milestone 08.4.2**: Implement uncertainty quantification
framework to track prediction confidence, as specified in RFC006.

Key goals:

1. Characterize different types of uncertainty
2. Map single-cell to bulk simulations
3. Enable population-level perturbation analysis (Milestone 10)
4. Support CD2 evaluation requirements

RFC006 Compliance
-----------------

The package implements all requirements specified in RFC006:

**Phase 1 (MS-08.4.2)**

.. list-table::
   :header-rows: 1
   :widths: 10 50 20 20

   * - #
     - Activity
     - Status
     - Module
   * - 1
     - Identify input/output variables
     - Complete
     - ``uq/inputs.py`` → ``uq/workflow.py``
   * - 2
     - Enable output via emitter
     - Complete (ParquetEmitter)
     - ``uq/observables.py``
   * - 3
     - Implement sample + evaluate workflow
     - Complete
     - ``uq/workflow.py``
   * - 4
     - Implement sensitivity analysis (PCE + Sobol)
     - Complete
     - ``uq/workflow.py``
   * - 5
     - Apply to representative simulations
     - Complete — demonstrated with HTML report + interactive dashboard
     - ``uq/report.py``, ``app/``

**Phase 2 (CD2/Milestone 10)**

.. list-table::
   :header-rows: 1
   :widths: 10 50 20 20

   * - #
     - Activity
     - Status
     - Module
   * - 6
     - Growth-stratified aggregation (Strategy 4)
     - Complete
     - ``uq/growth.py``
   * - 7
     - Per-stage PCE/Sobol across cell cycle θ
     - Complete — Strategy 4 wrapper shows θ-dependent parameter importance
     - ``uq/workflow.py`` (``run_strategy4_growth_stratified``)

Four Aggregation Strategies
---------------------------

As specified in RFC006 Section 1:

1. **Uniform**: Across all simulated cells and times (baseline)
2. **By Generation**: Stratified by cell generation
3. **By Lineage Seed**: Stratified by stochastic seed
4. **By Cell Cycle**: Stratified by cell cycle stage

GSA → Cell Cycle Feedback Loop
-------------------------------

Variance decomposition from strategies 1–3 (Uniform, By Generation, By Lineage
Seed) identifies observables with high residual variance — variance not explained
by input parameters — which are candidates for cell-cycle-related dynamics.

Architecture
------------

The implementation parametrizes three steps (RFC006 Section 4):

A. **Selection/Extraction**: ``outputs.py`` - Extract variables from simulation trajectories
B. **Temporal Aggregation**: ``aggregation.py`` - Aggregate into output variables Y
C. **Sensitivity Analysis**: ``sensitivity.py`` - Apply PCE-based methods

Libraries
---------

* **PyTUQ**: Primary library for PCE and Sobol analysis (Sandia National Labs)

Key Documents
-------------

* ``SAMPLING.md`` - How ``uq sample`` delegates to PyTUQ + vEcoli

Two-Stage Workflow
------------------

The ``uq`` CLI implements the RFC006 pipeline in two stages:

.. code-block:: bash

   # Stage 1: sample + evaluate
   uv run uq sample /path/to/simData.cPickle --n-samples 50 --cache-dir ./cache

   # Stage 2: quantify
   uv run uq quantify /path/to/simData.cPickle --cache-dir ./cache --export-path ./results

   # HTML report
   uv run uq report --results-path ./results

Phase 1 (strategies 1–3): PCE surrogate → Sobol indices
  ("Which parameters drive bulk output variance?")
Phase 2 (strategy 4): Growth-stratified θ → per-stage PCE → per-stage Sobol
  ("Which parameters drive variance WITHIN each cell cycle stage?")

See :doc:`cli_reference` for full flag documentation.
