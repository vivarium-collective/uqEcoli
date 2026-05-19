Tutorial: Full RFC006 Pipeline
==============================

.. caution::
   This tutorial uses the older pipeline API (``XSpaceVecoli``,
   ``execute_pipeline``, ``PipelineResult``) from ``libuq/``, which has been
   superseded by the two-stage CLI.  For the current workflow, see
   :doc:`../getting_started` and :doc:`../cli_reference`.

   Interactive marimo versions of this content are available via
   ``uv run uq tutorial`` or at ``tutorials/07_full_workflow.py``.

For the interactive Marimo notebook version, see ``tutorials/07_full_workflow.py``.

Overview
--------

The RFC006 pipeline has 7 steps that fork into two parallel phases:

.. code-block:: text

   Steps 1-4 (shared):
   Parameter Space → Load Data → Aggregate (strategies 1-3) → Variance Decomposition
                                                                       │
                                                    ┌──────────────────┤
                                                    │                  │
   Phase 1 (bulk):                                  │  Phase 2 (cell cycle):
   Morris → PCE → Sobol                             │  GSA obs → growth θ →
   "Which params drive bulk variance?"               │  Strategy4 → per-stage Sobol
                                                    │  "Which params drive
                                                    │   within-stage variance?"
                                                    └──────────────────────

   Output: PipelineResult(population=UqProfile, cell_cycle=UqProfile)

Prerequisites
-------------

* vEcoli installed with UQ dependencies (``pip install -e ".[uq]"``)
* For real data: simulation output from ``ParquetEmitter``
* For this tutorial: synthetic data is generated automatically

Quick Start: One-Line Pipeline
------------------------------

The simplest way to run the full pipeline is with ``execute_pipeline()``:

.. code-block:: python

   from uq import XSpaceVecoli
   from uq.pipeline.workflow import execute_pipeline
   from uq.synthetic import generate_synthetic_simulation_data

   # 1. Define parameter space
   param_space = XSpaceVecoli(
       include_vio=True,
       include_mecillinam=True,
       vio_expression_bounds=(0.0, 5.0),
       mecillinam_conc_bounds=(0.0, 10.0),
   )

   # 2. Load data (synthetic for demo)
   sim_data = generate_synthetic_simulation_data()

   # 3. Define simulation wrapper
   import numpy as np
   from uq.synthetic import generate_signal

   class MyWrapper:
       def __init__(self, param_names):
           self.param_names = param_names
       def __call__(self, x):
           signal = generate_signal(x, self.param_names, baseline_value=1.5)
           return np.array([signal.mean()])
       def evaluate_batch(self, X):
           return np.vstack([self(x) for x in X])

   wrapper = MyWrapper(param_space.parameter_names)

   # 4. Run the full pipeline
   result = execute_pipeline(
       param_space=param_space,
       simulation_func=wrapper,
       timeseries=sim_data,
       observable_columns=["listeners__mass__dry_mass", "listeners__fba_results__growth"],
       n_bins=10,
       polynomial_order=2,
       n_samples=50,
   )

   # 5. Inspect results
   print(f"Population Sobol: {result.population.sobol_indices[0].parameter_names}")
   print(f"Cell cycle stages: {len(result.cell_cycle.sobol_indices)}")

Step-by-Step Walkthrough
------------------------

Step 1: Define Parameter Space
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   from uq import XSpaceVecoli

   param_space = XSpaceVecoli(
       include_vio=True,
       include_mecillinam=True,
       vio_expression_bounds=(0.0, 5.0),
       vio_trl_eff_bounds=(0.0, 2.0),
       mecillinam_conc_bounds=(0.0, 10.0),
   )
   print(f"Parameters: {param_space.parameter_names}")
   # ['vio_expression', 'vio_trl_eff', 'mecillinam_conc']

Step 2: Load Simulation Data
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   from uq.outputs import OutputExtractor
   from ecoli.library.parquet_emitter import create_duckdb_conn, dataset_sql

   # Real data:
   # conn = create_duckdb_conn()
   # history_sql, config_sql, _ = dataset_sql("/path/to/sims", ["api_simulation_default"])
   # extractor = OutputExtractor(conn, history_sql, config_sql)
   # timeseries = extractor.load_timeseries()

   # Synthetic data:
   from uq.synthetic import generate_synthetic_simulation_data
   timeseries = generate_synthetic_simulation_data()

Step 3: Aggregate (Strategies 1-3)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   from uq.pipeline.workflow import aggregate_timeseries

   observable_columns = ["listeners__mass__dry_mass", "listeners__fba_results__growth"]
   agg_result = aggregate_timeseries(timeseries, observable_columns)

   print(f"Uniform: n={agg_result.uniform.n_samples}")
   print(f"By generation: {len(agg_result.generation.groups)} groups")
   print(f"By seed: {len(agg_result.seed.groups)} groups")

Step 4: Variance Decomposition
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   from uq.pipeline.workflow import get_variance_decomposition
   import numpy as np

   decomp = get_variance_decomposition(agg_result)
   print(f"Generation fraction: {np.mean(decomp['generation_fraction']):.1%}")
   print(f"Seed fraction: {np.mean(decomp['seed_fraction']):.1%}")

Phase 1: Population-Level GSA (Steps 5a-7a)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   from uq.pipeline.workflow import run_phase1

   sobol_bulk, surrogate_bulk = run_phase1(
       param_space=param_space,
       simulation_func=wrapper,
       polynomial_order=2,
       n_samples=50,
   )

   # Interpret results
   for name, si in zip(sobol_bulk.parameter_names, sobol_bulk.total_order.flatten()):
       print(f"  {name}: S_Ti = {si:.4f}")

   # Use surrogate for instant predictions
   prediction = surrogate_bulk.predict(np.array([[2.5, 1.0, 5.0]]))
   print(f"Surrogate prediction: {prediction}")

Phase 2: Cell-Cycle-Stratified GSA (Steps 5b-7b)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   from uq.pipeline.workflow import run_phase2

   per_stage_sobol, surrogate_cc = run_phase2(
       param_space=param_space,
       simulation_func=wrapper,
       agg_result=agg_result,
       observable_names=observable_columns,
       n_bins=10,
       polynomial_order=2,
       n_samples=50,
   )

   # Per-stage sensitivity
   for i, stage_sobol in enumerate(per_stage_sobol):
       top = max(zip(stage_sobol.parameter_names, stage_sobol.total_order.flatten()),
                 key=lambda x: abs(x[1]))
       print(f"  Stage {i} (θ≈{i/len(per_stage_sobol):.1f}): "
             f"dominant param = {top[0]} (S_Ti = {top[1]:.4f})")

Assemble PipelineResult
^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   from uq.pipeline.models import PipelineResult, UqProfile, StratificationLens

   result = PipelineResult(
       population=UqProfile(
           stratification=StratificationLens.POPULATION,
           sobol_indices=[sobol_bulk],
           surrogate=surrogate_bulk,
       ),
       cell_cycle=UqProfile(
           stratification=StratificationLens.CELL_CYCLE,
           sobol_indices=per_stage_sobol,
           surrogate=surrogate_cc,
       ),
   )

Export and Reload
^^^^^^^^^^^^^^^^^

.. code-block:: python

   # Save to disk
   result.export("./uq_results")

   # Reload later
   loaded = PipelineResult.from_export("./uq_results")
   assert len(loaded.population.sobol_indices) == 1
   assert len(loaded.cell_cycle.sobol_indices) == 10

Using Precomputed Samples (Two-Stage Workflow)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

For expensive simulations, pre-generate and cache samples, then analyze without re-running:

.. code-block:: bash

   # Stage 1: generate and cache
   uv run uq generate-samples exp1 exp2 /sims ./cache --n-samples 200

   # Stage 2: analyze from cache
   uv run uq demo --precomputed-path ./cache --export-path ./results

Or programmatically:

.. code-block:: python

   from uq.pipe import pipeline

   result = pipeline(
       experiment_ids=["api_simulation_default", "mecillinam"],
       sim_base_path="/path/to/sims",
       precomputed_path="./cache",
       polynomial_order=3,
       n_bins=10,
       export_path="./results",
   )

When ``precomputed_path`` is provided, Morris screening is skipped (it requires a live
wrapper for OAT trajectories). PCE is fit directly from cached ``(X, Y)`` data by scaling
samples to germ space and building the Legendre basis at user-supplied points.

Running the Example Script
--------------------------

The ``examples/uq_pipeline.py`` script runs the full pipeline with verbose output:

.. code-block:: bash

   uv run python examples/uq_pipeline.py --output-dir ./my_results

This produces:

* ``uq_results.json`` — Human-readable summary
* ``population_surrogate/`` — Phase 1 PCE surrogate
* ``cell_cycle_surrogate/`` — Phase 2 PCE surrogate
* ``population_sobol/`` — Phase 1 Sobol indices
* ``cell_cycle_sobol_stage_N/`` — Phase 2 per-stage Sobol indices
* ``metadata.json`` — Pipeline metadata

See Also
--------

* :doc:`basic_sensitivity` — Phase 1 only (simpler introduction)
* :doc:`variance_decomposition` — Detailed variance decomposition tutorial
* :doc:`cell_cycle_analysis` — Cell cycle variable implementations
* Interactive version: ``uv run marimo run tutorials/07_full_workflow.py``
