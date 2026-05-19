Tutorial: Variance Decomposition
=================================

.. caution::
   This tutorial uses an older API (``Aggregator``, ``compute_variance_decomposition``)
   that has been replaced by the two-stage ``uq sample`` → ``uq quantify`` CLI.
   The current CLI automatically decomposes variance as part of the HTML report.

   For the current workflow, see :doc:`../getting_started`.

Overview
--------

By comparing aggregation strategies, we can determine:

* How much variance is due to **generation effects** (convergence to steady-state)
* How much variance is due to **stochastic seeding** (exogenous noise)
* How much variance is **intrinsic** (within-group variability)

Setup
-----

.. code-block:: python

   from ecoli.library.parquet_emitter import create_duckdb_conn, dataset_sql
   from uq import Aggregator, AggregationStrategy, compute_variance_decomposition
   import numpy as np
   import matplotlib.pyplot as plt

   # Connect to simulation data
   conn = create_duckdb_conn()
   history_sql, config_sql, _ = dataset_sql("./output_dir", ["experiment_id"])

   aggregator = Aggregator(conn, history_sql, config_sql)

Step 1: Aggregate with Different Strategies
-------------------------------------------

.. code-block:: python

   # Strategy 1: Uniform (baseline)
   agg_uniform, ids = aggregator.aggregate_transcriptome(
       AggregationStrategy.UNIFORM,
       generation_lower_bound=2,
   )
   print(f"Uniform: mean shape {agg_uniform.mean.shape}, n_samples={agg_uniform.n_samples}")

   # Strategy 2: By generation
   agg_by_gen, _ = aggregator.aggregate_transcriptome(
       AggregationStrategy.BY_GENERATION,
       generation_lower_bound=2,
   )
   print(f"By generation: mean shape {agg_by_gen.mean.shape}, groups={agg_by_gen.groups}")

   # Strategy 3: By lineage seed
   agg_by_seed, _ = aggregator.aggregate_transcriptome(
       AggregationStrategy.BY_LINEAGE_SEED,
       generation_lower_bound=2,
   )
   print(f"By seed: mean shape {agg_by_seed.mean.shape}, groups={agg_by_seed.groups}")

Step 2: Compute Variance Decomposition
--------------------------------------

.. code-block:: python

   decomposition = compute_variance_decomposition(
       aggregated_by_gen=agg_by_gen,
       aggregated_by_seed=agg_by_seed,
       aggregated_uniform=agg_uniform,
   )

   # Print summary statistics
   print("\n=== Variance Decomposition ===")
   print(f"Total variance (mean): {decomposition['total_variance'].mean():.4f}")
   print(f"Between-generation variance: {decomposition['between_generation_variance'].mean():.4f}")
   print(f"Between-seed variance: {decomposition['between_seed_variance'].mean():.4f}")
   print(f"\nFraction explained by generation: {decomposition['generation_fraction'].mean():.2%}")
   print(f"Fraction explained by seed: {decomposition['seed_fraction'].mean():.2%}")

Step 3: Analyze Per-Gene Variance
---------------------------------

.. code-block:: python

   # Find genes with high generation-dependent variance
   gen_fraction = decomposition['generation_fraction']
   seed_fraction = decomposition['seed_fraction']

   # Top genes by generation effect
   top_gen_idx = np.argsort(gen_fraction)[::-1][:10]
   print("\n=== Top 10 Genes by Generation Effect ===")
   for idx in top_gen_idx:
       print(f"  {ids[idx]}: {gen_fraction[idx]:.2%}")

   # Top genes by seed effect
   top_seed_idx = np.argsort(seed_fraction)[::-1][:10]
   print("\n=== Top 10 Genes by Seed Effect ===")
   for idx in top_seed_idx:
       print(f"  {ids[idx]}: {seed_fraction[idx]:.2%}")

Step 4: Visualize Variance Components
-------------------------------------

.. code-block:: python

   fig, axes = plt.subplots(1, 3, figsize=(15, 5))

   # Plot 1: Variance fractions distribution
   ax = axes[0]
   ax.hist(gen_fraction, bins=50, alpha=0.7, label='Generation')
   ax.hist(seed_fraction, bins=50, alpha=0.7, label='Seed')
   ax.set_xlabel('Fraction of Variance')
   ax.set_ylabel('Number of Genes')
   ax.set_title('Variance Attribution Distribution')
   ax.legend()

   # Plot 2: Generation vs Seed scatter
   ax = axes[1]
   ax.scatter(gen_fraction, seed_fraction, alpha=0.3, s=10)
   ax.set_xlabel('Generation Fraction')
   ax.set_ylabel('Seed Fraction')
   ax.set_title('Generation vs Seed Effects')
   ax.plot([0, 1], [1, 0], 'r--', alpha=0.5)  # Reference line

   # Plot 3: Cumulative variance explained
   ax = axes[2]
   total_var = decomposition['total_variance']
   sorted_idx = np.argsort(total_var)[::-1]
   cumsum = np.cumsum(total_var[sorted_idx]) / np.sum(total_var)
   ax.plot(np.arange(len(cumsum)) / len(cumsum), cumsum)
   ax.set_xlabel('Fraction of Genes')
   ax.set_ylabel('Cumulative Variance Explained')
   ax.set_title('Variance Concentration')

   plt.tight_layout()
   plt.savefig('variance_decomposition.png', dpi=150)
   plt.show()

Step 5: Compare Output Types
----------------------------

.. code-block:: python

   output_types = {
       'Transcriptome': aggregator.aggregate_transcriptome,
       'Proteome': aggregator.aggregate_proteome,
       'Exchange Fluxes': lambda s, **kw: aggregator.aggregate_fluxes(s, exchange_only=True, **kw),
   }

   results = {}
   for name, agg_func in output_types.items():
       uniform, _ = agg_func(AggregationStrategy.UNIFORM, generation_lower_bound=2)
       by_gen, _ = agg_func(AggregationStrategy.BY_GENERATION, generation_lower_bound=2)
       by_seed, _ = agg_func(AggregationStrategy.BY_LINEAGE_SEED, generation_lower_bound=2)

       decomp = compute_variance_decomposition(by_gen, by_seed, uniform)
       results[name] = {
           'generation': decomp['generation_fraction'].mean(),
           'seed': decomp['seed_fraction'].mean(),
       }

   # Plot comparison
   fig, ax = plt.subplots(figsize=(10, 6))

   names = list(results.keys())
   gen_fracs = [results[n]['generation'] for n in names]
   seed_fracs = [results[n]['seed'] for n in names]

   x = np.arange(len(names))
   width = 0.35

   ax.bar(x - width/2, gen_fracs, width, label='Generation Effect')
   ax.bar(x + width/2, seed_fracs, width, label='Seed Effect')
   ax.set_ylabel('Fraction of Variance')
   ax.set_title('Variance Decomposition by Output Type')
   ax.set_xticks(x)
   ax.set_xticklabels(names)
   ax.legend()

   plt.tight_layout()
   plt.savefig('variance_by_output_type.png', dpi=150)

Interpretation Guide
--------------------

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Pattern
     - Interpretation
   * - High generation fraction
     - Output changes with cell age; may indicate non-steady-state
   * - High seed fraction
     - Output sensitive to initial conditions; high stochasticity
   * - Low both fractions
     - Output is robust; variance is intrinsic (within-cell)
   * - Gen + Seed > 1
     - Strong correlations between effects

Next Steps
----------

* Investigate specific genes with high variance fractions
* Use generation filtering to ensure steady-state
* Increase number of lineage seeds if seed variance is high
