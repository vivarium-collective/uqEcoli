Tutorial: Cell Cycle Analysis
==============================

.. caution::
   This tutorial uses the older ``CellCycleAggregator`` framework which has
   been removed from the public codebase.  The current approach (Strategy 4)
   uses a simpler growth-stratified θ variable described in :doc:`../cell_cycle`.

   For the current workflow, see :doc:`../cli_reference` and :doc:`../cell_cycle`.

Overview
--------

Cell cycle stratification enables:

* Analyzing how outputs vary within a cell's lifespan
* Identifying cell cycle-dependent gene expression
* Understanding temporal dynamics of metabolism

Setup
-----

.. code-block:: python

   from ecoli.library.parquet_emitter import create_duckdb_conn, dataset_sql
   from uq import CellCycleAggregator
   import numpy as np
   import matplotlib.pyplot as plt

   # Connect to simulation data
   conn = create_duckdb_conn()
   history_sql, config_sql, _ = dataset_sql("./output_dir", ["experiment_id"])

Step 1: Create Cell Cycle Aggregator
------------------------------------

.. code-block:: python

   # Create aggregator with mass-based cell cycle variable
   cc_agg = CellCycleAggregator(
       conn=conn,
       history_sql=history_sql,
       config_sql=config_sql,
       variable_type="mass_based",  # Options: mass_based, dna_replication, cell_angle
       n_stages=10,                  # Number of cell cycle bins
   )

   print(f"Cell cycle variable: {cc_agg.variable_computer.name}")
   print(f"Number of stages: {cc_agg.n_stages}")

Step 2: Compute Cell Cycle Variable
-----------------------------------

.. code-block:: python

   # Get raw cell cycle values
   cc_variable = cc_agg.compute_cell_cycle_variable(
       generation_lower_bound=2,  # Skip initial generations
   )

   print(f"Number of data points: {len(cc_variable.values)}")
   print(f"Value range: [{cc_variable.values.min():.3f}, {cc_variable.values.max():.3f}]")
   print(f"Normalized: {cc_variable.normalized}")

   # Visualize distribution
   plt.figure(figsize=(8, 5))
   plt.hist(cc_variable.values, bins=50, edgecolor='black')
   plt.xlabel('Cell Cycle Variable')
   plt.ylabel('Count')
   plt.title('Distribution of Cell Cycle Variable')
   plt.savefig('cell_cycle_distribution.png', dpi=150)

Step 3: Get Cell Cycle Profiles
-------------------------------

.. code-block:: python

   # Dry mass profile across cell cycle
   mass_profile = cc_agg.get_cell_cycle_profile(
       "listeners__mass__dry_mass",
       generation_lower_bound=2,
   )

   print(f"Stages: {mass_profile['stage']}")
   print(f"Mean shape: {mass_profile['mean'].shape}")

   # Plot profile
   plt.figure(figsize=(10, 6))
   stages = mass_profile['stage']
   means = mass_profile['mean'].flatten()
   stds = mass_profile['std'].flatten()

   plt.fill_between(stages, means - stds, means + stds, alpha=0.3)
   plt.plot(stages, means, 'o-', linewidth=2, markersize=8)
   plt.xlabel('Cell Cycle Stage')
   plt.ylabel('Dry Mass (fg)')
   plt.title('Dry Mass Across Cell Cycle')
   plt.grid(True, alpha=0.3)
   plt.savefig('dry_mass_cell_cycle.png', dpi=150)

Step 4: Analyze Multiple Outputs
--------------------------------

.. code-block:: python

   outputs = {
       'Dry Mass': 'listeners__mass__dry_mass',
       'Cell Volume': 'listeners__mass__volume',
       'DNA Mass': 'listeners__mass__dna_mass',
   }

   fig, axes = plt.subplots(1, 3, figsize=(15, 5))

   for ax, (name, col) in zip(axes, outputs.items()):
       profile = cc_agg.get_cell_cycle_profile(col, generation_lower_bound=2)
       stages = profile['stage']
       means = profile['mean'].flatten()
       stds = profile['std'].flatten()

       ax.fill_between(stages, means - stds, means + stds, alpha=0.3)
       ax.plot(stages, means, 'o-', linewidth=2)
       ax.set_xlabel('Cell Cycle Stage')
       ax.set_ylabel(name)
       ax.set_title(f'{name} Profile')

   plt.tight_layout()
   plt.savefig('multiple_cell_cycle_profiles.png', dpi=150)

Step 5: Compare Cell Cycle Variables
------------------------------------

.. code-block:: python

   variable_types = ['mass_based', 'dna_replication', 'cell_angle']

   fig, axes = plt.subplots(1, 3, figsize=(15, 5))

   for ax, var_type in zip(axes, variable_types):
       cc_agg = CellCycleAggregator(
           conn, history_sql, config_sql,
           variable_type=var_type,
           n_stages=10,
       )

       profile = cc_agg.get_cell_cycle_profile(
           "listeners__mass__dry_mass",
           generation_lower_bound=2,
       )

       stages = profile['stage']
       means = profile['mean'].flatten()
       stds = profile['std'].flatten()

       ax.errorbar(stages, means, yerr=stds, fmt='o-', capsize=3)
       ax.set_xlabel('Cell Cycle Stage')
       ax.set_ylabel('Dry Mass')
       ax.set_title(f'Variable: {var_type}')

   plt.tight_layout()
   plt.savefig('compare_cell_cycle_variables.png', dpi=150)

Step 6: Transcriptome Cell Cycle Dynamics
-----------------------------------------

.. code-block:: python

   # Get transcriptome profile
   from ecoli.library.parquet_emitter import field_metadata

   mrna_col = "listeners__rna_counts__mRNA_cistron_counts"
   mrna_ids = field_metadata(conn, config_sql, mrna_col)

   cc_agg = CellCycleAggregator(
       conn, history_sql, config_sql,
       variable_type="mass_based",
       n_stages=10,
   )

   profile = cc_agg.get_cell_cycle_profile(mrna_col, generation_lower_bound=2)
   means = profile['mean']  # Shape: (n_stages, n_genes)

   # Find genes with highest cell cycle variation
   gene_variation = np.std(means, axis=0)  # Std across stages
   top_genes_idx = np.argsort(gene_variation)[::-1][:5]

   print("=== Top 5 Cell Cycle-Dependent Genes ===")
   for idx in top_genes_idx:
       print(f"  {mrna_ids[idx]}: variation = {gene_variation[idx]:.2f}")

   # Plot top genes
   fig, ax = plt.subplots(figsize=(10, 6))
   stages = profile['stage']

   for idx in top_genes_idx[:5]:
       ax.plot(stages, means[:, idx], 'o-', label=mrna_ids[idx][:20])

   ax.set_xlabel('Cell Cycle Stage')
   ax.set_ylabel('mRNA Count')
   ax.set_title('Cell Cycle-Dependent Gene Expression')
   ax.legend(loc='upper left', fontsize=8)
   plt.tight_layout()
   plt.savefig('cell_cycle_genes.png', dpi=150)

Step 7: Custom Cell Cycle Variable
----------------------------------

.. code-block:: python

   from uq import CompositeCellCycleVariable, register_cell_cycle_variable

   def compute_dna_to_mass_ratio(data):
       """Cell cycle variable based on DNA/mass ratio."""
       dna = data["listeners__mass__dna_mass"].to_numpy()
       mass = data["listeners__mass__dry_mass"].to_numpy()
       ratio = dna / np.maximum(mass, 1e-10)
       return (ratio - ratio.min()) / (ratio.max() - ratio.min() + 1e-10)

   custom_var = CompositeCellCycleVariable(
       name="dna_mass_ratio",
       required_columns=[
           "listeners__mass__dna_mass",
           "listeners__mass__dry_mass",
           "generation",
           "agent_id",
           "time",
       ],
       compute_func=compute_dna_to_mass_ratio,
   )

   register_cell_cycle_variable("dna_mass_ratio", custom_var)

   # Use the custom variable
   cc_agg = CellCycleAggregator(
       conn, history_sql, config_sql,
       variable_type="dna_mass_ratio",
       n_stages=10,
   )

   profile = cc_agg.get_cell_cycle_profile(
       "listeners__mass__dry_mass",
       generation_lower_bound=2,
   )

Key Takeaways
-------------

1. **Choose appropriate variable**: Mass-based is robust; DNA-based captures replication events
2. **Use enough stages**: 5-10 stages balances resolution with statistics
3. **Filter transients**: Skip initial generations for steady-state analysis
4. **Compare variables**: Different definitions may reveal different dynamics
5. **Identify key genes**: Focus on genes with high cell cycle variation

Next Steps
----------

* Integrate cell cycle profiles into sensitivity analysis
* Compare profiles across different experimental conditions
* Correlate with experimental single-cell data
