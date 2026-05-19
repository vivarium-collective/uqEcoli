Tutorial: Basic Sensitivity Analysis
=====================================

.. caution::
   This tutorial uses an older API (``SensitivityAnalyzer``, ``InputParameterSpace``)
   that has been replaced by the two-stage ``uq sample`` → ``uq quantify`` CLI
   described in :doc:`../getting_started`.  For the current workflow, use
   :doc:`../cli_reference` and :doc:`../tutorial_workflow`.

   For an interactive marimo version of this content, run ``uv run uq tutorial``.

Prerequisites
-------------

* vEcoli installed with UQ dependencies (``pip install -e ".[uq]"``)
* Simulation data from ParquetEmitter
* A ``sim_data`` pickle file

Overview
--------

We will:

1. Define the input parameter space
2. Configure the simulation wrapper
3. Run PCE-based sensitivity analysis
4. Interpret and visualize results

Step 1: Define Input Parameters
-------------------------------

First, define which parameters we want to analyze:

.. code-block:: python

   from uq import InputParameterSpace

   # Create parameter space
   param_space = InputParameterSpace(
       # Include violacein pathway parameters
       include_vio=True,
       vio_expression_bounds=(0.0, 5.0),      # Expression factor range
       vio_trl_eff_bounds=(0.0, 2.0),         # Translation efficiency range

       # Include mecillinam antibiotic
       include_mecillinam=True,
       mecillinam_conc_bounds=(0.0, 10.0),    # Concentration range (mM)
   )

   print(f"Number of parameters: {param_space.n_parameters}")
   print(f"Parameter names: {param_space.parameter_names}")
   print(f"Parameter bounds:\n{param_space.bounds_array}")

Expected output::

   Number of parameters: 3
   Parameter names: ['vio_expression', 'vio_trl_eff', 'mecillinam_concentration']
   Parameter bounds:
   [[ 0.  5.]
    [ 0.  2.]
    [ 0. 10.]]

Step 2: Configure the Wrapper
-----------------------------

Set up the simulation wrapper with desired aggregation and outputs:

.. code-block:: python

   from uq import WrapperConfig, SimulationWrapper, AggregationStrategy, OutputType

   config = WrapperConfig(
       # Path to sim_data pickle
       sim_data_path="./data/sim_data.cPickle",

       # Output directories
       output_dir="./uq_analysis/simulations",
       cache_dir="./uq_analysis/cache",

       # Simulation settings
       generations=8,

       # Aggregation strategy
       aggregation_strategy=AggregationStrategy.UNIFORM,

       # Which outputs to analyze
       output_types=[
           OutputType.EXCHANGE_FLUXES,
           OutputType.HIGHER_ORDER_PROPERTIES,
       ],

       # Filter out initial transient
       generation_lower_bound=2,
       time_lower_bound=100.0,

       # Enable caching
       use_cache=True,
   )

   wrapper = SimulationWrapper(config, param_space)
   print(f"Output dimension: {wrapper.get_output_dimension()}")

Step 3: Run Sensitivity Analysis
--------------------------------

Create the analyzer and run PCE-based analysis:

.. code-block:: python

   from uq import SensitivityAnalyzer

   analyzer = SensitivityAnalyzer(param_space, wrapper)

   # Run PCE analysis
   sobol_indices, pce_surrogate = analyzer.analyze_with_pce(
       polynomial_order=3,  # Polynomial degree
       n_samples=100,       # Number of training samples
       use_uqpy=True,       # Use UQPy library
   )

   print("Analysis complete!")
   print(f"First-order indices: {sobol_indices.first_order}")
   print(f"Total-order indices: {sobol_indices.total_order}")

Step 4: Interpret Results
-------------------------

Identify the most influential parameters:

.. code-block:: python

   # Get ranked parameters
   print("\n=== Most Influential Parameters ===")
   for name, value in sobol_indices.get_most_influential(n=3, index_type="total"):
       print(f"  {name}: {value:.4f}")

   # Check for interactions
   print("\n=== Interaction Effects ===")
   for i, name in enumerate(param_space.parameter_names):
       s1 = sobol_indices.first_order[i]
       st = sobol_indices.total_order[i]
       interaction = st - s1
       print(f"  {name}: first={s1:.4f}, total={st:.4f}, interaction={interaction:.4f}")

Step 5: Visualize Results
-------------------------

Create a bar plot of sensitivity indices:

.. code-block:: python

   import matplotlib.pyplot as plt
   import numpy as np

   names = param_space.parameter_names
   s1 = sobol_indices.first_order
   st = sobol_indices.total_order

   # Handle multi-output case
   if s1.ndim > 1:
       s1 = np.mean(s1, axis=0)
       st = np.mean(st, axis=0)

   x = np.arange(len(names))
   width = 0.35

   fig, ax = plt.subplots(figsize=(10, 6))
   bars1 = ax.bar(x - width/2, s1, width, label='First-order (main effect)')
   bars2 = ax.bar(x + width/2, st, width, label='Total-order (with interactions)')

   ax.set_ylabel('Sobol Index')
   ax.set_title('Parameter Sensitivity Analysis')
   ax.set_xticks(x)
   ax.set_xticklabels(names, rotation=45, ha='right')
   ax.legend()
   ax.set_ylim(0, 1)

   # Add value labels
   for bar in bars1:
       height = bar.get_height()
       ax.annotate(f'{height:.2f}',
                   xy=(bar.get_x() + bar.get_width() / 2, height),
                   xytext=(0, 3), textcoords="offset points",
                   ha='center', va='bottom', fontsize=8)

   plt.tight_layout()
   plt.savefig('sensitivity_analysis.png', dpi=150)
   plt.show()

Complete Script
---------------

Here's the complete analysis script:

.. code-block:: python

   """
   Basic sensitivity analysis with vEcoli UQ framework.
   """
   import numpy as np
   import matplotlib.pyplot as plt

   from uq import (
       InputParameterSpace,
       WrapperConfig,
       SimulationWrapper,
       SensitivityAnalyzer,
       AggregationStrategy,
       OutputType,
   )

   # 1. Define parameter space
   param_space = InputParameterSpace(
       include_vio=True,
       include_mecillinam=True,
       vio_expression_bounds=(0.0, 5.0),
       mecillinam_conc_bounds=(0.0, 10.0),
   )

   # 2. Configure wrapper
   config = WrapperConfig(
       sim_data_path="./data/sim_data.cPickle",
       output_dir="./uq_analysis/simulations",
       cache_dir="./uq_analysis/cache",
       aggregation_strategy=AggregationStrategy.UNIFORM,
       output_types=[OutputType.EXCHANGE_FLUXES],
       generation_lower_bound=2,
   )

   wrapper = SimulationWrapper(config, param_space)

   # 3. Run analysis
   analyzer = SensitivityAnalyzer(param_space, wrapper)
   sobol, pce = analyzer.analyze_with_pce(polynomial_order=3, n_samples=100)

   # 4. Print results
   print("=== Sensitivity Analysis Results ===")
   print(f"Parameters: {param_space.parameter_names}")
   print(f"First-order indices: {sobol.first_order}")
   print(f"Total-order indices: {sobol.total_order}")
   print("\nMost influential:")
   for name, val in sobol.get_most_influential(3):
       print(f"  {name}: {val:.4f}")

   # 5. Visualize
   names = param_space.parameter_names
   s1 = sobol.first_order if sobol.first_order.ndim == 1 else np.mean(sobol.first_order, axis=0)
   st = sobol.total_order if sobol.total_order.ndim == 1 else np.mean(sobol.total_order, axis=0)

   fig, ax = plt.subplots(figsize=(8, 5))
   x = np.arange(len(names))
   ax.bar(x - 0.2, s1, 0.4, label='First-order')
   ax.bar(x + 0.2, st, 0.4, label='Total-order')
   ax.set_xticks(x)
   ax.set_xticklabels(names, rotation=45, ha='right')
   ax.set_ylabel('Sobol Index')
   ax.legend()
   plt.tight_layout()
   plt.savefig('sensitivity_results.png', dpi=150)

Next Steps
----------

* Try different :doc:`../aggregation_strategies`
* Explore :doc:`variance_decomposition`
* Analyze :doc:`cell_cycle_analysis`
