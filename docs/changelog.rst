Changelog
=========

All notable changes to the vEcoli UQ Framework.

[0.2.0] - 2026-05-19
-------------------

Major feature release with Q1 baseline, output PCA, SMS-API remote execution,
cross-condition GSA, interactive marimo tutorials, and an interactive HTML report.

Added
^^^^^

**Q1 Baseline & Output PCA** (``uq/`)

* ``compute_variance_budget()`` — ANOVA-style variance budget without input
  perturbation, separating generation, seed, and within-group variance
  (``uq/jupyter.py``, ``uq/pipeline.py``)
* ``--no-perturbation`` flag — run unperturbed simulations for baseline variance
* ``--output-pca K`` flag — PCA on high-dimensional outputs (transcriptome,
  proteome, fluxome) before PCE; per-PC Sobol indices
* PCA scree plots and PC loading tables in the HTML report

**SMS-API Remote Execution** (``uq/remote.py``)

* ``SmsApiClient`` — httpx wrapper for SMS-API REST endpoints
* ``uq sample --api-url`` — run vEcoli on AWS Batch remotely
* ``uq fetch <sim_id>`` — download + parse cd1 TSVs from completed simulations
* CD1 module map: transcriptomics (~4,300), proteomics (~4,300), fluxomics (~2,800),
  exchange fluxes (~165), higher-order properties (~6)
* ALB 502/504 auto-retry logic, tar.gz parsing via ``parse_cd1_tsvs()``

**Cross-Condition GSA** (``uq/multi_condition.py``)

* ``uq sample --conditions id1 --conditions id2`` — multi-parca variant config
* ``MultiConditionResult`` — per-condition Sobol + cross-condition metrics
* ``compute_rank_stability()`` — rank overlap between conditions
* ``find_universal_drivers()`` / ``find_condition_specific()``
* Rich cross-condition comparison table with rank stability (●●●○○) indicators

**Interactive Tutorial System** (``uq/jupyter.py``, ``tutorials/``)

* ``uq tutorial`` CLI command with 10 interactive marimo levels
* ``tutorials/03c_reactive_sensitivity_generalized.py`` — drag-to-explore PCE
* ``tutorials/10_dashboard.py`` — full RFC006 interactive dashboard (marimo)

**HTML Report** (``uq/report.py``)

* Self-contained single-file HTML report (``uq report``)
* Inline SVG Sobol bar charts (no external Plotly dependency)
* Cross-strategy comparison with automatic insight callouts
* Interactive PCE Explorer — client-side JS Legendre evaluation with sliders
* Cell cycle heatmap with viridis colour scale
* Parameter specs with biological role descriptions
* Provenance section (git SHAs, package versions)

**Additional CLI Commands**

* ``uq show-config`` — preview vEcoli workflow config JSON without running
* ``uq compare`` — side-by-side Sobol comparison across UQ experiments
* ``uq export-figures`` — publication-ready PDFs + LaTeX
* ``uq suggest-experiment`` — identify max-uncertainty parameter region
* ``uq init`` — guided project setup wizard

**Dashboard** (``app/``)

* ``uq_daw_simple.py`` — tkinter DAW-style dashboard (``uq dashboard --run-mode tk``)
* ``dashboard_simple.py`` — marimo dashboard (``uq dashboard --run-mode mo``)
* ``web_dashboard.py`` — Flask web dashboard
* Observable selector + per-output coefficients
* Per-stage surrogate evaluation (exact PCE, not linear approx)
* PredictedProfileCanvas — tune params to match desired observable profile
* Heatmap tracking dots for lineage-level visualization

Changed
^^^^^^^

* ``uq/cli.py`` refactored to two-stage CLI (``sample`` + ``quantify``),
  replacing the older single-pipeline design
* Default parameters expanded from 5 to 6 (added
  ``process.translation.basal_elongation_rate``)
* ``uq/growth.py`` — simplified growth-stratified approach replacing the
  older cell cycle variable framework (``CellCycleAggregator`` removed from
  public; see private ``spectral-music-apollo`` branch)
* ``PrecomputedCache`` now stores X, Y, timeseries, germ_train germ_test,
  and generation/seed metadata
* PyTUQ PCE refactored to use ``pytuq.surrogates.pce.PCE`` directly (UQPC
  workflow) with ``lsq``, ``bcs``, and ``anl`` regression backends

Fixed
^^^^^

* ``pce.pcrv.setCfs([pce.lreg.cf])`` call ordering for correct Sobol computation
* HTML report responsive layout for mobile viewing
* SMS-API timeout/retry handling for ALB 502/504 errors

Documentation
^^^^^^^^^^^^^

* ``SAMPLING.md`` — deep-dive on vEcoli variants API delegation
* ``get_started.rst`` — remote execution and cross-condition guides
* ``cli_reference.rst`` — all CLI flags documented
* ``tutorial_workflow.rst`` — full mathematical walkthrough of UQPC pipeline

[0.1.0] - 2026-03-03
--------------------

Initial release implementing RFC006 requirements for Milestone 08.4.2.

Added
^^^^^

**RFC006 Compliance**

* ``RFC006.md`` - Authoritative specification for UQ framework
* ``RFC006_VERIFICATION.md`` - Compliance analysis and implementation status
* ``CONTEXT.md`` - Claude context document referencing RFC006

**Input Parameters** (``uq/inputs.py``)

* ``VioPathwayParams`` - Violacein pathway configuration
* ``MecillinamParams`` - Mecillinam antibiotic conditions
* ``GeneKnockoutParams`` - Gene knockout configuration
* ``UQInputParameters`` - Combined parameter container
* ``InputParameterSpace`` - Parameter space for sensitivity analysis

**Output Extraction** (``uq/outputs.py``)

* ``OutputExtractor`` - Extract outputs from Parquet data
* ``OutputVariables`` - Container for extracted outputs
* ``OutputType`` - Enum of output types (transcriptome, proteome, fluxes, etc.)

**Aggregation Strategies** (``uq/aggregation.py``)

* ``AggregationStrategy`` - Enum with UNIFORM, BY_GENERATION, BY_LINEAGE_SEED, BY_CELL_CYCLE
* ``Aggregator`` - Main aggregation class
* ``AggregatedOutput`` - Container for aggregated statistics
* ``compute_variance_decomposition()`` - Decompose variance by source

**Wrappers** (``uq/wrappers.py``)

* ``SimulationWrapper`` - Run simulations for sensitivity analysis
* ``PrecomputedWrapper`` - Analyze existing results
* ``WrapperConfig`` - Wrapper configuration
* ``create_uqpy_model()`` - UQPy integration
* ``create_pytuq_model()`` - PyTUQ integration

**Sensitivity Analysis** (``uq/sensitivity.py``)

* ``SensitivityAnalyzer`` - Main analysis class
* ``SobolIndices`` - Container for Sobol indices
* ``PCESurrogate`` - PCE surrogate model
* ``run_sensitivity_analysis()`` - Convenience function
* ``analyze_precomputed_results()`` - Analyze existing data

**Cell Cycle Stratification** (``uq/cell_cycle.py``)

* ``CellCycleAggregator`` - Aggregation by cell cycle stage
* ``CellCycleVariable`` - Cell cycle variable container
* ``MassBasedCellCycleVariable`` - Mass-based implementation
* ``DNAReplicationCellCycleVariable`` - DNA-based implementation
* ``CellAngleCellCycleVariable`` - Cell angle implementation
* ``CompositeCellCycleVariable`` - Custom variable support
* ``register_cell_cycle_variable()`` - Register custom variables

Dependencies
^^^^^^^^^^^^

* Added ``PyTUQ`` as primary PCE/Sobol dependency

Documentation
^^^^^^^^^^^^^

* Added comprehensive Sphinx documentation
* Added tutorials for basic sensitivity, variance decomposition, and cell cycle analysis
* Added API reference for all modules
