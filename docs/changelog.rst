Changelog
=========

All notable changes to the vEcoli UQ Framework.

[Unreleased]
------------

BYO-variants escape hatch + ``uq/vecoli_config.py`` refactor (feature branch:
``feat/v2ecoli-integration``).

Added
^^^^^

**BYO variants — ``--variants-source base-config``** (``uq/cli.py``,
``uq/vecoli_config.py``)

* New ``--variants-source {params-file, base-config}`` flag on ``uq sample``.
  Default ``params-file`` preserves the standard PCRV-driven workflow.
* ``base-config`` mode skips PCRV sampling entirely and reverse-maps the
  user's ``sim_data_setattr`` mutation list back into ``X`` so ``uq quantify``
  regresses ``Y`` against the perturbations vEcoli actually applied.
* ``_x_from_variants()`` — inverse of ``_build_variants_from_samples``;
  errors clearly on non-``sim_data_setattr`` modules, missing ``attr_path``s,
  and index mismatches.
* ``_germ_from_physical()`` — affine inverse of the input PCRV map.
* ``_load_variants_from_base_config()`` — extracts the ``variants`` block
  from a vEcoli config JSON; fails fast if absent.
* Constraints: local ``--backend vecoli`` only; ``--backend v2ecoli`` and
  ``--api-url`` reject BYO mode with a clear error.  ``--n-test`` ignored
  in BYO mode (no validation surface without PCRV draws).

**PCE sample-size adequacy diagnostic** (``uq/vecoli_config.py``)

* ``_pce_basis_size(order, n_params)`` — total-order basis count
  :math:`\binom{p+d}{d}`, matches PyTUQ's ``get_mi`` row count.
* ``_pce_sample_adequacy(n_samples, n_params, order=2)`` returns
  ``"ok" | "marginal" | "underdetermined"`` plus an advisory string with
  concrete recommendations (add samples, drop ``--polynomial-order``, or
  switch to ``--regression bcs``).
* Surfaced in red / yellow / dim in the ``uq sample`` output under BYO
  mode, where ``n_samples`` is forced by the user's variants count and
  the diagnostic catches under-determined PCE fits before quantify.

**Override guard in ``_build_config``** (``uq/vecoli_config.py``)

* If ``--base-config`` JSON already declares a top-level ``variants`` key,
  it is preserved verbatim instead of being clobbered by the auto-generated
  ``sim_data_setattr`` block.  Strictly additive — default ``--params-file``
  behavior is unchanged when no user ``variants`` block is present.

Changed
^^^^^^^

**Module relocation: ``uq/tui.py`` → ``uq/vecoli_config.py``**

* The vEcoli workflow-config helpers (``_build_config``,
  ``_build_variants_from_samples``, ``_collect_variant_timeseries``,
  ``_count_completed_variants``, ``_get_vecoli_root``) moved out of
  ``uq/tui.py`` into the new CLI-owned module ``uq/vecoli_config.py``.
* Restores correct dependency direction: ``cli.py``, ``tui.py``,
  ``processes.py``, and ``bigraph/processes.py`` all import from
  ``vecoli_config``.  No reverse imports.
* Importers updated; no re-export shim in ``tui.py``.  New code should
  import from ``uq.vecoli_config`` directly.

Tests
^^^^^

* ``tests/test_byo_variants.py`` — 16 tests covering: ``_build_config``
  override guard (with and without user ``variants``, any variant module),
  ``_x_from_variants`` round-trip for scalar and indexed mutations,
  rejection of non-``sim_data_setattr`` modules and missing ``attr_path``s,
  ``_germ_from_physical`` correctness, ``_load_variants_from_base_config``
  error paths, ``_pce_basis_size`` formula, and adequacy thresholds across
  ``underdetermined`` / ``marginal`` / ``ok`` bands at varying orders.

Documentation
^^^^^^^^^^^^^

* ``README.md`` — BYO variants section under Getting Started.
* ``docs/cli_reference.rst`` — ``--variants-source`` flag with the BYO
  example and the PCE adequacy diagnostic output.
* ``docs/architecture.rst`` — ``uq/vecoli_config.py`` added to the module
  map with the new dependency-direction diagram.
* ``docs/getting_started.rst`` — Bring-your-own-variants subsection.
* ``docs/tutorial_workflow.rst`` — BYO sub-step under Step 3 with the
  affine-inverse derivation.
* ``SAMPLING.md`` — stale ``uq/tui.py`` references updated to
  ``uq/vecoli_config.py``.

[0.3.0] - 2026-05-19
--------------------

v2ecoli backend migration (in-process, process-bigraph composites).

Added
^^^^^

**v2ecoli in-process backend** (``uq/generators/v2ecoli.py``)

* ``V2ecoliGenerator`` — drop-in replacement for ``TimeseriesGeneratorVecoli``
  using in-process process-bigraph composites (no subprocess, no Nextflow)
* ``generate_cache_bundle()`` — wraps ``v2ecoli.core.save_cache()``
* ``build_v2ecoli_composite()`` — builds v2ecoli baseline composite from
  cache bundle with optional dot-path mutations
* ``run_v2ecoli_composite()`` — tick-loop with division detection and
  observable extraction (mass, higher_order, transcriptome, proteome,
  fluxome, exchange_fluxes)
* ``_apply_mutation_to_configs()`` — maps UQ dot-path mutations to
  v2ecoli process config keys (metabolism, transcription, translation,
  mass)
* Multiprocessing via ``ProcessPoolExecutor`` (``_run_batch_parallel``)

**Observable bridge** (``uq/v2ecoli_bridge.py``)

* ``OBSERVABLE_PATHS`` — scalar state-path map (mass scalars, growth,
  division)
* ``ARRAY_OBSERVABLE_PATHS`` — array state-path map (transcriptome,
  proteome, fluxome, exchange_fluxes)
* ``get_array_observable_from_state()`` — extracts numpy arrays from
  composite state, flattened to indexed columns
* ``PRESET_MAP`` now covers all 6 presets: mass, higher_order,
  transcriptome, proteome, fluxome, exchange_fluxes

**CLI** (``uq/cli.py``)

* ``--backend {vecoli,v2ecoli}`` flag on ``uq sample``
* ``--max-workers N`` for v2ecoli parallel execution
* ``_sample_v2ecoli()`` — end-to-end v2ecoli sampling path

**Remote execution** (``uq/remote.py``)

* ``SmsApiClient.submit_v2ecoli()`` — submits cache bundle + mutations
  to SMS-API as multipart upload

**Process-bigraph integration** (``pbg_uqEcoli/``)

* ``UQPipeline(Process)`` — wraps the full UQ pipeline as a
  process-bigraph Process with v2ecoli backend support
* ``pipeline_document()`` — composite document generator
* ``build_core()`` — core builder with ECOLI_TYPES registration

Changed
^^^^^^^

* ``libuq/generators/vecoli.py`` — ``TimeseriesGeneratorVecoli`` marked as
  deprecated (use ``V2ecoliGenerator`` with ``--backend v2ecoli`` instead)
* ``libuq/wrappers.py`` — ``SimulationWrapper`` marked as deprecated
* ``SAMPLING.md`` — added Section 7 documenting the v2ecoli backend
  architecture, key source files, and usage

Tests
^^^^^

* ``tests/test_v2ecoli_generator.py`` — 31 unit tests covering mutation
  logic, bundle deep-copy, V2ecoliGenerator construction,
  extract_timeseries, observable path mapping, array extraction,
  and import chains
* ``tests/test_backend_parity.py`` — regression test stubs for
  single-sample, Sobol, and aggregation metadata parity
  (requires ``simData.cPickle``)

[0.2.1] - 2026-05-19
--------------------

Hotfix and demo site release.

Added
^^^^^

**Demo Site** (``site/``)

* ``site/demo/`` — GH Pages demo directory at ``/uqEcoli/demo/``
* Landing page with card grid (active + greyed-out placeholder cards)
* Interactive CLI cheatsheet (``cli_cheatsheet.html``)
* Pipeline architecture section with collapsible SVG diagram inline in report
* "How to Read Sobol Indices" explainer box with plain-English definitions
* CLI command reference table (12 subcommands) embedded in report
* Sphinx docs + demo site links in provenance section
* Sticky nav bar with jump-to anchors on all major report sections
* WASM-exported marimo notebook (``reactive.wasm.html``) for client-side Pyodide
* README placeholders for report screenshots, marimo gallery, TUI, and cross-condition GSA

**GH Pages Deployment**

* ``.github/workflows/static-report.yml`` now deploys ``./site`` instead of ``./reports``
* ``.nojekyll`` marker and root redirect (``/`` → ``/demo/``)

Fixed
^^^^^

* PCE profile chart Y-axis: replaced ``toFixed(2)`` with adaptive ``fmtTick()`` showing
  full precision for small values (was showing ``0.00`` for near-zero observables)
* Added ``Predicted Value`` Y-axis label to match X-axis ``Cell Cycle Progress (θ)``
* Increased left chart margin from 60 to 70 for longer tick labels

[0.2.0] - 2026-05-19
---------------------

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
