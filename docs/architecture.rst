Architecture
============

The diagram below shows the full RFC006 UQ pipeline from parameter
definition through to exported artifacts.  Click to zoom.

.. image:: rfc006_workflow.svg
   :alt: RFC006 UQ Workflow
   :width: 100%
   :target: _images/rfc006_workflow.svg

Walkthrough
-----------

**Stage 1: ``uq sample``** (compute-intensive)

1. **Define Parameter Space** — scalar ``SimulationDataEcoli`` attributes
   specified as ``SimDataParameter`` objects with bounds.  Default 6
   parameters span transcription (RNAP), translation (ribosome speed),
   metabolism (FBA tuning), and cell composition (dry mass fraction).

2. **Load Simulation Data** — Parquet timeseries loaded via hive
   partitioning (vEcoli) or direct state traversal (v2ecoli).  Each sample
   is a full whole-cell simulation executed either as a subprocess through
   ``runscripts/workflow.py`` (vEcoli) or in-process via a process-bigraph
   composite (v2ecoli).  Parameter mutations use the vEcoli
   ``sim_data_setattr`` variant mechanism or in-memory dot-path mutation
   on a cache bundle respectively.

3. **Aggregation Strategies 1--3** — three parallel views of the same
   simulation output:

   - **Strategy 1 (Uniform):** all cells pooled — baseline population average.
   - **Strategy 2 (By Generation):** grouped by cell division generation —
     reveals if parameter importance shifts as cells age.
   - **Strategy 3 (By Lineage Seed):** grouped by stochastic lineage —
     reveals if sensitivity depends on random initial conditions.

4. **Variance Decomposition** — ANOVA-style partitioning of total
   variance into generation, seed, and residual fractions.  High residual
   suggests cell-cycle-driven variance, motivating Phase 2.

**Stage 2: ``uq quantify``** (fast, repeatable)

5. **PCE Surrogate** — Legendre polynomial chaos expansion fitted via
   PyTUQ (``lsq``, ``bcs``, or ``anl`` regression).  Produces a cheap
   polynomial approximation f(x) that can be evaluated instantly without
   running vEcoli.  PCE is well-suited for stochastic systems like vEcoli
   because it captures the full input-output relationship including
   nonlinear interactions.

6. **Sobol Indices** — first-order (S_i) and total-order (S_Ti) variance
   decomposition computed analytically from the PCE coefficients (Sudret
   2008).  No additional sampling required.

**Phase 2: Growth-Stratified (Strategy 4)**

The cell cycle coordinate theta is defined as normalized log(dry_mass):

.. code-block:: text

   theta = [log(m) - log(m_birth)] / [log(m_div) - log(m_birth)]

Timepoints are binned into ``n_bins`` stages (default 10), and an
independent PCE + Sobol analysis is performed per stage.  This reveals
how parameter importance evolves within a single cell cycle — e.g.,
``cell_dry_mass_fraction`` dominates near birth (theta ~ 0) while
``basal_elongation_rate`` dominates near division (theta ~ 1).

Outputs
-------

``uq quantify`` exports all artifacts to a single directory:

.. code-block:: text

   export_dir/
   +-- uq_results.json            # All strategies, Sobol indices, metadata
   +-- report.html                # Self-contained interactive HTML report
   +-- manifest.json              # Reproducibility (git SHAs, packages, CLI)
   +-- population_surrogate/      # PCE coefficients + multi-indices (.npy)
   +-- population_sobol/          # S_i, S_Ti arrays
   +-- generation_N_sobol/        # Per-generation Sobol (Strategy 2)
   +-- seed_N_sobol/              # Per-seed Sobol (Strategy 3)
   +-- growth_stage_N_sobol/      # Per-stage Sobol (Strategy 4)
   +-- growth_stratified_surrogate/

The **HTML report** (``report.html``) provides a publication-ready
summary with interactive PCE exploration — see :doc:`cli_reference`
for the ``uq report`` command.

Key modules
-----------

=============================  ================================================
Module                         Role
=============================  ================================================
``uq/workflow.py``             PyTUQ UQPC pipeline (sample + quantify)
``uq/report.py``               Self-contained HTML report generator
``uq/cli.py``                  Typer CLI (all commands; primary entry point)
``uq/vecoli_config.py``        vEcoli workflow-config helpers + BYO-variants
                               reverse-map + PCE adequacy diagnostic
``uq/growth.py``               Growth-stage theta binning
``uq/observables.py``          Observable presets + collection
``uq/remote.py``               SMS-API client for remote execution
``uq/multi_condition.py``      Cross-condition GSA
``uq/v2ecoli_bridge.py``       Observable path maps for v2ecoli backend
``uq/generators/v2ecoli.py``   v2ecoli in-process simulation generator
``uq/generators/vecoli.py``    vEcoli subprocess simulation generator (deprecated)
``uq/tui.py``                  Textual TUI client (consumer of vecoli_config)
=============================  ================================================

``uq/vecoli_config.py`` is the canonical home for the vEcoli workflow-config
machinery — ``_build_config`` (with the BYO override guard),
``_build_variants_from_samples``, ``_x_from_variants`` (the reverse-map for
BYO mode), ``_pce_sample_adequacy``, ``_collect_variant_timeseries``, and
``_get_vecoli_root``.  Dependency direction is strict::

    cli.py        ─┐
    tui.py        ─┼─→  vecoli_config.py
    processes.py  ─┘

No reverse imports — the TUI consumes the CLI's helpers, not the other way
around.  Any new vEcoli-config code should land in ``vecoli_config.py``.
