CLI Reference
=============

``uq`` is a `Typer <https://typer.tiangolo.com/>`_ application.  Every
command is also available programmatically through :py:mod:`uq.workflow`.

Top-level
---------

.. code-block:: text

   uv run uq [OPTIONS] COMMAND [ARGS]...

   Commands:
     help              Show help for a specific subcommand, or the main CLI.
      sample            UQPC Steps 1-3: sample via PCRV.sampleGerm(), run vEcoli or v2ecoli (local or remote).
     quantify          UQPC Steps 4-5: fit PCE surrogates, compute Sobol (all 4 strategies).
     report            Generate a self-contained HTML report from UQ results.
     fetch             Download cd1 analysis outputs from a completed SMS-API simulation.
     show-config       Preview the vEcoli workflow config JSON without running.
     compare           Side-by-side Sobol comparison across multiple UQ experiments.
     export-figures    Publication-ready PDFs and LaTeX from results.
     suggest-experiment  Identify max-uncertainty parameter region.
     dashboard         Launch the uq interactive dashboard.
     tui               Launch the UQPC interactive terminal UI (Textual).
     gui               Launch the UQPC interactive GUI (marimo).
     init              Guided project setup wizard.

``uq help``
-----------

Shows help for the main CLI or any subcommand.  Trailing ``help`` also
works at any nesting level, mirroring the Atlantis CLI pattern.

.. code-block:: text

   uv run uq help                # all commands
   uv run uq help sample         # help for sample
   uv run uq sample help         # same (trailing alias)
   uv run uq quantify help       # help for quantify
   uv run uq fetch help          # help for fetch

``uq sample``
-------------

UQPC steps 1-3.  Sets up the input PC, draws germ samples, and evaluates
the simulation through vEcoli (subprocess via ``runscripts/workflow.py``)
or v2ecoli (in-process via process-bigraph composite).

.. code-block:: text

   uv run uq sample SIM_DATA_PATH [OPTIONS]

Arguments:

* ``SIM_DATA_PATH`` — absolute or relative path to a pre-computed
  ``simData.cPickle`` produced by vEcoli's Parca.

Options:

``--cache-dir PATH``
    Directory to write cached ``X.npy``, ``Y.npy``, ``germ_train.npy``,
    ``timeseries/``, and (if ``--n-test > 0``) ``X_test.npy`` /
    ``Y_test.npy`` / ``timeseries_test/``.  Default ``./uq_cache``.

``--n-samples INTEGER``
    Number of training samples drawn from the germ measure.  Equivalent
    to the UQPC ``--nqd`` flag in random-sampling mode.  Default ``20``.

``--n-test INTEGER``
    Number of **held-out validation samples** (PyTUQ UQPC ``--ntst``).
    Concatenated onto the training samples and evaluated in the same
    vEcoli workflow; split back out before caching.  Consumed by
    ``quantify`` to compute test relative errors.  Default ``0``.

``--seed INTEGER``
    RNG seed for ``PCRV.sampleGerm``.  Default ``42``.

``--generations INTEGER``
    Generations per variant/lineage-seed pair.  Set ``>= 2`` to enable
    RFC006 aggregation strategy 2 (by generation).  Default ``1``.

``--n-init-sims INTEGER``
    Initial lineage seeds per variant.  Set ``>= 2`` to enable RFC006
    aggregation strategy 3 (by lineage seed).  Default ``1``.

``--max-duration FLOAT``
    Per-simulation wall-clock limit, seconds.  Default ``10800``.

``--params-file PATH``
    Optional JSON file of ``SimDataParameter`` dicts overriding the
    default six sim_data parameters.  See
    ``examples/uq_artifacts/params/params_demo.json``.

``--observables TEXT`` (repeatable)
    Observable presets to extract from vEcoli Parquet output, matching
    the cd1 analysis modules used for Vegas/Bermuda CD1 deliverables.
    Pass multiple times to compose presets.  Default ``mass``.

    ================  ================================================  ========
    Preset            cd1 module equivalent                              Features
    ================  ================================================  ========
    ``mass``          cd1_higher_order_properties (raw scalars)          5
    ``higher_order``  cd1_higher_order_properties (derived metrics)      6
    ``exchange_fluxes`` cd1_exchange_fluxes                              ~87
    ``transcriptome`` cd1_transcriptomics                                ~4,300
    ``proteome``      cd1_proteomics                                     ~4,300
    ``fluxome``       cd1_fluxomics (dry-mass normalized)                ~2,800
    ================  ================================================  ========

    Example — all CD1 analyses at once::

        uv run uq sample simData.cPickle \
            --observables higher_order \
            --observables exchange_fluxes \
            --observables transcriptome \
            --observables proteome \
            --observables fluxome

``--backend [vecoli|v2ecoli]``
    Simulation backend for generating samples:

    * ``vecoli`` (default) — subprocess execution via ``runscripts/workflow.py``,
      Parquet output, Nextflow-managed parallelism.
    * ``v2ecoli`` — in-process execution via process-bigraph composite,
      direct state traversal for observable extraction,
      ``ProcessPoolExecutor`` parallelism (controlled by ``--max-workers``).
      Preferred for new work.

``--max-workers INTEGER``
    Maximum parallel workers for the v2ecoli backend.  Only used when
    ``--backend v2ecoli``.  Controls the ``ProcessPoolExecutor`` concurrency
    level.  Default ``1``.

``--generation-lower-bound INTEGER``
    Skip generations below this value when aggregating observables.
    Mirrors the cd1 ``generation_lower_bound`` parameter — filters
    early transient dynamics so the sensitivity analysis focuses on
    steady-state growth.  Default ``0`` (keep all).

``--base-config PATH``
    Base vEcoli config JSON to merge with.  Preserves ``parca_variants``,
    ``analysis_options``, and other multi-parca keys from multi-condition
    configs.  UQ-specific fields (variants, emitter, output_dir) override
    the base; everything else is preserved.

``--conditions TEXT`` (repeatable)
    RNA-seq dataset IDs for **cross-condition UQ** (one per flag).
    Populates ``parca_variants`` for cross-condition global sensitivity
    analysis.  Requires vEcoli ``multi-parca-aws`` branch.  Example::

        uv run uq sample simData.cPickle \
            --conditions vecoli_m9_glucose_minus_aas \
            --conditions vecoli_m9_glucose_plus_aas

    ``uq quantify`` auto-detects multi-condition caches via
    ``conditions.json`` and runs per-condition quantification plus
    cross-condition rank stability analysis.

``--variants-source [params-file|base-config]``
    Where vEcoli ``variants`` come from.  Default ``params-file``.

    * ``params-file`` (default) — PCRV-sample from ``--params-file`` bounds,
      encode each row as a ``sim_data_setattr`` mutation.  Standard UQ
      workflow; ``n_variants == n_samples`` by construction.
    * ``base-config`` — **Bring-your-own-variants**.  Trust the
      ``variants`` block in ``--base-config`` verbatim, reverse-map the
      mutation values back into ``X`` so ``quantify`` can regress against
      the perturbations vEcoli actually applied.

    Constraints under ``base-config`` mode (local-only for now):

    * Requires ``--base-config`` pointing at a JSON whose top-level has a
      ``variants`` key.  Must be the ``sim_data_setattr`` module (other
      variant modules cannot be reverse-mapped to ``X``).
    * Forbidden with ``--api-url`` (remote mutation pushdown is unfinished)
      and ``--backend v2ecoli`` (different cache-bundle codepath).  Both
      raise a clear error.
    * ``--n-samples`` is **forced** to the length of the variants list.
    * ``--n-test`` is **ignored** (held-out validation requires PCRV
      sampling).
    * A PCE adequacy diagnostic prints after sampling:

      .. code-block:: text

         PCE adequacy: n_samples=12 < basis_size=28 for PCE order=2,
                       6 params. Least-squares is ill-posed. Need at
                       least 28 samples; 56+ recommended. Either add
                       variants, drop --polynomial-order to 1
                       (basis_size=7), or use --regression bcs.

      Status is one of ``underdetermined`` (ratio < 1, red),
      ``marginal`` (1 <= ratio < 2, yellow), or ``ok`` (ratio >= 2, dim).

    Example BYO run::

        uv run uq sample simData.cPickle \
            --variants-source base-config \
            --base-config examples/vecoli_configs/mec.json \
            --params-file examples/uq_artifacts/params/params_demo.json \
            --cache-dir ./uq_cache_mec \
            --generations 4 --n-init-sims 2

    Override-guard behavior in default mode:  even without
    ``--variants-source base-config``, if ``--base-config`` is supplied
    and its JSON has a top-level ``variants`` key, that block is now
    preserved instead of being clobbered by the auto-generated
    ``sim_data_setattr`` block.  Note that this leaves ``X`` derived from
    PCRV sampling — unrelated to the variants vEcoli runs — so the cache
    will mis-align unless you also pass ``--variants-source base-config``.

Remote execution flags
^^^^^^^^^^^^^^^^^^^^^^

These flags switch ``uq sample`` from local subprocess execution to
**SMS-API remote execution**.  The API runs vEcoli on AWS Batch and
returns cd1 analysis TSVs instead of raw Parquet.

``--api-url TEXT``
    SMS-API base URL (e.g. ``http://localhost:8080``).  When set,
    simulations run on the SMS-API cluster instead of local vEcoli.
    The stanford-test deployment is accessed via ``kubectl port-forward``
    or ``ptools-proxy.sh``.

``--simulator-id INTEGER``
    SMS-API simulator ``database_id`` (required with ``--api-url``).
    Identifies which vEcoli build on the server to use.

``--config-filename TEXT``
    vEcoli config filename on the server (``configs/`` directory).
    Default ``api_simulation_default.json``.

``--ecoli-sources-repo TEXT``
    GitHub URL for the ``ecoli-sources`` data repo.  The server downloads
    and syncs to S3 automatically — no local clone or AWS CLI needed.

``--ecoli-sources-ref TEXT``
    Git ref (branch/tag/commit) for the ecoli-sources repo.
    Default ``main``.

``uq fetch``
------------

Download and inspect cd1 analysis outputs from a **completed** SMS-API
simulation, without running a full UQ campaign.  Useful for verifying
the data format and observable coverage before committing to a remote
sampling run.

.. code-block:: text

   uv run uq fetch SIMULATION_ID [OPTIONS]

Arguments:

* ``SIMULATION_ID`` — SMS-API simulation ``database_id``.

Options:

``--cache-dir PATH``
    Directory to download into.  Default ``./uq_cache``.

``--api-url TEXT``
    SMS-API base URL.  Default ``http://localhost:8080``.

``--observables TEXT`` (repeatable)
    cd1 observable presets to extract from the downloaded TSVs.
    Default: all five cd1 modules.

The command downloads the tar.gz from ``/simulations/{id}/data``,
parses the cd1 TSV files (3-column tab-separated: identifier, mean,
std), and prints an observable summary table:

.. code-block:: text

   uv run uq fetch 48 --api-url http://localhost:8080

   ┌──────────────────────────────────────────────────────┐
   │ PRESET            MODULE                   ROWS      │
   │ higher_order      cd1_higher_order_prop…      5      │
   │ transcriptome     cd1_transcriptomics      4345      │
   │ proteome          cd1_proteomics           4309      │
   │ fluxome           cd1_fluxomics            2820      │
   │ exchange_fluxes   cd1_metabolomics          165      │
   └──────────────────────────────────────────────────────┘
   Total observables: 11644

``uq quantify``
---------------

UQPC steps 4-5.  Fits PCE surrogates per aggregation strategy, computes
Sobol indices, reports relative errors.

.. code-block:: text

   uv run uq quantify SIM_DATA_PATH [OPTIONS]

Options:

``--cache-dir PATH``
    Path to the cache produced by ``uq sample``.  Default ``./uq_cache``.

``--export-path PATH``
    Where to write ``uq_results.json``, per-strategy Sobol ``.npy``
    files, and the two exported PCE surrogates.  Default ``./uq_results``.

``--n-bins INTEGER``
    Number of growth-stratified bins (strategy 4).  Bins are uniform in
    ``θ = [log(mass) − log(mass_birth)] / [log(mass_div) − log(mass_birth)]``.
    Default ``10``.

``--polynomial-order INTEGER``
    Output PCE polynomial degree (UQPC ``--outord``).  Default ``2``.

``--regression [lsq|bcs|anl]``
    PyTUQ regression backend:

    * ``lsq`` — least squares (default)
    * ``bcs`` — Bayesian Compressed Sensing, sparse
    * ``anl`` — analytical Bayesian projection

``--tol FLOAT``
    BCS sparsity tolerance (UQPC ``--tol``).  Only used when
    ``--regression=bcs``.  Default ``1e-3``.

``--bootstrap INTEGER``
    Number of bootstrap resamples for empirical 95% CIs on Sobol indices.
    Default ``0`` (off).  Standard non-parametric Sobol bootstrap
    (Archer/Iooss/Saltelli): resample ``(X, Y)`` rows with replacement,
    refit the PCE via the same PyTUQ regression backend, recompute Sobol
    via ``PCRV.computeSens`` / ``computeTotSens``, report the 2.5–97.5
    percentile interval per parameter.  When > 0, Sobol tables grow from
    ``S_Ti`` to ``S_Ti  [low, high]`` so you can tell whether
    ``S_Ti = 0.42`` for one parameter and ``S_Ti = 0.38`` for another are
    meaningfully different.  Typical value: ``200``.  Threaded through
    all four RFC006 strategies.

Report
^^^^^^

``quantify`` produces two reports.  Beyond the Sobol tables, the terminal
report includes four diagnostic panels designed for honest interpretation:

1. **DESIGN QUALITY @ order=N** — re-runs the count + κ(A) check at the
   *actual* ``--polynomial-order`` used in this quantify run.  The
   sample-time check assumed order 2; if you pass ``--polynomial-order 3``
   the basis size jumps from 28 to 84 (for 6 params) and an ok design at
   p=2 can become underdetermined at p=3 without this panel catching it.

2. **SURROGATE QUALITY** — per-output training relative error, plus a
   ``TEST`` column when the cache contains ``X_test``/``Y_test``, plus a
   ``CV (5-fold)`` column when no held-out test set is present (BYO mode
   or default with ``--n-test 0``).  k-fold CV is skipped automatically
   when ``N < 2.5 × basis_size`` — at that point CV error is noise, not
   generalization, and the subtitle says so.

3. **NOISE FLOOR // ALEATORIC vs EPISTEMIC** — ANOVA-style separation of
   total Y variance into between-variant (epistemic, PCE-explainable) and
   within-variant (aleatoric, irreducible) components using vEcoli's
   ``lineage_seed`` replicate structure.  Verdict color-coded: signal
   ≥ 80% dim ("interpret straightforwardly"); 50% ≤ signal < 80% yellow
   ("Sobol indices undershoot the per-parameter share of *explainable*
   variance"); < 50% red ("aleatoric exceeds parameter signal").  When
   ``n_init_sims = 1`` (the default), prints a "not estimable" advisory
   directing the user to rerun ``uq sample --n-init-sims 4``.

4. **Per-strategy Sobol tables** — population, by-generation,
   by-lineage-seed, growth-stratified — each shows top parameters with
   either ``S_Ti`` or ``S_Ti  [95% CI]`` depending on whether
   ``--bootstrap`` is set.

The **HTML report** (``report.html`` in the export directory) is
automatically generated via ``uq report``.  See ``uq report`` for details.

``uq report``
-------------

Generate a self-contained HTML report from UQ export artifacts.
Automatically run after ``uq quantify``, but can also be invoked
standalone to regenerate a report from existing results.

The report is a **single HTML file** with no external dependencies —
inline SVG charts, CSS, and JavaScript.  It includes:

* All 4 RFC006 strategies (population, by-generation, by-lineage,
  growth-stratified) with bar charts, heatmaps, and ranking tables
* Cross-strategy comparison with automatic insight callouts
* Interactive PCE Explorer — drag parameter sliders to evaluate the
  fitted surrogate in real time (client-side Legendre evaluation)
* Experimental design section with parameter specs, SimData dot-paths,
  and biological role descriptions
* Provenance metadata (git SHAs, package versions, CLI command)

.. code-block:: text

   uv run uq report [OPTIONS]

Options:

``--results-path PATH``
    Path to the UQ export directory (must contain ``uq_results.json``).
    Default ``./uq_results``.

``--output PATH``
    Output HTML file path.  Default ``<results-path>/report.html``.

``uq compare``
--------------

Side-by-side Sobol comparison across multiple UQ export directories.

.. code-block:: text

   uv run uq compare DIR1 DIR2 [DIR3 ...]

``uq export-figures``
---------------------

Generate publication-ready PDFs and a LaTeX Sobol table from UQ results.
Requires ``kaleido`` (``uv pip install kaleido``).

.. code-block:: text

   uv run uq export-figures [OPTIONS]

Options:

``--results-path PATH``
    Path to the UQ export directory.  Default ``./uq_results``.

``--output-dir PATH``
    Output directory for generated files.  Default ``<results>/figures/``.

Outputs: ``sobol_bar_chart.pdf``, ``spectrogram.pdf``,
``response_curves.pdf``, ``sobol_table.tex``.

``uq suggest-experiment``
-------------------------

Identify the parameter-space region where prediction uncertainty is
highest — i.e. where additional experimental measurements would most
reduce model uncertainty.

.. code-block:: text

   uv run uq suggest-experiment [OPTIONS]

Options:

``--results-path PATH``
    Path to the UQ export directory.  Default ``./uq_results``.

``--n-grid INTEGER``
    Grid points for variance scanning.  Default ``1000``.

``uq show-config``
------------------

Generates the full vEcoli workflow config JSON that ``uq sample`` would
pass to the simulation backend (``runscripts/workflow.py`` for vEcoli,
or the v2ecoli composite builder), **without running anything**.  Useful
for stakeholder review, debugging, or manual execution.

.. code-block:: text

   uv run uq show-config SIM_DATA_PATH [OPTIONS]

Options:

``--n-samples INTEGER``
    Number of variants to include in the config.  Default ``5``.

``--output-file PATH``
    Write JSON to a file instead of printing to stdout.

All other flags (``--seed``, ``--generations``, ``--n-init-sims``,
``--max-duration``, ``--params-file``) mirror ``uq sample``.

``uq dashboard``
----------------

.. code-block:: text

   uv run uq dashboard [--results-path PATH] [--run-mode tk|mo]

``--run-mode tk``
    Tkinter DAW-style dashboard with draggable parameter markers
    (:py:mod:`app.uq_daw_simple`).  Default.

``--run-mode mo``
    Launches ``app/dashboard_simple.py`` as a marimo notebook via
    ``marimo edit``.

``uq tui``
----------

Textual terminal UI — the same workflow with live progress, tabbed
strategy tables, and interactive parameter selection.

.. code-block:: text

   uv run uq tui

``uq gui``
----------

Launches ``app/gui.py`` as a marimo application:

.. code-block:: text

   uv run uq gui

``uq init``
-----------

Guided project setup wizard — interactively configure a UQ experiment.

.. code-block:: text

   uv run uq init

``uq plan``
-----------

Size an upcoming ``uq sample`` run.  Given a compute budget (total vEcoli
runs) plus PCE settings and replicate targets, derives ``n_samples``
under the requested constraints and reports PCE adequacy + noise-floor
estimability for the recommended config plus alternatives showing the
trade-off.  Pure arithmetic — no model fits, no simData required.

.. code-block:: text

   uv run uq plan [OPTIONS]

Options:

``--budget INTEGER`` (required)
    Total vEcoli runs available
    (``n_samples × n_init_sims × generations``).

``--params INTEGER``
    PCE input dimension.  Default ``6`` (matches
    ``DEFAULT_SIM_DATA_PARAMETERS``).

``--polynomial-order INTEGER``
    PCE order at quantify time.  Default ``2``.

``--noise-replicates INTEGER``
    Minimum ``n_init_sims`` required to estimate the noise floor.
    Default ``4``.

``--generations INTEGER``
    Target generations per variant.  Default ``4``.

Output: a table with the recommended config plus up to two alternatives.
Each row reports ``n_samples``, ``n_init_sims``, ``generations``, the
adequacy ratio ``N / basis_size``, PCE adequacy status
(``ok``/``marginal``/``underdetermined``), and whether the noise floor
will be estimable.  Example::

    Budget: 1000 vEcoli runs   ·   d=6, p=2, basis_size=28

    OPTION             n_samples  n_init_sims  generations  ratio  status  noise floor?
    recommended              62            4            4   2.21×  ok      yes
    halve generations       125            4            2   4.46×  ok      yes
    single replicate        250            1            4   8.93×  ok      no (n=1)

    Trade-offs:
    - halve generations: 2× the samples for PCE fit, half the cell-cycle coverage.
    - single replicate: 4× the samples, but Triage 5 / Gap 2 noise floor unestimable.

Programmatic API
----------------

Everything in the CLI delegates to two functions:

.. code-block:: python

   from uq.workflow import sample, quantify

   # vEcoli backend (default)
   cache = sample(
       sim_data_path="/path/to/simData.cPickle",
       cache_dir="./uq_cache",
       n_samples=200,
       generations=2,
   )

   # v2ecoli in-process backend
   cache = sample(
       sim_data_path="/path/to/simData.cPickle",
       cache_dir="./uq_cache",
       n_samples=200,
       generations=2,
       backend="v2ecoli",
       max_workers=4,
   )

   # quantify is backend-agnostic — same cache format
   result = quantify(
       cache_dir="./uq_cache",
       sim_data_path="/path/to/simData.cPickle",
       polynomial_order=3,
       regression="lsq",
       export_path="./uq_results",
   )

See :py:mod:`uq.workflow` for the full signatures.
