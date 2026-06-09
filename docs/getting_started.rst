Getting Started
===============

This guide walks through the shortest path from "I have a simulation
and a ``simData.cPickle``" to "I have PCE Sobol indices for five
sim_data parameters."

Prerequisites
-------------

The repository supports two simulation backends:

**vEcoli backend** (``--backend vecoli``, default)
   1. A working **vEcoli** checkout at ``../vEcoli`` (editable install).
   2. A pre-computed ``simData.cPickle`` produced by vEcoli's Parca
      (for example ``../vEcoli/reconstruction/sim_data/kb/simData.cPickle``).

**v2ecoli backend** (``--backend v2ecoli``, preferred)
   1. A working **v2ecoli** checkout at ``../v2ecoli`` (editable install).
   2. A pre-computed ``simData.cPickle`` (same format; v2ecoli converts
      it to a cache bundle at runtime).

Both backends share the same ``simData.cPickle`` format and the same
``uq quantify`` / ``uq report`` pipeline — only the execution engine
differs.

.. code-block:: bash

   git clone https://github.com/.../uqEcoli.git
   cd uqEcoli
   uv sync --all-extras

This repository depends on vEcoli (and optionally v2ecoli) as editable
packages installed alongside it.

The two-stage workflow
----------------------

``uq`` exposes the PyTUQ UQPC workflow as two commands:

* ``uq sample`` — UQPC steps **1-3**: build the input PC, draw germ
  samples, evaluate vEcoli, cache ``(X, Y, timeseries)`` to disk.
* ``uq quantify`` — UQPC steps **4-5**: fit the PCE surrogate, compute
  Sobol indices for all four RFC006 aggregation strategies, report
  surrogate relative errors, and export a dashboard-ready artifact
  directory.

Two axes of variation — pick the right mode
-------------------------------------------

Before reaching for a CLI flag, decide which axis you're varying.
``sim_data`` can be mutated along two mathematically distinct axes:

* **UQ axis** :math:`P_{uq}` — *continuous* parameters with a known
  prior (uniform on a bounded interval).  PCRV samples them; PCE
  decomposes ``Var[Y]`` across them via Sobol.  Declared via
  ``--params-file``.
* **Design axis** :math:`P_{design}` — *categorical / structural*
  perturbations (knockouts on/off, environment swaps, timeline events
  at fixed boundaries).  No continuous interval; PCE cannot decompose
  variance across this axis.  Handled via per-condition PCE +
  cross-condition Sobol comparison.  Declared via ``--design-config``
  (variant-level) or ``--conditions`` (parca-level).

When both axes are present, the framework runs an :math:`M \times N`
grid: M design conditions, each with N PCRV samples reused across
conditions for paired comparison.  **The two parameter sets must be
disjoint at the attr_path level** for PCE Sobol to be sound — if they
overlap, ``sim_data_setattr`` silently overwrites the design's value
and the per-condition Sobol indices answer the wrong question.  The CLI
surfaces this overlap before any compute is committed.

See :doc:`cli_reference` § "Two axes of variation: design conditions ×
UQ samples" for the full decision table, the collapse rule (when
:math:`P_{design} = P_{uq}`, use single-layer ``--variants-source
base-config`` or default mode instead of two-layer ``--design-config``),
and the subtle case where finely-spaced designs should be folded into
the UQ layer for cheaper, cleaner Sobol.

Stage 1 — sample
^^^^^^^^^^^^^^^^

.. code-block:: bash

   uv run uq sample /path/to/simData.cPickle \
       --cache-dir ./uq_cache \
       --n-samples 50 \
       --n-test 10 \
       --generations 2 \
       --observables higher_order \
       --observables exchange_fluxes \
       --observables transcriptome \
       --generation-lower-bound 2

What happens:

1. ``ParameterDataset`` loads ``simData.cPickle`` and projects the six
   default parameters into a generic ``XSpaceVecoli`` parameter space.
2. ``_setup_input_pc`` builds a ``pytuq.rv.pcrv.PCRV`` (Legendre basis,
   uniform priors) encoding the affine germ-to-physical map.
3. ``PCRV.sampleGerm(n_samples)`` draws 50 germ samples; ``evalPC``
   maps them to physical space.  ``--n-test 10`` adds 10 held-out
   validation samples (UQPC ``--ntst``).
4. Each row of ``X`` becomes one vEcoli variant via the upstream
   ``sim_data_setattr`` variant function.
5. ``runscripts/workflow.py`` is spawned as a subprocess.  Nextflow
   manages variant instantiation, simulation, and Parquet emission.
6. Observables are extracted from hive-partitioned Parquet using the
   selected ``--observables`` presets (see :doc:`cli_reference` for the
   full preset table).  Each preset mirrors a cd1 analysis module.
   ``--generation-lower-bound`` filters early transient generations.
7. Results are saved to ``./uq_cache`` as a ``PrecomputedCache``.
   Test samples are sliced off and stored as ``X_test.npy`` /
   ``Y_test.npy``.

Stage 2 — quantify
^^^^^^^^^^^^^^^^^^

.. code-block:: bash

   uv run uq quantify /path/to/simData.cPickle \
       --cache-dir ./uq_cache \
       --export-path ./uq_results \
       --polynomial-order 3 \
       --regression lsq \
       --bootstrap 200

What happens:

1. The cache is loaded, parameter space is reconstructed, and germ
   samples are read back from ``germ_train.npy``.
2. ``run_uqpc`` fits a multi-output PCE per aggregation strategy using
   PyTUQ's ``pcrv.evalBases``/``lsq.fita`` stack.
3. Sobol main, total, and joint indices are computed analytically from
   the PCE coefficients via ``PCRV.computeSens``/``computeTotSens``
   (Sudret 2008).
4. Relative training errors (and test errors if ``X_test``/``Y_test``
   are in the cache; otherwise a 5-fold CV column) are computed per
   output and displayed in the Rich report.
5. With ``--bootstrap 200``, empirical 95% CIs on Sobol indices are
   computed (non-parametric bootstrap: resample rows → refit PCE →
   recompute Sobol → percentile interval).  Sobol tables grow from
   ``S_Ti`` to ``S_Ti  [low, high]``.
6. A ``DESIGN QUALITY`` panel re-runs the count + κ(A) check at the
   actual ``--polynomial-order`` used here (catches the case where
   ``polynomial-order 3`` silently turned an ok design underdetermined).
7. A ``NOISE FLOOR`` panel separates total Y variance into between-
   variant (epistemic) and within-variant (aleatoric, from
   ``lineage_seed`` replicates) components — color-coded by signal
   fraction.  Reports "not estimable" when ``n_init_sims = 1``.
8. A ``QuantifyResult`` is exported to ``./uq_results/`` as a dashboard
   schema, per-strategy Sobol ``.npy`` files, and the two PCE surrogates
   (population + growth-stratified).

Sizing a run with ``uq plan``
-----------------------------

Before launching a vEcoli batch, ``uq plan`` checks whether a compute
budget is enough for a stable PCE fit at the order you plan to use:

.. code-block:: bash

   uv run uq plan --budget 1000 \
       --params 6 --polynomial-order 2 \
       --noise-replicates 4 --generations 4

Pure arithmetic — no simData required.  Prints a recommended
``(n_samples, n_init_sims, generations)`` allocation plus 1–2
alternatives showing the trade-off (halve generations → more PCE
samples; drop to one replicate → maximum samples but no noise-floor
estimability).  Each row reports the adequacy ratio ``N / basis_size``
and a status: ``ok`` / ``marginal`` / ``underdetermined``.

Bring your own variants
-----------------------

If you already have a vEcoli config JSON with a fully-spec'd ``variants``
block — a previously-curated sweep, a multi-condition design, the output
of an Atlantis run — you can hand it to ``uq sample`` directly and have
UQ run *those* variants instead of generating new ones from a parameter
file:

.. code-block:: bash

   uv run uq sample /path/to/simData.cPickle \
       --variants-source base-config \
       --base-config examples/vecoli_configs/mec.json \
       --params-file examples/uq_artifacts/params/params_demo.json \
       --cache-dir ./uq_cache_mec \
       --generations 4 --n-init-sims 2

Under ``--variants-source base-config``:

1. ``_load_variants_from_base_config()`` reads the ``variants`` block
   from the JSON.  Must be the ``sim_data_setattr`` module (other modules
   carry no scalar mutation values to reverse-map).
2. ``_x_from_variants()`` walks the mutation list and reconstructs the
   ``X`` matrix column-ordered by ``--params-file``'s ``attr_path`` list.
   Every mutation entry must contain every spec's ``attr_path`` —
   missing-key and index mismatches raise clear errors.
3. PCRV sampling is **skipped**; ``germ_train`` is computed via affine
   inverse of the input PCRV map.  ``n_samples`` is forced to the
   variants-list length.  ``--n-test`` is ignored (held-out validation
   requires PCRV sampling).
4. A **PCE adequacy diagnostic** prints — red if your variants count is
   below the PCE basis size for the default order-2 quantify run, yellow
   if marginal, dim if comfortable.  It tells you exactly how many more
   variants to add, what ``--polynomial-order`` to drop to at quantify
   time, or whether ``--regression bcs`` would handle the sparsity better.

Constraints (BYO is local-mode only for now):

* ``--backend vecoli`` only.  ``--backend v2ecoli`` and ``--api-url``
  reject ``--variants-source base-config`` with a clear error message;
  remote mutation pushdown is tracked as a follow-up.
* The variants block must use ``sim_data_setattr``.  Other modules
  (e.g. ``condition``, ``flux_kinetics``) skip the reverse-map and
  can't drive a meaningful ``quantify`` run.

Stage 2 (``uq quantify``) is identical to the default path — the cache
format is unchanged.

Remote execution via SMS-API
---------------------------

If you don't have a local vEcoli checkout, you can run the sampling step
against the **SMS-API** — a REST API that runs vEcoli on AWS Batch and
returns cd1 analysis outputs as pre-aggregated TSVs.

Prerequisites for remote mode:

1. Access to an SMS-API deployment (e.g. stanford-test at
   ``http://localhost:8080`` via ``kubectl port-forward``).
2. A ``simulator_id`` on that deployment (use ``uq fetch`` to explore).
3. A local ``simData.cPickle`` (still needed for parameter space setup).

.. code-block:: bash

   # Inspect what a completed simulation looks like
   uv run uq fetch 48 --api-url http://localhost:8080

   # Remote sampling: Steps 1-2 run locally, Step 3 submits to SMS-API
   uv run uq sample /path/to/simData.cPickle \
       --api-url http://localhost:8080 \
       --simulator-id 11 \
       --n-samples 20 \
       --observables higher_order \
       --observables transcriptome

In remote mode, observables come from cd1 analysis TSVs (3-column format:
identifier, mean, std) instead of raw Parquet.  The ``mass`` preset is
local-only; remote mode supports ``higher_order``, ``transcriptome``,
``proteome``, ``fluxome``, and ``exchange_fluxes``.

Stage 2 (``uq quantify``) is identical regardless of how Stage 1 ran —
the cache format is the same.

Cross-condition GSA
-------------------

With the vEcoli ``multi-parca-aws`` branch, you can run UQ across
multiple growth conditions and compare which parameters are universally
important vs. condition-specific:

.. code-block:: bash

   uv run uq sample /path/to/simData.cPickle \
       --conditions vecoli_m9_glucose_minus_aas \
       --conditions vecoli_m9_glucose_plus_aas \
       --n-samples 20

   uv run uq quantify /path/to/simData.cPickle

``quantify`` auto-detects the multi-condition cache and prints a
cross-condition comparison table with rank stability indicators.

Interactive clients
-------------------

All clients wrap the same two-stage workflow — pick your surface:

.. code-block:: bash

   uv run uq report                         # self-contained HTML report (auto-generated by quantify)
   uv run uq tui                            # Textual TUI
   uv run uq gui                            # marimo browser GUI
   uv run uq dashboard                      # tkinter DAW-style dashboard
   uv run uq fetch 48                       # inspect SMS-API simulation outputs
   uv run uq compare ./results_a ./results_b  # side-by-side Sobol comparison
   uv run uq export-figures                 # publication-ready PDFs + LaTeX
   uv run uq suggest-experiment             # identify max-uncertainty region

See :doc:`cli_reference` for a per-flag breakdown of every command and
:doc:`tutorial_workflow` for the underlying math.
