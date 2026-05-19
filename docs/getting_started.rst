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
       --regression lsq

What happens:

1. The cache is loaded, parameter space is reconstructed, and germ
   samples are read back from ``germ_train.npy``.
2. ``run_uqpc`` fits a multi-output PCE per aggregation strategy using
   PyTUQ's ``pcrv.evalBases``/``lsq.fita`` stack.
3. Sobol main, total, and joint indices are computed analytically from
   the PCE coefficients via ``PCRV.computeSens``/``computeTotSens``
   (Sudret 2008).
4. Relative training errors (and test errors if ``X_test``/``Y_test``
   are in the cache) are computed per output and displayed in the Rich
   report.
5. A ``QuantifyResult`` is exported to ``./uq_results/`` as a dashboard
   schema, per-strategy Sobol ``.npy`` files, and the two PCE surrogates
   (population + growth-stratified).

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
