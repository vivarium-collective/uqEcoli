uqEcoli — UQ for the vEcoli whole-cell model
============================================

``uqEcoli`` is a thin glue layer between **vEcoli**
(<https://covertlab.github.io/vEcoli/>) or **v2ecoli** (in-process
process-bigraph composites) and **PyTUQ's UQPC workflow**
(<https://sandialabs.github.io/pytuq/apps/uqpc.html>).  It turns the
six-step UQPC pipeline (setup → sample → evaluate → surrogate → error →
Sobol) into a two-command CLI with two simulation backends:

.. important::

   Every numerically-meaningful step lives in PyTUQ or vEcoli.  This
   repository only adapts their I/O formats and adds end-user entry
   points (CLI, TUI, GUI, marimo dashboard).  See ``SAMPLING.md`` at the
   repo root for a full accounting of *what lives where*.

Two-stage workflow
------------------

.. code-block:: bash

   # Stage 1 — UQPC steps 1-3: germ sampling + simulation evaluation
   # Two backends: --backend vecoli (legacy subprocess, default) or
   #               --backend v2ecoli (in-process, preferred)
   uv run uq sample /path/to/simData.cPickle \
       --cache-dir ./uq_cache \
       --n-samples 200 \
       --n-test 40 \
       --generations 2 \
       --observables higher_order \
       --observables exchange_fluxes \
       --observables transcriptome \
       --generation-lower-bound 2 \
       --backend v2ecoli

   # Stage 2 — UQPC steps 4-5: PCE surrogate fit + Sobol decomposition
   uv run uq quantify /path/to/simData.cPickle \
       --cache-dir ./uq_cache \
       --export-path ./uq_results \
       --polynomial-order 3 \
       --regression lsq

See :doc:`cli_reference` for every flag, and :doc:`tutorial_workflow` for
the full mathematical walkthrough.

Why two stages?
---------------

The expensive operation is step 3 (running the simulation).  Caching its output
means you can iterate freely on PCE order, regression backend, or
aggregation strategy without re-simulating.  The cache directory
contains ``X.npy``, ``Y.npy``, ``germ_train.npy``, optional
``X_test.npy``/``Y_test.npy`` (UQPC ``--ntst``), and per-sample Parquet
timeseries — everything ``quantify`` needs.

User-facing entry points
------------------------

The same two-stage workflow (``uq sample`` → ``uq quantify``) can be
driven through five interfaces plus several utility commands:

==============  =====================================  =========================================
Client / Command  Usage                                 Best for
==============  =====================================  =========================================
CLI (Rich)      ``uv run uq sample`` / ``quantify``    Headless runs, scripts, CI
HTML Report     ``uv run uq report``                   Shareable, publication-ready results
TUI             ``uv run uq tui``                      Terminal dashboards with live progress
GUI (marimo)    ``uv run uq gui``                      Reactive browser notebook
Dashboard       ``uv run uq dashboard``                Draggable DAW-style exploration
---             ---                                    ---
``uq fetch``    ``uv run uq fetch <sim_id>``           Inspect SMS-API simulation outputs
``uq compare``  ``uv run uq compare dir1 dir2``        Side-by-side Sobol comparison
``uq export-figures`` | ``uv run uq export-figures``   Publication-ready PDFs + LaTeX
``uq show-config`` | ``uv run uq show-config``         Preview vEcoli config JSON
``uq suggest-experiment`` | ``uv run uq suggest-experiment`` | Identify max-uncertainty region
``uq init``     ``uv run uq init``                     Guided project setup wizard
``uq tutorial`` | ``uv run uq tutorial``               ­Interactive marimo tutorial
==============  =====================================  =========================================

.. note::
   See :doc:`cli_reference` for per-flag documentation of every command.

Documentation contents
----------------------

.. toctree::
   :maxdepth: 2
   :caption: User Guide

   getting_started
   cli_reference
   tutorial_workflow

.. toctree::
   :maxdepth: 2
   :caption: Topics

   architecture
   aggregation_strategies
   sensitivity_analysis
   cell_cycle

.. toctree::
   :maxdepth: 2
   :caption: UQ-Backed Inference (Layer B & C)

   inference_methodology
   inference_uq_integration
   inference_design_rationale
   inference_tutorial
   api/inference

.. toctree::
   :maxdepth: 1
   :caption: Development

   design_document
   changelog

Indices and tables
------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

References
----------

* PyTUQ UQPC workflow: https://sandialabs.github.io/pytuq/apps/uqpc.html
* vEcoli workflows + variants: https://covertlab.github.io/vEcoli/workflows.html
* Sudret, B. (2008). *Global sensitivity analysis using polynomial chaos expansions*.
  Reliability Engineering & System Safety 93(7), 964-979.
* Xiu, D. & Karniadakis, G.E. (2002). *The Wiener–Askey polynomial chaos for
  stochastic differential equations*. SIAM J. Sci. Comput. 24(2), 619-644.
* Macklin, D. N. *et al.* (2020). *Simultaneous cross-evaluation of heterogeneous
  E. coli datasets via mechanistic simulation*. Science 369(6502).
* Ahn-Horst, T. A. *et al.* (2022). *An expanded whole-cell model of E. coli
  links cellular physiology with mechanisms of growth rate control*. npj
  Systems Biology and Applications 8:30.
