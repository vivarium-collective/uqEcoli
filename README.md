# uqEcoli

[![Documentation Status](https://readthedocs.org/projects/uqecoli/badge/?version=latest)](https://uqecoli.readthedocs.io/en/latest/)
[![Live Demo](https://img.shields.io/badge/demo-live-brightgreen)](https://vivarium-collective.github.io/uqEcoli/demo/)

> **Canonical repo:** `vivarium-collective/uqEcoli`. The original fork at `AlexPatrie/uqEcoli` is archived (pullable, no further development).

**Uncertainty quantification for the vEcoli whole-cell model.**

`uqEcoli` is a thin glue layer between [vEcoli](https://covertlab.github.io/vEcoli/) and
[PyTUQ's UQPC forward-UQ workflow](https://sandialabs.github.io/pytuq/apps/uqpc.html).
It turns the five-step UQPC pipeline (setup → sample → evaluate → surrogate → Sobol)
into a two-command CLI driven by vEcoli's native Nextflow workflows, plus a small
family of end-user clients (CLI, TUI, marimo GUI, tkinter dashboard) over the same
`uq.workflow` engine.

Every numerically-meaningful step is delegated to an upstream project. This repo
only adapts I/O formats and provides ergonomic entry points. See [SAMPLING.md](SAMPLING.md)
for a full accounting of *what lives where*.

---

## Installation

Prerequisites:

1. An editable [vEcoli](https://github.com/CovertLab/vEcoli) checkout at `../vEcoli`
   (this repo depends on it as a path dependency).
2. A pre-computed `simData.cPickle` produced by vEcoli's Parca
   (e.g. `../vEcoli/reconstruction/sim_data/kb/simData.cPickle`).
3. [`uv`](https://docs.astral.sh/uv/) for environment management.

```bash
git clone https://github.com/vivarium-collective/uqEcoli.git
cd uqEcoli
uv sync --all-groups --all-extras
```

That is the entire install. `uv` resolves PyTUQ, vEcoli (editable), and every other
dependency into an isolated virtualenv. All commands in this README are invoked via
`uv run`.

---

## Getting Started

`uqEcoli` exposes the UQPC pipeline as two commands — one expensive, one cheap —
separated by an on-disk cache so you can iterate freely on the analysis side.

### Stage 1 — sample  (UQPC steps 1-3)

```bash
uv run uq sample /path/to/simData.cPickle \
    --cache-dir ./uq_cache \
    --n-samples 200 \
    --n-test 40 \
    --generations 2 \
    --observables higher_order \
    --observables exchange_fluxes \
    --observables transcriptome \
    --generation-lower-bound 2
```

1. Loads `simData.cPickle` and builds a generic parameter space from the six default
   `SimDataParameter` specs.
2. Constructs a PyTUQ `PCRV` (Legendre basis, order 1) encoding the affine map
   `ξ ∈ [-1, 1]^d → [lb, ub]`.
3. Draws `n_samples` training + `n_test` held-out validation samples (UQPC `--ntst`)
   from the germ measure via `PCRV.sampleGerm()`.
4. Encodes every sample as a vEcoli *variant* using the upstream `sim_data_setattr`
   variant function, then spawns `runscripts/workflow.py --config ...` as a subprocess.
5. Extracts observables from the hive-partitioned Parquet tree using the selected
   `--observables` presets — each mirrors a cd1 analysis module (higher-order
   properties, exchange fluxes, transcriptome, proteome, fluxome).
   `--generation-lower-bound` skips early transient generations.
6. Caches `(X, Y, timeseries)` to `./uq_cache`.

### Stage 2 — quantify  (UQPC steps 4-5)

```bash
uv run uq quantify /path/to/simData.cPickle \
    --cache-dir ./uq_cache \
    --export-path ./uq_results \
    --polynomial-order 3 \
    --regression lsq
```

1. Loads the cache (no re-simulation).
2. Fits a PCE surrogate per aggregation strategy via PyTUQ (`lsq` / `bcs` / `anl`).
3. Computes Sobol main + total + joint indices analytically from the PCE
   coefficients (Sudret 2008).
4. Runs all four [RFC006](readmes/RFC006.md) aggregation strategies:
   uniform, by generation, by lineage seed, and growth-stratified
   (θ = normalized log dry-mass).
5. Reports per-output training and held-out test relative errors, plus a Rich
   terminal report ranking Sobol indices per strategy.
6. Exports a dashboard-ready artifact directory under `./uq_results/`.

### Remote execution via SMS-API

If you don't have a local vEcoli checkout, sampling can run against
the **SMS-API** — a REST API that runs vEcoli on AWS Batch and returns
cd1 analysis TSVs (transcriptomics, proteomics, fluxomics, metabolomics,
higher-order properties).

```bash
# Inspect a completed simulation's cd1 outputs
uv run uq fetch 48 --api-url http://localhost:8080

# Remote sampling: PCRV sampling is local, simulation runs on AWS Batch
uv run uq sample /path/to/simData.cPickle \
    --api-url http://localhost:8080 \
    --simulator-id 11 \
    --n-samples 20 \
    --observables transcriptome \
    --observables proteome
```

Stage 2 (`quantify`) is identical regardless of execution mode.

### Cross-condition GSA

Compare sensitivity rankings across multiple growth conditions:

```bash
uv run uq sample /path/to/simData.cPickle \
    --conditions vecoli_m9_glucose_minus_aas \
    --conditions vecoli_m9_glucose_plus_aas \
    --n-samples 20

uv run uq quantify /path/to/simData.cPickle
# Auto-detects multi-condition cache → prints cross-condition comparison
```

### Other clients

All clients wrap the same two-stage workflow — pick your surface:

```bash
uv run uq report                                  # self-contained HTML report (auto-generated by quantify)
uv run uq show-config /path/to/simData.cPickle    # preview the vEcoli config JSON
uv run uq tui                                     # Textual terminal dashboard with live progress
uv run uq gui                                     # marimo browser notebook
uv run uq dashboard                               # tkinter DAW-style result explorer
uv run uq fetch 48                                # inspect SMS-API simulation outputs
uv run uq compare ./results_a ./results_b         # side-by-side Sobol comparison
uv run uq export-figures                          # publication-ready PDFs + LaTeX
uv run uq suggest-experiment                      # identify max-uncertainty region
uv run uq help                                    # list all commands
```

Trailing `help` works everywhere — `uq sample help`, `uq quantify help`,
`uq fetch help` all show the same output as `--help`.

Full per-flag documentation lives in [`docs/cli_reference.rst`](docs/cli_reference.rst).

---

## Methodology

### What problem are we solving?

vEcoli is a whole-cell simulator with hundreds of physiologically-meaningful
parameters. A single run is expensive, stochastic, and produces high-dimensional
time-resolved output. The scientific question we want to answer is:

> *Which parameters drive the variance in observable Y, and by how much?*

This is **forward uncertainty quantification** — propagating a prior over inputs
through the model and decomposing the resulting output variance by input.

### Polynomial Chaos Expansion + Sobol

We approximate the map `x → Y(x)` with a **Polynomial Chaos Expansion (PCE)**:

```
Ŷ(ξ) = Σ_α c_α Φ_α(ξ)
```

where `ξ ∈ [-1, 1]^d` is the *germ* (the random variable on which the basis is
orthogonal), `Φ_α` are multivariate Legendre polynomials indexed by multi-indices
`α` with `|α| ≤ p`, and `c_α` are fit coefficients.

Legendre is the right basis because we assume uniform priors on each bounded
parameter — Legendre polynomials are exactly the orthogonal family under the
uniform measure on `[-1, 1]`. The affine map between the germ and physical space
is encoded as a PyTUQ `PCRV` of order 1.

Once the PCE is fit, the **Sobol sensitivity indices** fall out analytically from
the coefficients (Sudret 2008). Because the basis is orthonormal under the input
measure, variance decomposes as a diagonal sum:

```
Var[Ŷ] = Σ_{|α|≥1} c_α² ‖Φ_α‖²
```

and the first-order and total-order Sobol indices for parameter `i` are partial
sums of this series over the multi-indices that touch (resp. include) dimension
`i`. No extra model evaluations are required for sensitivity — the entire Sobol
table is a function of the fit coefficients.

### Why PyTUQ?

[PyTUQ](https://sandialabs.github.io/pytuq/) (Sandia National Labs) is the
reference implementation of the UQPC five-step workflow. Its `pytuq.rv.pcrv.PCRV`
class gives us:

- The germ measure and its sampler (`sampleGerm` → no hand-rolled Monte Carlo / LHS)
- The multivariate Legendre basis and multi-indices (`get_mi`, `evalBases`)
- Three drop-in regression backends in `pytuq.lreg` — `lsq` (least squares),
  `bcs` (Bayesian Compressed Sensing, sparse), and `anl` (analytical Bayesian)
- Analytical Sobol computation via `PCRV.computeSens` / `computeTotSens` /
  `computeJointSens`

These are the exact primitives Sandia's `uq_pc.py` script composes, so our
`uq/workflow.py` is essentially a thin, vEcoli-specific port. Every step in
`run_uqpc` is annotated with the corresponding block from `uq_pc.py`.

Using PyTUQ directly (rather than re-implementing the PCE algebra) means:

- **Correctness** — Sudret-style sensitivity analysis is notoriously easy to get
  wrong; PyTUQ's PCRV keeps basis normalization, multi-index bookkeeping, and
  variance decomposition consistent.
- **Flexibility** — swapping `lsq` for `bcs` to handle sparse regimes
  (`N < |α|`) is a one-flag change.
- **Leverage** — any improvement upstream (new sampling schemes, sparse
  regression refinements) is an upgrade for us too.

### Why vEcoli's variants API?

vEcoli ships a canonical way to vary simulation parameters: the
[variants system](https://covertlab.github.io/vEcoli/workflows.html#variants).
Each variant is a Python function (`apply_variant`) that receives a
`SimulationDataEcoli` object and a parameter dict, and returns a mutated copy.
vEcoli's workflow runner automatically materializes, pickles, and schedules one
simulation per variant through Nextflow.

For uqEcoli, every PCE sample becomes one variant via the upstream
`ecoli.variants.sim_data_setattr` function, which walks a dot-path
(`process.translation.basal_elongation_rate`, `mass.cell_dry_mass_fraction`, …)
and writes the sampled value directly onto `sim_data`. The *entire* simulation
step — from Parca-skipping to Parquet emission — runs through vEcoli's own
`runscripts/workflow.py`. This repo emits a workflow config and reads the
resulting hive-partitioned Parquet; everything in between is upstream code.

The upshot: we don't reimplement anything. PyTUQ provides the sampling and the
math, vEcoli provides the simulator and the orchestration, and `uqEcoli` provides
the ergonomic glue — a two-command CLI, a TUI, a marimo GUI, and a tkinter
dashboard — over the resulting `PrecomputedCache`.

### Aggregation strategies (RFC006 §3)

Whole-cell variance has several sources: stochastic gene expression, lineage-seed
noise, transient convergence toward steady state, and cell-cycle progress. A
single population-averaged Sobol table hides all of these, so `quantify` runs
the same PCE/Sobol pipeline four times on four different aggregations of the
cached timeseries:

1. **Uniform** — bulk mean over all cells and timesteps (baseline).
2. **By generation** — one PCE per cell generation, controlling for transient
   dynamics in early generations.
3. **By lineage seed** — one PCE per lineage seed, controlling for stochastic
   seed-to-seed variance.
4. **Growth-stratified** — one PCE per cell-cycle bin, with cell-cycle progress
   defined as `θ = [log m(t) − log m_birth] / [log m_div − log m_birth]`
   (a monotonic surrogate that avoids any spectral decomposition).

All four share the same cached `(X, Y_timeseries)`; they only differ in how
`Y` is aggregated before being handed to `run_uqpc`.

### Further reading

- [`SAMPLING.md`](SAMPLING.md) — end-to-end trace of how `uq sample` uses the
  vEcoli variants API, with file:line citations and an ownership table.
- [`docs/tutorial_workflow.rst`](docs/tutorial_workflow.rst) — the full UQPC
  walkthrough with the Legendre/PCE math spelled out.
- [`docs/cli_reference.rst`](docs/cli_reference.rst) — every flag on every
  client.
- [`readmes/RFC006.md`](readmes/RFC006.md) — the original requirements document.
- Sudret, B. (2008). *Global sensitivity analysis using polynomial chaos
  expansions*. Reliability Engineering & System Safety 93(7), 964–979.
- Macklin, D. N. *et al.* (2020). Whole-cell model of *E. coli*. **Science** 369.


## Why a DAW format?

The dashboard (`uv run uq dashboard`) does something no other tool in the
vEcoli ecosystem does: it lets a biologist **ask "what if" questions about
sim_data parameters and get instant answers in physical units** — without
running a single simulation.

Before this dashboard, the workflow was: guess parameters → run vEcoli
(minutes to hours) → inspect Parquet output → repeat.  Now it's: drag a
slider → see `Dry mass = 345.2 fg (+5.1%)` and watch the cell-cycle profile
bend in real time.

### What makes it useful

1. **Searchable observable selector with units + baseline delta** — select any
   of the cd1 observables (5 mass/growth properties, 87 exchange fluxes,
   4,345 genes, 4,309 proteins, or 2,797 reaction fluxes depending on which
   `--observables` presets were used at sampling time).  A search-as-you-type
   filter handles thousands of entries.  The readout shows the selected
   observable in physical units with % deviation from wild-type baseline:
   `Growth rate = 0.88 hr⁻¹ (-3.2%)`.

2. **Predicted observable profile** — per-cell-cycle-stage PCE prediction for
   the selected observable.  X-axis = θ (growth progress, birth → division),
   Y-axis = predicted value.  All observables shown simultaneously as
   shape-guides; the selected one is bold with its own Y-axis units and a
   baseline reference line.  When `(aggregate)` is selected, all observables
   are shown as % deviation from baseline on a common axis.

3. **Sensitivity spectrogram with tracking dots** — the RFC006 strategy-4
   Sobol heatmap (parameters × θ-stages) with live-updating position dots
   that follow the current slider values.  Shows the user where they are
   in the sensitivity landscape.

4. **PCE response curves** — per-parameter sweep curves with draggable
   markers, baseline reference line, and the selected observable's label
   and units on the Y-axis.

5. **Target mode** — type a desired observable value, click *Find*, and the
   dashboard solves `min_x (Ŷ(x) − target)²` via L-BFGS-B on the PCE
   surrogate.  The sliders snap to the found parameters instantly.  This is
   the reverse of exploration: "I want growth rate = 0.95 hr⁻¹ — what
   parameters produce that?"

6. **Patch save/load** — save named parameter configurations as JSON, recall
   them later, share with colleagues.

### Per-gene / per-protein observable selection

When the pipeline is run with `--observables transcriptome` (4,345 mRNA
cistron counts) or `--observables proteome` (4,309 monomer counts), every
gene or protein becomes a selectable observable in the dashboard.  The
search bar filters the list instantly — type a gene name substring and
click to select.  The response curves, predicted profile, and readout all
switch to show that specific gene's PCE surrogate.

This lets a biologist ask: *"which sim_data parameters drive variance in
this specific gene's expression across the cell cycle?"* — and get an
interactive, slider-driven answer.  No other UQ dashboard provides this.

```bash
# Example: transcriptome run
uv run uq sample /path/to/simData.cPickle \
    --observables transcriptome --n-samples 50
uv run uq quantify /path/to/simData.cPickle
uv run uq dashboard
```

<div style="display:flex;justify-content:center;margin-bottom:20px;">
    <div style="display:inline-flex;align-items:center;gap:0;">
      <div style="
        border:2px solid #ff3366;
        border-radius:50px;
        padding:0.7rem 2rem;
        background:linear-gradient(135deg, rgba(233,30,144,0.08), rgba(0,229,255,0.05));
        text-align:center;
        min-width:320px;
      ">
        <div style="
          font-family:'Courier New',monospace;
          font-weight:900;
          font-size:1.4rem;
          letter-spacing:0.15em;
          background:linear-gradient(90deg, #ff3366, #00f0ff);
          -webkit-background-clip:text;
          -webkit-text-fill-color:transparent;
        ">&#x1F9EC; ATLANTIS &#x1F9EC;</div>
        <div style="
          font-family:'Courier New',monospace;
          font-size:0.7rem;
          color:#00f0ff;
          letter-spacing:0.1em;
          margin-top:0.2rem;
        ">whole-cell simulation platform</div>
      </div>
      <svg width="90" height="60" viewBox="0 0 90 60" style="margin-left:-2px;">
        <path d="M0,30 C15,15 25,45 40,30 S60,15 75,30 S85,42 90,35" stroke="#ff3366" fill="none" stroke-width="2" opacity="0.7"/>
        <path d="M0,25 C12,10 22,40 35,25 S55,10 70,25 S82,38 88,30" stroke="#00f0ff" fill="none" stroke-width="1.5" opacity="0.5"/>
        <path d="M0,35 C18,20 28,50 45,35 S65,20 80,35 S88,48 90,40" stroke="#ffd700" fill="none" stroke-width="1.5" opacity="0.5"/>
      </svg>
    </div>
</div>