# Tutorial: the `uq sample → uq quantify` pipeline

A fine-grained walkthrough of every step in the two-stage CLI pipeline,
cross-referenced to **RFC006** ([`readmes/start/tools/RFC006.md`](readmes/start/tools/RFC006.md))
and the **PyTUQ UQPC workflow** ([sandialabs.github.io/pytuq/apps/uqpc.html](https://sandialabs.github.io/pytuq/apps/uqpc.html)).

```
  uq sample                              uq quantify
  ┌──────────────────────────────────┐   ┌───────────────────────────────────┐
  │ UQPC 1: bounds → Legendre PCRV  │   │ UQPC 4: pc_fit (lsq/bcs/anl)    │
  │ UQPC 2: PCRV.sampleGerm()       │──▶│         → PCRV + Sobol            │
  │ UQPC 3: vEcoli workflow.py      │   │ UQPC 5: relative errors, export  │
  │         (sim_data_setattr)       │   │         (all 4 RFC006 strategies) │
  └──────────────────────────────────┘   └───────────────────────────────────┘
```

---

## Quick reference

```bash
# Stage 0 — size the run before committing compute (pure arithmetic, no vEcoli)
uv run uq plan --budget 1000 \
    --params 6 --polynomial-order 2 \
    --noise-replicates 4 --generations 4

# Stage 1 — run vEcoli simulations + cache
uv run uq sample /path/to/simData.cPickle \
    --cache-dir ./uq_cache \
    --n-samples 50 --n-test 10 \
    --generations 2 --n-init-sims 4

# Stage 2 — PCE fit + Sobol decomposition (no vEcoli calls)
# Auto-generates report.html in the export directory
# --bootstrap 200 adds empirical 95% CIs on Sobol indices
uv run uq quantify /path/to/simData.cPickle \
    --cache-dir ./uq_cache \
    --export-path ./uq_results \
    --polynomial-order 2 --regression lsq --bootstrap 200

# Regenerate the HTML report standalone (if needed)
uv run uq report --results-path ./uq_results
```

---

## Stage 0: `uq plan`  (size the run before committing compute)

PCE adequacy is governed by two ratios: `n_samples / basis_size` (does
the LSQ fit have enough rows?) and `n_init_sims` (≥ 2 to estimate the
noise floor). `uq plan` solves the bookkeeping problem before you spend
any vEcoli cycles:

```bash
uv run uq plan --budget 1000 \
    --params 6 --polynomial-order 2 \
    --noise-replicates 4 --generations 4
```

For a 1000-run budget, 6-param order-2 PCE, the recommended config is
`n_samples=62, n_init_sims=4, generations=4` (adequacy ratio 2.21×, ok).
The table also surfaces two alternatives:

- **halve generations** → `n_samples=125, n_init_sims=4, generations=2`
  (2× the PCE rows, half the cell-cycle coverage)
- **single replicate** → `n_samples=250, n_init_sims=1, generations=4`
  (4× the PCE rows, but the Triage 5 noise floor becomes unestimable)

The `status` column (`ok` / `marginal` / `underdetermined`) and the
`noise floor?` column tell you whether the recommended config is honest
to interpret without further caveats. If `underdetermined`, raise the
budget, drop `--polynomial-order`, or plan to use `--regression bcs`.

---

## Stage 1: `uq sample`  (UQPC steps 1-3)

### Step 1 — setup inputs

| What happens | Code | RFC / UQPC reference |
|---|---|---|
| Load `simData.cPickle` via `ParameterDataset` | `uq/cli.py:114` | RFC006 §4, activity 1: "Identify the scientifically most relevant input and output variables" |
| Build `XSpaceVecoli` from `SimDataParameter` specs | `libuq/pipeline/param_loader.py` | RFC006 §4, activity 1: inputs include "vio pathway presence, mecillinam condition, gene knockouts" (generalised to any scalar sim_data attribute) |
| Construct input `PCRV` (Legendre basis, order 1) | `uq/workflow.py::_setup_input_pc` | UQPC step 1 (`--pdom`): "Setup Inputs — define uncertain input parameters through marginal distributions" |

<details>
<summary><b>UQPC reference → our code (step 1)</b></summary>

**What `uq_pc.py` does** (equivalent to `python uq_pc.py --pdom bounds.txt --pctype LU --pcord 1`):

1. Reads a two-column parameter domain file (`--pdom`) with `[lb, ub]` per parameter.
2. Computes midpoints and half-ranges, builds the PC coefficient matrix
   `pcf_all` (row 0 = midpoints, rows 1..d = diag(half_ranges)).
3. Constructs a `PCRV(ndim, npc, "LU", mi=mi, cfs=pcf_all.T)`.

**Our code** (`uq/workflow.py:110-145`, function `_setup_input_pc`):

```python
# Exactly the same construction — bounds → midpoints + half_ranges → PCRV
midpoints   = 0.5 * (bounds[:, 1] + bounds[:, 0])
half_ranges = 0.5 * (bounds[:, 1] - bounds[:, 0])
pcf_all     = np.vstack((midpoints, np.diag(half_ranges)))

mi = get_mi(in_pcord, in_pcdim)                          # pytuq.utils.mindex.get_mi
pc = PCRV(in_pcdim, n_params, "LU", mi=mi, cfs=pcf_all.T)  # pytuq.rv.pcrv.PCRV
```

The flag mapping is:

| `uq_pc.py` flag | Our equivalent | Value |
|---|---|---|
| `--pdom bounds.txt` | `bounds = np.array(param_space.parameter_bounds)` | `(n_params, 2)` array |
| `--pctype LU` | hard-coded `"LU"` in `_setup_input_pc` | Legendre — correct for uniform priors |
| `--pcord 1` | hard-coded `in_pcord = 1` | order 1 = affine map from germ → physical |
| `--pcdim` | `n_params = bounds.shape[0]` | auto-detected from bounds |

</details>

**Default parameters** (6; `DEFAULT_SIM_DATA_PARAMETERS`):

| sim_data dot-path | Physical meaning |
|---|---|
| `process.transcription.fraction_active_rnap_free` | ppGpp-free RNAP fraction |
| `process.transcription.fraction_active_rnap_bound` | ppGpp-bound RNAP fraction |
| `process.translation.basal_elongation_rate` | Ribosome speed (aa/s) |
| `process.metabolism.kinetic_objective_weight` | FBA kinetic vs homeostatic objective |
| `process.metabolism.secretion_penalty_coeff` | Penalty on overflow secretion |
| `mass.cell_dry_mass_fraction` | Dry mass fraction |

Custom parameters: pass `--params-file params.json` (see `examples/uq_artifacts/params/params_demo.json`).

**Generating a `--params-file` from scratch** (`uq create mutations`):

```bash
uv run uq create mutations /path/to/simData.cPickle \
    --output ./my_params.json \
    --perturbation-scheme examples/perturbation_schemes/pct_around_baseline.py
```

The `--perturbation-scheme` argument loads a Python file or dotted module
that defines either a static `PARAMETERS: list[SimDataParameter]` or a
dynamic `build_parameters(sim_data, n_samples=0, seed=42) -> list[SimDataParameter]`.
When omitted, `DEFAULT_SIM_DATA_PARAMETERS` (the six canonical knobs above)
is used. **Disclaimer**: that default is vEcoli-domain-pragmatic, not
UQ-methodology-canonical — PyTUQ/Sudret/Saltelli prescribe only the form
of the prior (uniform on a bounded interval) and are silent on bounds
selection. See `docs/cli_reference.rst` § "On default bounds —
methodology silence" for the full framing.

**Observable presets** (`--observables`, composable):

The `sample` command extracts outputs using presets that mirror the
**cd1 analysis modules** (Vegas/Bermuda CD1 deliverables). Verified on
real vEcoli Parquet data:

| Preset | cd1 module | Features | Description |
|---|---|---|---|
| `mass` | cd1_higher_order_properties (raw) | 5 | dry_mass, cell_mass, volume, growth, instantaneous_growth_rate |
| `higher_order` | cd1_higher_order_properties (derived) | 6 | doubling time (hours), growth rate (1/h), DNA/RNA/dry mass fractions, cell volume |
| `exchange_fluxes` | cd1_exchange_fluxes | 87 | external metabolite fluxes (glucose uptake, acetate secretion, etc.) |
| `transcriptome` | cd1_transcriptomics | 4,345 | mRNA cistron counts per gene |
| `proteome` | cd1_proteomics | 4,309 | monomer counts per protein |
| `fluxome` | cd1_fluxomics | 2,797 | base reaction fluxes (normalized by dry mass) |

Example — run the same outputs the CD1 reports used:

```bash
uv run uq sample /path/to/simData.cPickle \
    --observables higher_order \
    --observables exchange_fluxes \
    --observables transcriptome \
    --observables proteome \
    --generation-lower-bound 2
```

**Generation filtering** (`--generation-lower-bound N`):

Skips the first N generations before aggregating Y, matching the cd1
`generation_lower_bound` parameter.  This focuses the sensitivity
analysis on steady-state growth, excluding transient initialization
dynamics.

---

### Step 2 — generate samples

| What happens | Code | RFC / UQPC reference |
|---|---|---|
| Draw `n_samples` germ-space realizations via `PCRV.sampleGerm()` | `uq/cli.py:128-131` | UQPC step 2 (`--sampl rand`): "Draw training realizations from the input PC" |
| Map to physical space via `PCRV.evalPC()` | `uq/cli.py:131` | UQPC step 2: writes `ptrain.txt` / `qtrain.txt` |
| Optionally draw `n_test` held-out validation samples | `uq/cli.py:134-138` | UQPC `--ntst`: "Testing realizations (optional)" |

<details>
<summary><b>UQPC reference → our code (step 2)</b></summary>

**What `uq_pc.py` does** (equivalent to `python uq_pc.py --sampl rand --nqd 50 --ntst 10 --seed 42`):

1. `germ_train = pc.sampleGerm(nqd)` — draw training germs from U(−1, 1)^d.
2. `X_train = pc.evalPC(germ_train)` — map germ → physical.
3. Saves `qtrain.txt` (germ) and `ptrain.txt` (physical).
4. If `ntst > 0`: same for `germ_test`, saves `qtest.txt`, `ptest.txt`.

**Our code** (`uq/cli.py:128-138`):

```python
input_pc, _, _ = _setup_input_pc(bounds)

np.random.seed(seed)
germ_train = input_pc.sampleGerm(n_samples)   # ← pc.sampleGerm(nqd)
X_train    = input_pc.evalPC(germ_train)       # ← pc.evalPC(germ_train)

if n_test > 0:
    np.random.seed(seed + 1)
    germ_test = input_pc.sampleGerm(n_test)    # ← UQPC --ntst
    X_test    = input_pc.evalPC(germ_test)
```

The flag mapping:

| `uq_pc.py` flag | Our equivalent | Value |
|---|---|---|
| `--sampl rand` | always random (no quadrature path) | `PCRV.sampleGerm` |
| `--nqd 50` | `--n-samples 50` | number of training draws |
| `--ntst 10` | `--n-test 10` | number of held-out validation draws |
| `--seed 42` | `--seed 42` | RNG seed for `sampleGerm` |
| `ptrain.txt` | `X.npy` in cache | physical-space training samples |
| `qtrain.txt` | `germ_train.npy` in cache | germ-space training samples |
| `ptest.txt` | `X_test.npy` in cache | held-out physical samples |

</details>

---

### Step 3 — evaluate the model (vEcoli)

| What happens | Code | RFC / UQPC reference |
|---|---|---|
| Encode each sample row as a `sim_data_setattr` variant | `uq/vecoli_config.py::_build_variants_from_samples` | RFC006 §4, activity 3: "Implement input→output wrapper functions that can be called from numerical libraries" |
| Build a vEcoli workflow config JSON | `uq/vecoli_config.py::_build_config` | vEcoli variants API ([covertlab.github.io/vEcoli/workflows.html#variants](https://covertlab.github.io/vEcoli/workflows.html#variants)) |
| Spawn `runscripts/workflow.py --config ...` as a subprocess | `uq/cli.py:175-182` | UQPC step 3 (`--regime online_bb`): "Evaluate the model" |
| Collect hive-partitioned Parquet via Polars | `uq/vecoli_config.py::_collect_variant_timeseries` | RFC006 §4, activity 2: "Enable output of relevant variables" |

<details>
<summary><b>UQPC reference → our code (step 3)</b></summary>

**What `uq_pc.py` does** (equivalent to `python uq_pc.py --regime online_bb`):

1. Reads training samples from `ptrain.txt`.
2. For each sample, calls external executable `model.x` and captures stdout → `ytrain.txt`.
3. If test samples exist, evaluates them too → `ytest.txt`.

**Our code** — vEcoli replaces `model.x`:

```python
# Build the N-variant config using vEcoli's sim_data_setattr grammar
variants = _build_variants_from_samples(X_all, param_space._sim_data_parameters)
# → {"sim_data_setattr": {"mutations": {"value": [dict_per_sample, ...]}}}

config = _build_config(sim_data_path, output_dir, variants, ...)
# → standard vEcoli workflow JSON (emitter=parquet, generations=G, n_init_sims=S)

# Spawn vEcoli's own Nextflow runner as a subprocess
workflow_script = os.path.join(vecoli_root, "runscripts", "workflow.py")
cmd = [sys.executable, workflow_script, "--config", str(config_path)]
proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, ...)
```

The equivalent of UQPC's `ytrain.txt` is our `Y.npy` (time-averaged
observables per variant, collected from hive-partitioned Parquet).

| `uq_pc.py` concept | Our equivalent |
|---|---|
| `model.x` (external black-box) | `runscripts/workflow.py` (vEcoli, subprocess) |
| `ptrain.txt` → feed to model | `X_all` → encoded as `sim_data_setattr` mutations |
| `ytrain.txt` (model output) | `Y.npy` (collected from Parquet) |
| `ytest.txt` | `Y_test.npy` (split from same workflow run) |

**Key architectural point:** we concatenate training + held-out samples
into a *single* variant list so vEcoli runs one workflow.  The train/test
split is applied after collection (variant indices `1..N` = train,
`N+1..N+N_test` = test).

</details>

**Observables extracted** are determined by the `--observables` presets
(see table above).  The default `mass` preset reads 5 scalar columns;
composing all cd1 presets produces ~11,500 features covering the full
multi-omics output space reported for CD1 (Vegas/Bermuda).

These satisfy RFC006 §4 activity 1: *"Outputs: transcriptome, proteome,
metabolic fluxes (particularly exchange fluxes), and higher order
properties."*

**What gets cached to `./uq_cache/`:**

```
uq_cache/
├── X.npy                        # (n_samples, n_params) physical space
├── Y.npy                        # (n_samples, n_outputs) time-averaged
├── germ_train.npy               # (n_samples, n_params) germ space
├── X_test.npy                   # (n_test, n_params) held-out validation
├── Y_test.npy                   # (n_test, n_outputs) held-out outputs
├── metadata.json                # parameter names, bounds, observables
└── timeseries/
    ├── sample_0000.npy          # (n_timesteps, n_obs) raw timeseries
    ├── sample_0000_meta.npz     # generation + lineage_seed per row
    └── ...
```

---

## Stage 2: `uq quantify`  (UQPC steps 4-5)

### Step 4 — build the PCE surrogate

| What happens | Code | RFC / UQPC reference |
|---|---|---|
| Build multi-index for output order `p` | `uq/workflow.py::_fit_surrogate` | UQPC step 4 (`--outord`): "Build the PC Surrogate" |
| Construct output PCRV (Legendre "LU") | `uq/workflow.py:316` | UQPC step 4: "PCE, PCRV, regression classes in lreg module" |
| Evaluate basis matrix at germ samples | `pcrv.evalBases(germ_train, 0)` | UQPC: design matrix `A ∈ ℝ^{N × |α|}` |
| Per-output: fit coefficients via `lsq`/`bcs`/`anl` | `uq/workflow.py:326-341` | UQPC step 4 (`--method`): "lsq, bcs, anl" |
| Sync multi-indices + coefficients → PCRV | `pcrv.setMiCfs(...)` | UQPC: "Fitted surrogate PCRV object" |

<details>
<summary><b>UQPC reference → our code (step 4)</b></summary>

**What `uq_pc.py` does** (equivalent to `python uq_pc.py --method lsq --outord 2`):

1. Reads training germs `qtrain.txt` and outputs `ytrain.txt`.
2. Builds multi-index: `mi = get_mi(outord, ndim)`.
3. Constructs output PCRV: `pcrv = PCRV(n_out, ndim, "LU", mi=mi)`.
4. Evaluates basis matrix: `Amat = pcrv.evalBases(qtrain, 0)`.
5. Per output column `j`:
   - Instantiates regressor: `reg = lsq()` (or `bcs(eta=tol)`, or `anl()`).
   - Fits: `reg.fita(Amat, ytrain[:, j])`.
   - Records `reg.used` (active basis indices) and `reg.cf` (coefficients).
6. Syncs into PCRV: `pcrv.setMiCfs(mindices_list, cfs_list)`.
7. Sets evaluation function: `pcrv.setFunction()`.
8. Pickles the result → `results.pk`.

**Our code** (`uq/workflow.py:269-347`, function `_fit_surrogate`) — identical logic:

```python
# Step 1-2: Multi-index + PCRV
mindex = get_mi(polynomial_order, n_dim)            # ← get_mi(outord, ndim)
pcrv   = PCRV(n_out, n_dim, "LU", mi=mindex)        # ← PCRV(n_out, ndim, "LU", mi=mi)

# Step 3: Basis matrix
Amat = pcrv.evalBases(germ_train, 0)                 # ← pcrv.evalBases(qtrain, 0)

# Step 4: Per-output fitting
for j in range(n_out):
    if regression == "bcs":
        lreg_obj = bcs(eta=tolerance)                # ← bcs(eta=tol)
    elif regression == "anl":
        lreg_obj = anl()                             # ← anl()
    else:
        lreg_obj = lsq()                             # ← lsq()

    lreg_obj.fita(Amat, Y_train[:, j])               # ← reg.fita(Amat, ytrain[:, j])
    mindices_list.append(mindex[lreg_obj.used, :])   # ← active basis rows
    cfs_list.append(lreg_obj.cf)                     # ← fit coefficients

# Step 5-6: Sync coefficients and set evaluation function
pcrv.setMiCfs(mindices_list, cfs_list)               # ← pcrv.setMiCfs(...)
pcrv.setFunction()                                    # ← pcrv.setFunction()
```

The flag mapping:

| `uq_pc.py` flag | Our equivalent | Value |
|---|---|---|
| `--method lsq` | `--regression lsq` | `pytuq.lreg.lreg.lsq` |
| `--method bcs` | `--regression bcs` | `pytuq.lreg.bcs.bcs` |
| `--method anl` | `--regression anl` | `pytuq.lreg.anl.anl` |
| `--outord 2` | `--polynomial-order 2` | output PCE order |
| `--tol 1e-3` | `--tol 1e-3` | BCS sparsity tolerance |
| `results.pk` | `QuantifyResult.export(...)` | our export is a directory, not a pickle |

</details>

The PCE approximates each observable output `Y_j` as:

```
Ŷ_j(ξ) = Σ_{|α|≤p} c_{j,α} · Φ_α(ξ)
```

where `Φ_α` are multivariate Legendre polynomials and `c_{j,α}` are fit
coefficients.  `|α|` is the total polynomial degree; for 6 parameters and
order 2 there are `C(8,2) = 28` basis terms.

**Regression backends** (`--regression`):

| Flag | PyTUQ class | When to use |
|---|---|---|
| `lsq` | `pytuq.lreg.lreg.lsq` | Default. OLS. Need `n_samples > 2 × n_terms`. |
| `bcs` | `pytuq.lreg.bcs.bcs` | Sparse. Retains only significant terms. OK when `n_samples ≈ n_terms`. Sparsity tolerance: `--tol`. |
| `anl` | `pytuq.lreg.anl.anl` | Analytical Bayesian. Calibrated prediction variance. |

Rule of thumb: PCE terms = `C(n_params + order, order)`.
For 6 params, order 2 → 28 terms → need ≥ 56 samples for stable LSQ.

---

### Step 5 — post-processing: relative errors + Sobol

| What happens | Code | RFC / UQPC reference |
|---|---|---|
| Predict at training points, compute `‖Y − Ŷ‖₂ / ‖Y‖₂` per output | `uq/workflow.py::_compute_relative_errors` | UQPC step 5: "relative model-surrogate errors" |
| If held-out test data in cache, compute test relative errors | `uq/workflow.py:630-632` | UQPC `--ntst`: "Testing relative errors" |
| Compute Sobol main, total, and joint indices from PCRV | `uq/workflow.py::_compute_sobol` | UQPC step 5: "Sobol main and total indices" |
| Variance-weight across outputs for multi-output models | `uq/workflow.py:444-462` | Standard multi-output extension (Sudret 2008) |

<details>
<summary><b>UQPC reference → our code (step 5)</b></summary>

**What `uq_pc.py` does:**

1. Predicts at training points: `Y_pc = pcrv.function(qtrain)`.
2. Computes per-output relative error: `‖ytrain − Y_pc‖₂ / ‖ytrain‖₂`.
3. If test data exists: same at test points.
4. Computes Sobol indices from PCRV coefficients:
   - `allsens_main  = pcrv.computeSens()`     — first-order S_i
   - `allsens_total = pcrv.computeTotSens()`  — total-order S_Ti
   - `allsens_joint = pcrv.computeJointSens()` — second-order S_ij
5. Pickles everything into `results.pk`.

**Our code:**

```python
# Predict at training points
Y_train_pc = output_pcrv.function(germ_train)         # ← pcrv.function(qtrain)

# Relative errors (UQPC step 5)
norms = np.linalg.norm(Y_true, axis=0)
relerr = np.linalg.norm(Y_true - Y_pred, axis=0) / norms  # per-output ε

# If test data from --n-test:
if Y_test is not None and Y_test_pc is not None:
    relerr_test = _compute_relative_errors(Y_test, Y_test_pc)

# Sobol indices — exactly the same PCRV methods
allsens_main  = output_pcrv.computeSens()              # ← pcrv.computeSens()
allsens_total = output_pcrv.computeTotSens()           # ← pcrv.computeTotSens()
allsens_joint = output_pcrv.computeJointSens()         # ← pcrv.computeJointSens()
```

The only addition beyond the reference `uq_pc.py` is **variance-weighted
aggregation across outputs** for multi-output models (our Y has 4 mass/growth
observables, not 1):

```python
# Weight each output's Sobol indices by its fraction of total variance
output_vars = np.var(Y_train, axis=0)
weights = output_vars / output_vars.sum()
first_order = sum(weights[j] * allsens_main[j] for j in range(n_outputs))
total_order = sum(weights[j] * allsens_total[j] for j in range(n_outputs))
```

| `uq_pc.py` output | Our equivalent |
|---|---|
| `results.pk["pcrv"]` | `UQPCResult.pcrv` |
| `results.pk["sensitivities"]` | `UQPCResult.sobol` (as `SobolIndices` dataclass) |
| `results.pk["relerr_train"]` | `UQPCResult.relerr_train` |
| `results.pk["relerr_test"]` | `UQPCResult.relerr_test` |
| `plot.py sens total` | `uq quantify` Rich report's Sobol table |

</details>

Sobol indices are computed **analytically from PCE coefficients** — no
additional model evaluations.  Because the Legendre basis is orthonormal
under the input measure, variance decomposes as:

```
Var[Ŷ_j] = Σ_{|α|≥1} c²_{j,α} · ‖Φ_α‖²

S_i   = (1/Var) Σ_{α ∈ A_i}    c²_α ‖Φ_α‖²     (first-order)
S_Ti  = (1/Var) Σ_{α: α_i > 0} c²_α ‖Φ_α‖²     (total-order)
```

where `A_i = {α : α_i > 0, α_{k≠i} = 0}`.

> **Reference:** Sudret, B. (2008). *Global sensitivity analysis using
> polynomial chaos expansions.* Reliability Engineering & System Safety
> 93(7), 964-979.

#### Honesty diagnostics added in §25-D triage

Single-fit Sobol numbers can be misleading on their own — three things
silently bias the interpretation. `quantify` now prints four additional
panels that surface what was being hidden:

**DESIGN QUALITY @ order=N** — re-runs the count + κ(A) check at the
*actual* `--polynomial-order` you passed to `quantify`. The sample-time
check assumed order 2; if you pass `--polynomial-order 3` the basis
size jumps from 28 to 84 (for 6 params) and a previously-ok design can
become underdetermined silently. Status is `ok` / `marginal` /
`underdetermined`; the κ(A) line below it gives the PyTUQ-native joint
geometry measure (Triage 2 / Gap 4a).

**SURROGATE QUALITY** — train + (test or CV) relative errors per output.

| Column | Source | When it shows |
|---|---|---|
| `TRAIN` | `‖Y − Ŷ‖₂ / ‖Y‖₂` at training rows | always |
| `TEST` | same on held-out `X_test/Y_test` | when `--n-test > 0` was used at sample time |
| `CV (5-fold)` | k-fold CV using PyTUQ regression backends per fold | when no test set is present and `N ≥ 2.5 × basis_size` (Triage 3 / Gap 3) |

Training error alone is uninformative — PCE fits training rows nearly
exactly when N ≈ basis_size. The CV column is the BYO-mode substitute
for the lost `--n-test` generalization check.

**NOISE FLOOR // ALEATORIC vs EPISTEMIC** — ANOVA-style separation of
total Y variance into:

```
Var[Y]_total  =  Var_between_variants  +  Var_within_variants
                  ↑ epistemic            ↑ aleatoric (noise floor)
                  ↑ PCE can explain      ↑ irreducible
```

Computed from vEcoli's `lineage_seed` replicate structure already in
the cache (Triage 5 / Gap 2). Verdict color-coded by mean signal
fraction:

| Mean signal | Verdict | Interpretation |
|---|---|---|
| ≥ 80% | dim | "Sobol indices interpret straightforwardly" |
| 50–80% | yellow | Sobol indices undershoot the per-parameter share of the *explainable* variance by ~1/η²× |
| < 50% | red | Aleatoric exceeds parameter signal — Sobol measures only η²·Var[Y] |

If `n_init_sims = 1` (the default), the panel says "not estimable —
rerun with `--n-init-sims 4`". This is structural: with one replicate
per variant there is no within-variant variance to estimate.

**Sobol CIs (`--bootstrap N`)** — standard non-parametric bootstrap
(Triage 4 / Gap 4b): resample `(X, Y)` rows with replacement, refit the
PCE via the same `--regression` backend, recompute Sobol via
`PCRV.computeSens` / `computeTotSens`, accumulate K samples and report
the 2.5–97.5 percentile interval per parameter. Sobol tables grow from
`S_Ti` to `S_Ti  [low, high]`. Typical N = 200; sub-second for
moderate output dimensions. Threaded through all four RFC006 strategies.

Without CIs you cannot tell whether `S_Ti = 0.42` for one parameter and
`S_Ti = 0.38` for another reflect a real difference or fit noise.

```python
# Bootstrap loop (uq/workflow.py::_bootstrap_sobol_cis)
for b in range(n_bootstrap):
    idx = rng.integers(0, n_samples, size=n_samples)        # resample with replacement
    pcrv_b.fit(germ[idx], Y[idx])                            # refit PCE
    main_samples[b], total_samples[b] = pcrv_b.computeSens(), pcrv_b.computeTotSens()
fo_ci = np.percentile(main_samples, [2.5, 97.5], axis=0).T   # per-param CI
```

---

### Two axes of variation: design conditions × UQ samples

Before the strategies, a one-paragraph framing.  `sim_data` can be
mutated along two mathematically distinct axes — and the pipeline has
separate flags for each:

| Axis | Type | How it's declared | What PCE does with it |
|------|------|-------------------|----------------------|
| **UQ** $P_{uq}$ | Continuous (continuous parameters with uniform priors on bounded intervals) | `--params-file` | Decomposes `Var[Y]` via Sudret Sobol |
| **Design** $P_{design}$ | Categorical / structural (knockouts, environments, timeline events) | `--design-config` (variant-level) or `--conditions` (parca-level) | Per-condition PCE + cross-condition rank stability |

When both axes are present, the framework runs an `M × N` grid
(M conditions × N UQ samples per condition, same X reused across
conditions for paired comparison).  **`P_design ⊥ P_uq` is a soundness
requirement, not a stylistic choice** — overlapping attr_paths cause
`sim_data_setattr` to silently overwrite the design's value, breaking
the per-condition Sobol interpretation.  The CLI surfaces this overlap
before any compute is committed.

The collapse rule: if `P_uq = P_design` (you want to perturb the same
attr_paths via both), there's only one axis structurally.  Drop into a
single-layer mode — BYO (`--variants-source base-config`) for
hand-picked values, default mode for PCRV-driven values.  Two-layer
`--design-config` earns its complexity only when the design axis is
genuinely irreducible to a continuous parameter.  Full decision table
+ subtle "finely-spaced design should be folded into UQ" guidance lives
in `docs/cli_reference.rst` § "Two axes of variation".

### How the 4 RFC006 strategies work

`quantify` runs steps 4-5 **four times** on four different aggregations of
the cached timeseries.  They share X; they differ only in how Y is computed
from the raw data.

| Strategy | RFC006 §1 requirement | What is collapsed to form Y | Y shape | Code entry point |
|---|---|---|---|---|
| 1: Uniform | "Uniformly across all simulated cells and times (baseline)" | seeds, generations, agents, timesteps → mean per variant | `(N, n_obs)` | `run_strategy1_uniform` |
| 2: By generation | "Stratified by generation (control of convergence towards steady-state growth)" | seeds, agents, timesteps → mean per variant **per generation** | `(N, n_obs)` × one PCE per gen | `run_strategy2_by_generation` |
| 3: By lineage seed | "Stratified by lineage seed (control of exogenous variance)" | generations, agents, timesteps → mean per variant **per seed** | `(N, n_obs)` × one PCE per seed | `run_strategy3_by_seed` |
| 4: Growth-stratified | "Stratified by cell cycle stage, according to a physiological variable" | timesteps binned by θ = normalised log(dry_mass) → per-stage means | `(N, n_bins × n_obs)` + per-stage PCE | `run_strategy4_growth_stratified` |

The `N`-variants axis is always the row dimension — that is what PCE
regresses over.  The strategies only change what is in the columns.

**Strategy 4's cell-cycle variable** is:

```
θ(t) = [log m(t) − log m_birth] / [log m_div − log m_birth]
```

A monotonic, model-free proxy for cell-cycle progress: θ = 0 at birth,
θ = 1 at division.  Timesteps are binned into `n_bins` uniform intervals
of θ.  This satisfies RFC006 §3: "the definition of a low-dimensional
(possibly scalar) cell cycle variable computed from omics variables …
deterministically binning simulation data into cell stages."

**How `--generations` and `--n-init-sims` interact with `--n-samples`:**

`--n-samples 50` controls one thing: how many points `PCRV.sampleGerm(50)`
draws.  Each becomes one `sim_data_setattr` variant.  PCE always sees 50
training rows.

`--generations 2 --n-init-sims 2` controls how many simulations vEcoli
runs **per variant**: `(50 + 1 baseline) × 2 seeds × 2 gens = 204 total
simulations`.  These extra simulations do not increase the PCE sample
count; they increase the statistical richness *within* each sample,
enabling strategies 2 and 3:

- Strategy 2: partitions each variant's timeseries rows by `generation`, computes
  per-generation means, fits a separate PCE per generation.  With `generations=1`
  there is only one group → strategy 2 = strategy 1.
- Strategy 3: same idea with `lineage_seed`.  With `n_init_sims=1` → strategy 3 = strategy 1.
- Strategy 4: bins by θ regardless of `generations`/`n_init_sims` — works even
  with `generations=1, n_init_sims=1`.

---

## RFC006 §4 activities → pipeline mapping

| RFC006 activity | Status | Pipeline component |
|---|---|---|
| **1.** Identify scientifically relevant I/O | ✅ | `DEFAULT_SIM_DATA_PARAMETERS` (6 params), 4 mass/growth observables; configurable via `--params-file` |
| **2.** Enable output via emitter | ✅ | vEcoli's Parquet emitter + hive partitioning; collected by `_collect_variant_timeseries` |
| **3.** Input→output wrapper functions | ✅ | `sim_data_setattr` variant function (vEcoli upstream) + vEcoli `workflow.py` subprocess |
| **4.** GSA for strategies 1-3 via PCE | ✅ | `run_strategy1_uniform`, `run_strategy2_by_generation`, `run_strategy3_by_seed` in `uq/workflow.py` |
| **5.** Report on representative simulations | ✅ | `uq quantify` Rich report + exported `uq_results.json` + dashboard |
| **6.** Cell-cycle stratification strategy | ✅ | θ = normalised log(dry_mass) in `_compute_growth_fraction` |
| **7.** Implement + apply cell-cycle GSA | ✅ | `run_strategy4_growth_stratified` — per-stage PCE + Sobol |

---

## PyTUQ UQPC workflow → pipeline mapping

| UQPC step | `uq_pc.py` flag | Pipeline function | Location |
|---|---|---|---|
| 1. Setup inputs | `--pdom`, `--pctype LU`, `--pcord 1` | `_setup_input_pc` → `PCRV(…, "LU", mi=get_mi(1, d), cfs=…)` | `uq/workflow.py:110-145` |
| 2. Generate samples | `--sampl rand`, `--nqd N`, `--ntst N_t`, `--seed` | `PCRV.sampleGerm(N)` → `PCRV.evalPC(germ)` | `uq/cli.py:128-138` |
| 3. Evaluate model | `--regime online_bb` | `subprocess.Popen(workflow.py)` + `sim_data_setattr` variants | `uq/cli.py:175-182` |
| 4. Build surrogate | `--method lsq\|bcs\|anl`, `--outord p`, `--tol` | `get_mi` → `PCRV.evalBases` → `lsq().fita` → `PCRV.setMiCfs` → `setFunction` | `uq/workflow.py:269-347` |
| 5. Post-process | — | `PCRV.function` → `relerr` + `computeSens/TotSens/JointSens` | `uq/workflow.py:383-469` |

---

## Interpreting the output

### Sobol indices

- **S_i (first-order):** fraction of output variance explained by parameter `i` alone.
- **S_Ti (total-order):** fraction explained by `i` plus all its interactions with other parameters.
- `Σ S_i ≈ 1` for additive models; `S_Ti − S_i` measures interaction strength.

### Surrogate quality

- **Training relative error** (`relerr_train`): `‖Y − Ŷ‖₂ / ‖Y‖₂` per output.
  Small = good fit.  Large = PCE order too low, or the model has discontinuities
  the smooth basis cannot capture.
- **Test relative error** (`relerr_test`, when `--n-test > 0`): same metric on
  held-out data.  Large test error with small training error = overfitting —
  reduce `--polynomial-order` or increase `--n-samples`.
- **CV relative error** (`relerr_cv`, when no test set): 5-fold cross-validation
  using PyTUQ regression backends per fold.  Replaces `relerr_test` when
  `--n-test 0` (default) or when BYO mode skipped the held-out draws.  Skipped
  automatically when `N < 2.5 × basis_size`.

### Sobol confidence intervals

- **Point estimate** (`S_Ti = 0.42`): always reported.  Single-fit Sobol from
  the PCE coefficients (Sudret 2008).
- **Empirical 95% CI** (`S_Ti  [low, high]`): printed when `--bootstrap N`
  is set on `uq quantify`.  Lets you compare two parameters' total-order
  indices honestly — non-overlapping CIs = the ranking is real, not fit noise.

### Noise floor

- **Signal fraction (η²_between)**: fraction of total Y variance attributable
  to parameter variation.  PCE Sobol indices are computed against this share.
- **Noise fraction (η²_within)**: aleatoric / irreducible.  Estimated from
  vEcoli's per-variant `lineage_seed` replicates.  Verdict color-coded by
  the mean signal fraction across observables.

### Regression choice

| Method | Flag | Best for |
|---|---|---|
| LSQ | `--regression lsq` | Default. Fast, exact on smooth polynomial data. `n_samples ≫ n_terms`. |
| BCS | `--regression bcs` | Sparse PCE. Retains only significant terms. OK when `n_samples ≈ n_terms`. Use `--tol` to tune sparsity. |
| ANL | `--regression anl` | Analytical Bayesian. Calibrated uncertainty on predictions. |

---

## Export structure

```
uq_results/
├── uq_results.json              # all strategies in a dashboard-compatible schema
├── population_surrogate/        # PCE coefficients + multi-indices (strategy 1)
│   ├── coefficients.npy
│   ├── multi_indices.npy
│   └── input_bounds.npy
├── population_sobol/            # strategy 1 Sobol npy
│   ├── first_order.npy
│   └── total_order.npy
├── generation_0_sobol/          # strategy 2
├── generation_1_sobol/
├── seed_0_sobol/                # strategy 3
├── growth_stage_0_sobol/        # strategy 4
├── growth_stage_1_sobol/
├── ...
└── growth_stratified_surrogate/ # strategy 4 combined PCE
```

`uq_results.json` is the single file the dashboard consumes.  It contains
the per-strategy Sobol indices as `{parameter_name: float}` dicts, plus
metadata (framework label, observable names, stage θ-ranges).

---

## Entry points

All four clients expose the same `uq.workflow.sample` → `uq.workflow.quantify`
pipeline in different presentation formats:

| Client | Command | Best for |
|---|---|---|
| CLI (Rich) | `uv run uq sample` / `uv run uq quantify` | Scripts, CI, headless runs |
| HTML Report | `uv run uq report` | Shareable, publication-ready results (auto-generated by quantify) |
| CLI (inspect) | `uv run uq show-config` | Preview the vEcoli config JSON |
| TUI (Textual) | `uv run uq tui` | Interactive terminal with live progress |
| GUI (marimo) | `uv run uq gui` | Reactive browser notebook |
| Dashboard (tkinter) | `uv run uq dashboard` | DAW-style result exploration |

---

## Verified end-to-end run

The following was run on 2026-04-13 and produced the artifacts in
`uq_cache_e2e/` and `uq_results_e2e/`:

```bash
# Preview config (no vEcoli execution)
uv run uq show-config sim_data/baseline/kb/simData.cPickle \
    --n-samples 10 --output-file ./uq_cache/preview_config.json

# Stage 1: sample (10 variants, 6 params, ~4.5 min on M1)
uv run uq sample sim_data/baseline/kb/simData.cPickle \
    --cache-dir ./uq_cache_e2e \
    --n-samples 10 --seed 42

# Stage 2: quantify (instant — all matrix algebra)
uv run uq quantify sim_data/baseline/kb/simData.cPickle \
    --cache-dir ./uq_cache_e2e \
    --export-path ./uq_results_e2e \
    --polynomial-order 2 --regression lsq --n-bins 5
```

Results (strategy 1 — population-averaged Sobol S_Ti):

```
basal_elongation_rate        28.5%
fraction_active_rnap_bound   25.2%
cell_dry_mass_fraction       23.6%
fraction_active_rnap_free    13.6%
secretion_penalty_coeff      13.4%
kinetic_objective_weight     11.5%
```

Surrogate quality: training relative errors ~1e-15 (machine precision —
10 samples, 28 PCE terms, underdetermined but LSQ still fits exactly
for this smooth problem).

Strategy 4 shows `cell_dry_mass_fraction` dominates early in the cell
cycle (θ 0–20%: 28.8%) but declines toward division (θ 80–100%: 18.9%),
while `fraction_active_rnap_bound` increases (23.4% → 28.8%).
