# SAMPLING — how `uq sample` uses the vEcoli variants API

`uq sample` is a thin, vEcoli-specific implementation of steps 1–3 of the
**PyTUQ UQPC forward-UQ workflow**
(<https://sandialabs.github.io/pytuq/apps/uqpc.html>). It delegates every
heavy step to either PyTUQ (input PC, germ sampling) or vEcoli (parameter
mutation, simulation, Parquet emission). There is no hand-rolled parameter
evaluator, scheduler, or variant expander anywhere in this repository.

This document walks through that delegation end-to-end and cites the exact
functions and line numbers in both repos.

---

## 1. Setup inputs (UQPC step 1)

An input PC representation is built from per-parameter bounds. PyTUQ owns
the polynomial basis and the affine domain transform.

- `uq/workflow.py::_setup_input_pc` (lines 110–145) constructs a
  `pytuq.rv.pcrv.PCRV` with `pctype="LU"` (Legendre, uniform priors, order 1)
  whose coefficients encode the affine map `xi ∈ [-1, 1] → [lb, ub]`.
- The parameter list comes from `libuq/pipeline/param_loader.py` —
  `ParameterDataset.to_parameter_space(...)` projects scalar `SimulationDataEcoli`
  attributes into generic `SimDataParameter` specs. The defaults are the six
  parameters in `DEFAULT_SIM_DATA_PARAMETERS`.

**No hand-rolled samplers, CDFs, or bound checks** — PyTUQ's `PCRV` is the
single source of truth for the parameter space.

---

## 2. Generate samples (UQPC step 2)

`uq/cli.py::sample` draws samples from the germ measure using PyTUQ
directly, then lifts them back into physical space with `evalPC`.

- `uq/cli.py` lines ~128–141:
  ```python
  input_pc, _, _ = _setup_input_pc(bounds)
  np.random.seed(seed)
  germ_train = input_pc.sampleGerm(n_samples)
  X_train   = input_pc.evalPC(germ_train)
  if n_test > 0:
      np.random.seed(seed + 1)
      germ_test = input_pc.sampleGerm(n_test)
      X_test    = input_pc.evalPC(germ_test)
  ```

This is the UQPC `--sampl rand` path, with `--ntst = n_test` for held-out
validation (PyTUQ's surrogate-quality cross-check).

`X_test` is concatenated onto `X_train` before evaluation, so vEcoli runs
**one** workflow that covers both train and validation — the split is
preserved in the cache and consumed by `uq quantify` for test relative
errors.

---

## 3. Evaluate the model (UQPC step 3) — vEcoli variants API

Every sample becomes one vEcoli variant through the **`sim_data_setattr`**
variant function that lives in vEcoli itself (not in this repo), and the
whole batch is executed by vEcoli's Nextflow workflow. There is no shortcut
around Parca, no in-process `EcoliSim`, no hand-rolled mutation code.

### 3a. Variant encoding — `sim_data_setattr`

For each germ sample, we emit one dictionary of attribute-path mutations.
The variant list (one entry per sample) is handed to vEcoli's variant system
via the `sim_data_setattr` variant function:

- `uq/vecoli_config.py::_build_variants_from_samples` (lines 127–142):
  ```python
  def _build_variants_from_samples(X, param_specs):
      mutations_list = []
      for i in range(X.shape[0]):
          mutations = {}
          for j, spec in enumerate(param_specs):
              val = float(X[i, j])
              if spec.index is not None:
                  mutations[spec.attr_path] = {"__index__": spec.index, "__value__": val}
              else:
                  mutations[spec.attr_path] = val
          mutations_list.append(mutations)
      return {"sim_data_setattr": {"mutations": {"value": mutations_list}}}
  ```

The `"value": [...]` form is what vEcoli's variant parser recognises as a
"make one variant per list element" expansion — this is the canonical API
documented at <https://covertlab.github.io/vEcoli/workflows.html#variants>
and implemented in `../vEcoli/runscripts/create_variants.py::parse_variants`
(the `if param_type == "value"` branch).

The actual mutation logic lives in vEcoli:

- `../vEcoli/ecoli/variants/sim_data_setattr.py::apply_variant` (full file,
  58 lines) walks each dot-path with `getattr` and writes the new value
  (or indexed slot) directly onto the `SimulationDataEcoli` object before
  the sim runs. **This lives in vEcoli, not here**, so every uqEcoli run is
  driven by the official variant pipeline.

We contribute **zero** custom simulation-data editing code — the only thing
this repo does is translate a PyTUQ germ sample into the dict format
`apply_variant` expects.

### 3b. Config assembly — the vEcoli workflow config

- `uq/vecoli_config.py::_build_config` (lines 100–124) builds a minimal, standard
  vEcoli workflow config:
  ```python
  {
      "sim_data_path": sim_data_path,   # pre-computed simData.cPickle
      "experiment_id": "uqpc_batch",
      "emitter": "parquet",             # official vEcoli Parquet emitter
      "emitter_arg": {"out_dir": output_dir},
      "max_duration": max_duration,
      "n_init_sims": n_init_sims,       # lineage seeds — strategy 3
      "generations": generations,       # generations — strategy 2
      "single_daughters": True,
      "suffix_time": False,
      "variants": variants_section,     # {"sim_data_setattr": ...}
  }
  ```

  The config is written to
  `<cache_dir>/_batch/workflow_config.json` and passed verbatim to
  vEcoli. Notice that we pass `sim_data_path` so vEcoli **skips Parca** —
  it reuses the pre-computed pickle we already loaded in step 1.

### 3c. Execution — `runscripts/workflow.py --config`

`uq/cli.py` lines 175–182 spawn vEcoli's own Nextflow-backed workflow
entry point as a subprocess:

```python
workflow_script = os.path.join(vecoli_root, "runscripts", "workflow.py")
cmd = [sys.executable, workflow_script, "--config", str(config_path)]
proc = subprocess.Popen(cmd, ..., cwd=vecoli_root, env=env)
```

This is the exact command documented in vEcoli's workflow manual. vEcoli
then:

1. Runs `runscripts/create_variants.py` to materialize
   `variant_sim_data_0000.cPickle`, `..._0001.cPickle`, … from our mutation
   list (one baseline + one per sample).
2. Dispatches Nextflow processes for each `(variant, lineage_seed, generation)`
   triple.
3. Emits Parquet files in the official hive-partitioned layout:
   `history/experiment_id=.../variant=.../lineage_seed=.../generation=.../agent_id=.../NNN.pq`

While this runs, `uq/cli.py::sample` drives a rich-progress bar by scraping
stdout and polling `_count_completed_variants` (`uq/vecoli_config.py`), which in
turn walks the hive path looking for completed `variant=N` partitions.

### 3d. Result collection — hive-partitioned Parquet

- `uq/vecoli_config.py::_collect_variant_timeseries` (lines 164–214) reads the
  Parquet tree with `polars.read_parquet(..., hive_partitioning=True)` —
  the `variant`, `lineage_seed`, `generation`, and `time` columns come
  directly from vEcoli's emitter.
- For each variant index it extracts the observable columns, computes the
  per-variant timeseries mean (→ aggregated `Y`), and preserves the
  per-row `generation` / `lineage_seed` arrays. That per-row metadata
  is what enables RFC006 strategies 2 (by generation) and 3 (by lineage
  seed) in `uq quantify`.

The train/test split is restored by slicing the collected arrays at index
`n_samples` — the first `n_samples` variants are training, any remaining
variants are the UQPC `--ntst` validation set. Everything is stored in
`libuq.sampling.PrecomputedCache` (including new `X_test.npy`, `Y_test.npy`,
and `timeseries_test/`).

---

## 3e. Alternative: SMS-API remote execution

When `--api-url` is provided, step 3 routes through the **SMS-API** instead
of a local subprocess. The SMS-API (documented at
<https://sms-api.readthedocs.io>) runs vEcoli on AWS Batch and produces
cd1 analysis outputs as pre-aggregated TSV files.

The remote flow:

1. Steps 1-2 (parameter space + PCRV sampling) remain **local**.
2. Step 3a (variant encoding): same `_build_variants_from_samples` — but
   currently submitted as individual simulations to the API rather than as
   a single multi-variant workflow config.
3. Step 3c (execution): `SmsApiClient.submit_simulation()` → `POST
   /api/v1/simulations` with simulator_id, generations, seeds, observables.
4. Step 3d (collection): `POST /api/v1/simulations/{id}/data` returns a
   tar.gz archive containing cd1 analysis TSVs (not raw Parquet):

```
{experiment_id}/analyses/variant=0/plots/analysis=cd1_transcriptomics/transcriptomics.tsv
{experiment_id}/analyses/variant=0/plots/analysis=cd1_proteomics/proteomics.tsv
{experiment_id}/analyses/variant=0/plots/analysis=cd1_fluxomics/cd1_fluxomics_detailed.tsv
{experiment_id}/analyses/variant=0/plots/analysis=cd1_metabolomics/metabolomics.tsv
{experiment_id}/analyses/variant=0/plots/analysis=cd1_higher_order_properties/higher_order_properties.tsv
```

Each TSV has 3 tab-separated columns: `identifier`, `mean`, `std`. The
`mean` column becomes the Y vector for UQ. The mapping between cd1
modules and UQ observable presets is defined in `uq/remote.py::CD1_MODULE_MAP`:

| UQ preset | cd1 module | Observables |
| --- | --- | --- |
| `higher_order` | `cd1_higher_order_properties` | ~5 (mass, volume, DNA/RNA fractions) |
| `transcriptome` | `cd1_transcriptomics` | ~4,345 EcoCyc Gene IDs |
| `proteome` | `cd1_proteomics` | ~4,309 EcoCyc Monomer IDs |
| `fluxome` | `cd1_fluxomics` | ~2,820 EcoCyc Reaction IDs |
| `exchange_fluxes` | `cd1_metabolomics` | ~165 EcoCyc Compound IDs |

The `mass` preset is local-only (requires raw Parquet timeseries).

The `SmsApiClient` in `uq/remote.py` handles ALB 502/504 retries
automatically (a common transient failure on the stanford-test deployment).

---

## 4. What's in this repo vs. what's in vEcoli / PyTUQ / SMS-API

| Concern | Owner | Where |
| --- | --- | --- |
| Polynomial basis, germ sampling, PC evaluation | **PyTUQ** | `pytuq.rv.pcrv.PCRV` |
| Input PC construction (Legendre affine map) | PyTUQ (invoked) | `uq/workflow.py::_setup_input_pc` |
| Variant function (attribute mutation) | **vEcoli** | `ecoli/variants/sim_data_setattr.py` |
| Variant expansion (`"value": [...]`) | **vEcoli** | `runscripts/create_variants.py::parse_variants` |
| Parca / `simData` build | **vEcoli** | handled by `runscripts/workflow.py` |
| Simulation dispatch + Nextflow orchestration | **vEcoli** | `runscripts/workflow.py` |
| Parquet emission + hive partitioning | **vEcoli** | vEcoli parquet emitter |
| cd1 analysis modules (TSV output) | **vEcoli** | cd1_transcriptomics, etc. |
| Remote simulation execution | **SMS-API** | `POST /api/v1/simulations` |
| Remote data download | **SMS-API** | `POST /api/v1/simulations/{id}/data` |
| Mutation dict building | uqEcoli | `uq/vecoli_config.py::_build_variants_from_samples` |
| Config assembly | uqEcoli | `uq/vecoli_config.py::_build_config` |
| Subprocess driver + progress UI | uqEcoli | `uq/cli.py::_sample_local` |
| SMS-API client + cd1 TSV parsing | uqEcoli | `uq/remote.py`, `uq/cli.py::_sample_remote` |
| Hive Parquet → `(X, Y, meta)` reshape | uqEcoli | `uq/vecoli_config.py::_collect_variant_timeseries` |
| Cache format (`PrecomputedCache`) | uqEcoli | `libuq/sampling.py` |

The uqEcoli rows are intentionally thin. Every numerically-meaningful step
is implemented in vEcoli or PyTUQ; we only adapt their I/O formats.

---

## 5. Verification

You can verify each link in this chain yourself:

```bash
# The variant function we use is shipped by vEcoli, not this repo
ls ../vEcoli/ecoli/variants/sim_data_setattr.py

# Every parameter mutation goes through vEcoli's apply_variant
grep -n apply_variant ../vEcoli/ecoli/variants/sim_data_setattr.py

# The CLI actually spawns vEcoli's workflow.py
grep -n "runscripts/workflow.py\|runscripts.workflow" uq/cli.py uq/vecoli_config.py

# Germ sampling uses PyTUQ's PCRV
grep -n "sampleGerm\|evalPC" uq/workflow.py uq/cli.py

# Variant expansion is vEcoli's — we only emit the list form it parses
grep -n '"value"' uq/vecoli_config.py ../vEcoli/runscripts/create_variants.py
```

---

## 6. The novelty

This repo's value-add is not in re-implementing UQ primitives; it is in the
**end-user entrypoints** that glue PyTUQ and vEcoli together:

- `uq sample` / `uq quantify` — the two-stage CLI (Typer + rich).
- `uq tui` — a Textual terminal UI for the same workflow.
- `uq gui` / `uq dashboard` — marimo and tkinter GUIs over the cached
  results (`app/gui.py`, `app/dashboard_simple.py`, `app/uq_daw_simple.py`).
- `QuantifyResult.export(...)` — writes a dashboard-ready artifact
  directory (`population_surrogate/`, `growth_stratified_surrogate/`,
  per-strategy Sobol `.npy` files, `uq_results.json`).
- RFC006's four aggregation strategies (uniform, by generation, by lineage
  seed, growth-stratified) as a first-class feature of `quantify` — each
  still runs through PyTUQ's `PCRV`/`lreg` stack and the shared
  `run_uqpc` function in `uq/workflow.py`.

Everything else — the polynomial algebra, the sampling, the simulation —
belongs to the upstream projects.

---

## 7. v2ecoli backend (`--backend v2ecoli`)

An alternative in-process execution backend replaces the subprocess + Nextflow
orchestration with a **process-bigraph composite** (`v2ecoli`).  This is the
preferred backend for new work.

### What changes

| Concern | vEcoli (legacy) | v2ecoli (preferred) |
| --- | --- | --- |
| Execution | `subprocess.run(workflow.py --config ...)` | In-process `Composite.run(interval)` |
| Mutation | `sim_data_setattr` variant function on sim_data | In-memory `apply_mutations_to_configs()` on cache bundle |
| Observable extraction | `polars.read_parquet()` on hive-partitioned Parquet | Direct state traversal of `composite.state['agents']` |
| Parallelism | Nextflow-managed | `ProcessPoolExecutor` with `--max-workers N` |
| Dependency | vEcoli repo, Nextflow, Parca | v2ecoli package (`uv add ../v2ecoli --editable`) |

### Architecture

```
uq sample --backend v2ecoli
  ↓
LHS / PCRV sampling  ──>  X (n_samples × n_params)
  ↓
_sample_v2ecoli()
  └─ generate_cache_bundle(sim_data_path, cache_dir)
  │   → cache_dir/{initial_state.json, sim_data_cache.dill}
  └─ V2ecoliGenerator(cache_dir)._run_batch(X)
      └─ For each sample:
          1. Load baseline cache bundle (memoized)
          2. Deep-copy configs dict, apply dot-path mutations
          3. Build v2ecoli baseline composite
          4. Run for N generations (tick loop)
          5. Extract observables from composite.state
      └─ Collect Y_aggregated, Y_timeseries, Y_meta
  ↓
PrecomputedCache (X, Y, timeseries, metadata)
```

### Key source files

- `uq/generators/v2ecoli.py` — `V2ecoliGenerator`, cache bundle generation,
  mutation logic, composite builder, tick-loop runner
- `uq/v2ecoli_bridge.py` — Observable path map (`OBSERVABLE_PATHS`) connecting
  UQ observable names to v2ecoli state traversal paths

### Usage

```bash
uv run uq sample api_simulation_default \
    --sim-base-path /path/to/sims \
    --cache-dir ./uq_cache \
    --n-samples 20 \
    --backend v2ecoli \
    --max-workers 4

uv run uq quantify api_simulation_default \
    --sim-base-path /path/to/sims \
    --precomputed-path ./uq_cache
```

The `quantify` command is backend-agnostic — it reads the same
`PrecomputedCache` format regardless of how the samples were generated.

---

## How vEcoli is used in the sampling mechanism

### Using the `vEcoli/ecoli/variants/sim_data_setattr.py` Variant module

One way to explicitly specify simdata(model parameter) mutations to be applied during the sampling process is to 
use the generalized, universal `sim_data_setattr` variant module in vEcoli.

The `sim_data_setattr` variant factory mechanism is parameterized by the following object:

```
{
  "mutations": {
      "value": [
          <MUTATION_CONFIG>, 
      ]
  }
  
}
```
...where `<MUTATION_CONFIG>` is: 

```
{
  "<DOT_SEPARATED_HEIRARCHICAL_SIMDATA_PATH>": <VALUE>
}
```


### Example

The following example uses the `sim_data_setattr` variant module/mechanism to create 10 unique variants from the baseline.

**NOTE**: Users of this mutation method should be careful to include the _same_ keys in each object within 
the `"value"` field of the `mutations` parameter of `sim_data_setattr`, as is shown below.

```json5
{
  // rest of vEcoli config JSON here
  "variants": {
    "sim_data_setattr": {
      "mutations": {
        "value": [
          {
            "process.transcription.fraction_active_rnap_free": 0.33239882614641975,
            "process.transcription.fraction_active_rnap_bound": 0.12205844942958022,
            "process.translation.basal_elongation_rate": 22.954087631390934,
            "process.metabolism.kinetic_objective_weight": 3.2339518335564726e-07,
            "process.metabolism.secretion_penalty_coeff": 0.0010491720568015044,
            "mass.cell_dry_mass_fraction": 0.3469584627764558
          },
          {
            "process.transcription.fraction_active_rnap_free": 0.45915714741018154,
            "process.transcription.fraction_active_rnap_bound": 0.21699098521619942,
            "process.translation.basal_elongation_rate": 16.81342018847654,
            "process.metabolism.kinetic_objective_weight": 1.2673585565928122e-07,
            "process.metabolism.secretion_penalty_coeff": 0.0027282960955007157,
            "mass.cell_dry_mass_fraction": 0.32751328233611143
          },
          {
            "process.transcription.fraction_active_rnap_free": 0.4110386671985091,
            "process.transcription.fraction_active_rnap_bound": 0.20324426408004215,
            "process.translation.basal_elongation_rate": 18.797880430957836,
            "process.metabolism.kinetic_objective_weight": 7.927321684337582e-08,
            "process.metabolism.secretion_penalty_coeff": 0.0006547483450184822,
            "mass.cell_dry_mass_fraction": 0.3439498941564189
          },
          {
            "process.transcription.fraction_active_rnap_free": 0.381704866523348,
            "process.transcription.fraction_active_rnap_bound": 0.1412339110678276,
            "process.translation.basal_elongation_rate": 19.76270396281799,
            "process.metabolism.kinetic_objective_weight": 4.76998491764e-07,
            "process.metabolism.secretion_penalty_coeff": 0.00459194180935452,
            "mass.cell_dry_mass_fraction": 0.3394827350427649
          },
          {
            "process.transcription.fraction_active_rnap_free": 0.28432410089733606,
            "process.transcription.fraction_active_rnap_bound": 0.13818249672071004,
            "process.translation.basal_elongation_rate": 20.928909794821468,
            "process.metabolism.kinetic_objective_weight": 4.845344148835517e-07,
            "process.metabolism.secretion_penalty_coeff": 0.0016645099172000759,
            "mass.cell_dry_mass_fraction": 0.3097899978811085
          },
          {
            "process.transcription.fraction_active_rnap_free": 0.2843187944739646,
            "process.transcription.fraction_active_rnap_bound": 0.13834045098534337,
            "process.translation.basal_elongation_rate": 25.207287498109178,
            "process.metabolism.kinetic_objective_weight": 4.137788066524075e-07,
            "process.metabolism.secretion_penalty_coeff": 0.003481350279592919,
            "mass.cell_dry_mass_fraction": 0.3421874235023117
          },
          {
            "process.transcription.fraction_active_rnap_free": 0.26277839467700387,
            "process.transcription.fraction_active_rnap_bound": 0.15042422429595376,
            "process.translation.basal_elongation_rate": 17.595759168058677,
            "process.metabolism.kinetic_objective_weight": 1.8707619612801682e-07,
            "process.metabolism.secretion_penalty_coeff": 0.0019026998424023491,
            "mass.cell_dry_mass_fraction": 0.25884925020519195
          },
          {
            "process.transcription.fraction_active_rnap_free": 0.4405587520704857,
            "process.transcription.fraction_active_rnap_bound": 0.17247564316322378,
            "process.translation.basal_elongation_rate": 21.685047699376952,
            "process.metabolism.kinetic_objective_weight": 9.395245130287276e-08,
            "process.metabolism.secretion_penalty_coeff": 0.0028403060953001483,
            "mass.cell_dry_mass_fraction": 0.2695982862419145
          },
          {
            "process.transcription.fraction_active_rnap_free": 0.3822453025835059,
            "process.transcription.fraction_active_rnap_bound": 0.16319450186421156,
            "process.translation.basal_elongation_rate": 22.70138939520655,
            "process.metabolism.kinetic_objective_weight": 3.579048619304706e-07,
            "process.metabolism.secretion_penalty_coeff": 0.0029601962570447583,
            "mass.cell_dry_mass_fraction": 0.2545227288910538
          },
          {
            "process.transcription.fraction_active_rnap_free": 0.40577596711513,
            "process.transcription.fraction_active_rnap_bound": 0.14912291401980418,
            "process.translation.basal_elongation_rate": 15.60385536535997,
            "process.metabolism.kinetic_objective_weight": 2.4806862218282063e-07,
            "process.metabolism.secretion_penalty_coeff": 0.0013318450498648713,
            "mass.cell_dry_mass_fraction": 0.28253303307632643
          }
        ]
      }
    }
  }
}
```

### test

```bash
uv run uq sample /path/to/simData.cPickle \
      --variants-source base-config \
      --base-config examples/vecoli_configs/mec.json \
      --params-file examples/uq_artifacts/params/params_demo.json \
      --cache-dir ./uq_cache_mec \
      --generations 4 --n-init-sims 2
```

## REMAINING GAPS

Honest answer: the bookkeeping is now sound, the theory is not yet.
  
Before this work, the cache could ship X rows that had no relationship to what vEcoli actually
applied (silently bogus Sobol). That's fixed. But "every X row matches a variant" is a
necessary, not sufficient, condition for the PCE+Sobol decomposition to mean what we claim it
means. Here's the honest accounting of what's still broken.

### What's now correct

1. 1:1 correspondence: every row of X in the cache is a parameter point vEcoli physically ran.
True in both params-file and base-config modes.
2. Forward-map invertibility: the affine germ↔physical map is exact and _germ_from_physical is
its true inverse.
3. Sample-count adequacy: the basis-size diagnostic catches N < B before quantify silently fits an underdetermined system.

That's it. Three properties.

### Theoretical gaps that remain

#### Gap 1 — the input measure assumption is unenforced under BYO  ⚠️ biggest

The PCE math is built on a single assumption: the input variables are uniformly distributed on
the bound box. Legendre polynomials are orthogonal under that measure, and the Sudret
variance decomposition Var[Ŷ] = Σ c_α² ‖Φ_α‖² is exact only because of that orthogonality. The
Sobol indices S_i, S_Ti then have their textbook variance-decomposition interpretation.

Under default params-file mode this holds by construction — PCRV.sampleGerm draws uniform
samples.  

Under base-config mode, we have no idea what distribution the user's variants came from. Could
be LHS, could be a hand-picked grid, could be the cartesian product of two conditions with
all the mass in two corners of the box. We fit a Legendre PCE to those points and read out
"Sobol indices" — but if the empirical X distribution isn't approximately uniform, the basis
is not orthogonal under that measure, and the coefficient-square decomposition doesn't
decompose anything meaningful. The numbers we print are well-defined polynomial fit
statistics; they are not Sobol sensitivity indices in the Sobol sense.

What's needed: a uniformity test on the empirical X distribution (KS test per dimension, or a
discrepancy measure like the L2-star discrepancy used in QMC literature). Warn if the
distribution is far from uniform; refuse if it's pathological.

#### Gap 2 — the forward model isn't deterministic

PCE assumes f: X → Y is deterministic. Our f(x) = time-mean of one stochastic vEcoli run is a
single sample of a stochastic process, not an estimate of E[vEcoli(x)]. With --n-init-sims 1
(the default), we have no replicate structure to average over.

Consequences:
- The variance the PCE attributes to "parameters" is contaminated with aleatoric (stochastic)noise.
- Residual variance in the fit could be model misspecification OR irreducible noise — we cannot separate them.
- Sobol indices conflate "parameter explains Y" with "lineage_seed explains Y".

This is a whole-framework gap, not BYO-specific, but BYO makes it worse because users often
inherit configs with --n-init-sims 1 and never reconsider.

What's needed: explicit replicate handling. Either (a) require --n-init-sims ≥ K and average
internally as an E[f(x)] estimator, or (b) carry replicates as separate rows and fit a
noise-aware PCE (e.g. PyTUQ's BCS with explicit noise term). At minimum, a noise-floor
diagnostic: fit a constant-only PCE, compare residual to total variance.

#### Gap 3 — surrogate validation is lost in BYO

In default mode, --n-test N is the only honest fit-quality check: independent points from the
same prior, evaluated separately, test relative error reported. The training error is
uninformative (PCE will fit training points exactly when N ≈ B).

Under BYO, we silently drop --n-test. The user only sees training error, which is essentially
the fit residual and can be arbitrarily small while the model generalizes terribly.

What's needed: k-fold cross-validation for BYO mode. Hold out 10–20% of the variants, fit on
the rest, report CV error. This is a moderate change but it's the only honest way to get a
generalization estimate when you can't draw new samples.

#### Gap 4 — the adequacy diagnostic is one-dimensional

The check n_samples ≥ 2 × basis_size is a count rule. Real conditioning depends on the
geometry of X, not its size. Two failure modes the count check misses:

- Design matrix conditioning: even with N >> B, if the variants clump in one corner of the box, the design matrix A = pcrv.evalBases(Ξ, 0) can have condition number κ(A) >> 1. LSQ then amplifies any noise by κ. We don't compute or surface κ(A).
- Sobol index uncertainty: PCE coefficients have estimation variance; Sobol indices are nonlinear functions of those coefficients and inherit their own uncertainty. We report S_Ti = 0.42 with no confidence interval. The user has no way to tell whether S_Ti = 0.42 and S_Ti = 0.38 for two different parameters are meaningfully different.

What's needed:
- Add κ(A) to the diagnostic. Warn at κ > 1e6; refuse at κ > 1e10.
- Bootstrap Sobol indices: resample (X, Y) rows with replacement, refit PCE, compute Sobol, report empirical CIs. Cheap because quantify is fast.

#### Gap 5 — out-of-bounds variants are warned, not rejected

Under BYO, if a user's mutation value falls outside the bounds in --params-file, the germ
value lands outside [-1, 1], the Legendre polynomials are evaluated there anyway, and we
proceed with a yellow warning. The math at that point: orthogonality is gone, the Sobol
decomposition is invalid, but we print numbers as if they were.

What's needed: one of three honest choices. Either (a) refuse the cache when more than a small
fraction is out-of-bounds, (b) auto-derive bounds from the variants themselves (X.min/max ± 
ε) and require the user to opt in via --auto-bounds, or (c) trim out-of-bounds rows with a
clear count of what was dropped. Current behavior is worst-case: bad numbers, weak warning.

#### Gap 6 — n_samples and compute budget are decoupled

A user with n_samples=20, n_init_sims=4, generations=8 spends 640 vEcoli-runs; another with
n_samples=80, n_init_sims=1, generations=8 spends the same 640 — but the second has 4× the
PCE-relevant data and (in expectation) tighter Sobol estimates. The diagnostic doesn't surface
this trade-off, and the docs don't either.

What's needed: a compute-vs-fit-quality calculator. Given target Sobol-index CI width,
recommend (n_samples, n_init_sims, polynomial_order) triples that bracket it.

#### Gap 7 — per-strategy effective sample size varies

Strategies 2 and 3 fit a separate PCE per group (one per generation, one per seed). Each
per-group fit uses the same basis size B but only N_per_group ≤ N samples — so the adequacy
check should fire per-strategy with the actual per-group N, not the global N. Strategy 4
stacks across stages so its effective N is N × n_bins. The current diagnostic reports one
number against the global N.

What's needed: per-strategy adequacy printout in the quantify report — Strategy 2 (gen=3): 
N=20 vs B=28 → underdetermined.

### Triage

If you want to close the most-impactful gaps next, in order of impact-per-effort:

1. Gap 1 + Gap 5: uniformity check + out-of-bounds rejection (one CLI session of work, surfaces the most catastrophic failure mode of BYO).
2. Gap 4a: condition-number addition to the adequacy diagnostic (~30 min).
3. Gap 3: k-fold CV for BYO (replaces the lost --n-test).
4. Gap 4b: bootstrap Sobol CIs (the most user-visible upgrade — finally lets people compare parameters honestly).
5. Gap 2: noise-floor / replicate handling (deepest, biggest refactor).
6. Gaps 6, 7: docs/diagnostic polish.

So: bookkeeping is sound. Statistics are not — yet. The current code is honest about what it 
ran but not yet honest about what its numbers mean. Want me to add §25-D-G items to todo.md
and stage out the first batch (uniformity check + OOB rejection + condition number)?