# Pipeline Annotations

Key numerical outputs at each step of the RFC006 UQPC workflow.

## Step 1: Define Parameter Space
- **N = 6** parameters (rnap_free, rnap_bound, basal_elongation_rate, kinetic_objective_weight, secretion_penalty_coeff, dry_mass_frac)
- **Bounds:** ±15–50% around calibrated baseline per parameter
- **Space:** 6-dimensional hypercube → mapped to [-1, 1]^6 germ space

## Step 2: Load Simulation Data
- **Source:** hive-partitioned Parquet from vEcoli single-cell listener outputs
- **Dimensions:** ~10 generations × 1000 seeds × timepoints

## Step 3: Aggregation (3 strategies)
| Strategy | Grouping | Output Shape |
|----------|----------|-------------|
| Uniform | All cells pooled | Y_agg ∈ R^{N_samp × N_obs} |
| By Generation | Per generation | Y_gen ∈ R^{N_samp × N_gen × N_obs} |
| By Lineage Seed | Per stochastic seed | Y_seed ∈ R^{N_samp × N_seed × N_obs} |

## Step 4: Variance Decomposition
- **Generation fraction:** σ²_gen / σ²_total = 0.0 (requires N_gen > 1)
- **Seed fraction:** σ²_seed / σ²_total = 0.0 (requires N_seed > 1)
- **Residual fraction:** 1.0 — feeds into Phase 2 observable selection

## Step 5: PCE Surrogate
- **Basis:** Legendre polynomials, order p = 2
- **Number of basis terms:** N_basis = (p + d)!/(p! d!) = 28 for d=6, p=2
- **Samples:** N = 10 training (UQPC default), N_test = 0 held-out
- **Regression:** least squares (lsq) — overdetermined for N_basis=28, N=10
- **Validation:** Held-out relative error (requires --n-test)

## Step 5b: GSA-Informed Selection
- **High-residual observables selected for Phase 2**

## Step 6: Sobol Indices (Phase 1)
| Parameter | S_Ti | S_i | Interaction |
|-----------|------|-----|-------------|
| basal_elongation_rate | 0.2850 | 0.2515 | 0.0335 |
| rnap_bound | 0.2516 | 0.1754 | 0.0763 |
| dry_mass_frac | 0.2363 | 0.2249 | 0.0115 |
| rnap_free | 0.1362 | 0.0755 | 0.0607 |
| secretion_penalty_coeff | 0.1343 | 0.1024 | 0.0319 |
| kinetic_objective_weight | 0.1153 | 0.0113 | 0.1040 |

## Step 6b-6d: Growth-Stratified Phase 2
- **θ = normalized log mass:** 10 bins from birth to division
- **Key insight:** dry_mass_frac S_Ti decreases from 0.294 (early) → 0.185 (late); rnap_bound increases from 0.231 → 0.292

## Pipeline Outputs
- 1× PCESurrogate (bulk) + 10× per-stage surrogates
- 1× SobolIndices (population) + 10× per-stage SobolIndices
- Self-contained HTML report with interactive JS PCE explorer
