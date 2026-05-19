# Cross-Condition GSA

The cross-condition sensitivity analysis extends the UQ framework to compare
parameter importance across multiple environmental or genetic conditions.

## How It Works

```bash
# Sample with multiple conditions:
uv run uq sample /path/to/simData.cPickle \
    --conditions glucose_minus_aas \
    --conditions glucose_plus_aas \
    --n-samples 50 --live

# Quantify (auto-detects multi-condition cache):
uv run uq quantify experiment_id \
    --sim-base-path /path/to/sims \
    --precomputed-path ./uq_cache
```

## Output

A `MultiConditionResult` is produced with:

| Metric | Description |
|--------|-------------|
| **Rank stability** | How consistent parameter rankings are across conditions (●●●○○ indicators) |
| **Universal drivers** | Parameters that are top-ranked regardless of condition |
| **Condition-specific** | Parameters that matter only in certain environments |
| **Differential Sobol** | Change in S_Ti between conditions for each parameter |

## Visualization

- `uq compare` produces a side-by-side Sobol group bar chart
- The dashboard (tk or marimo) shows a condition selector combobox
- CLI output includes a Rich table with rank stability indicators

## Example Output

```
Parameter          glucose_minus_aas   glucose_plus_aas   Stability
──────────────────────────────────────────────────────────────
basal_elongation_rate   0.285 (#1)        0.271 (#1)        ●●●
rnap_bound              0.252 (#2)        0.248 (#2)        ●●●
dry_mass_frac           0.236 (#3)        0.185 (#5)        ●○○
kinetic_objective_weight 0.115 (#6)        0.213 (#3)        ○○○
```
