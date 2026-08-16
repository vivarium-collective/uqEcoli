## Feedback Session 1
                                                                                                                                                                                   
*Plan: Item 16 — Baseline Variance Accounting (Q1) + Output-side PCA (high-dim)*

The stakeholder feedback has two distinct parts. Q1 is flagged as the primary MS 08.4.2 deliverable.                                                                                  

### Part A: Q1 — Baseline Variance Accounting (--no-perturbation): COMPLETE

Concept: Run vEcoli at fixed baseline params (no LHS perturbation), decompose observable variance into σ²_gen, σ²_seed, σ²_θ, σ²_residual. No PCE, no Sobol — pure variance-components  
decomposition.

#### A1. uq sample --no-perturbation flag (uq/cli.py)                                                                                                                                        

- Skip PCRV sampling entirely (no X generation, no LHS)                                                                                                                                 
- Run vEcoli with 0 variants (baseline only), using the specified --n-init-sims seeds × --generations gens                                                                            
- Save cache with metadata.mode = "baseline", a dummy single-row X (baseline param values from simData), and the full per-seed/per-generation timeseries in timeseries/sample_0000.npy +
sample_0000_meta.npz                                                                                                                                                                   
- The timeseries already stores per-timestep generation and lineage_seed arrays in the meta — this is all Q1 needs                                                                      
                                                                                                                                                                                        
#### A2. compute_variance_budget() — new function in uq/workflow.py                                                                                                                        

- Input: timeseries array (n_timesteps, n_obs) + generation/seed/growth-fraction metadata per timestep                                                                                  
- For each observable column, computes a 4-way additive decomposition:
- σ²_total = Var(all observations)                                                                                                                                                    
- σ²_gen = Var(per-generation means) — generation drift                                                                                                                             
- σ²_seed = Var(per-seed means) — stochastic seed variance                                                                                                                            
- σ²_θ = Var(per-growth-stage means) — cell cycle phase variance                                                                                                                    
- σ²_residual = σ²_total − σ²_gen − σ²_seed − σ²_θ — unexplained                                                                                                                      
- Returns a VarianceBudget dataclass with per-observable arrays + fraction vectors                                                                                                      
- Builds on existing _aggregate_by_group() and _bin_by_growth_stage() helpers                                                                                                           
                                                                                                                                                                                        
#### A3. uq quantify --no-perturbation flag (uq/cli.py + uq/workflow.py)                                                                                                                     

- Auto-detected if cache metadata has mode: "baseline", or forced via flag                                                                                                              
- Skips PCE fitting + Sobol computation entirely                                                                                                                                      
- Loads timeseries from cache → calls compute_variance_budget()                                                                                                                         
- Exports variance_budget.json to the results directory
- Generates Q1-specific HTML report                                                                                                                                                     
                                                                                                                                                                                      
#### A4. Report: "Variance Budget" section (uq/report.py)                                                                                                                                    

- Stacked bar SVG: one bar per observable, 4 color-coded segments (gen / seed / θ / residual)                                                                                           
- Summary table: absolute variance + percentage per component per observable
- When --no-perturbation, the report leads with this section (no Sobol/PCE sections)                                                                                                    
- When running normal forward-UQ (Q2), this section still appears as contextual denominator                                                                                             
                                                                                                                                                                                        
#### A5. Tests                                                                                                                                                                               

- Unit test for compute_variance_budget() with synthetic timeseries (known variance partition)                                                                                          
- Integration test: sample --no-perturbation → quantify --no-perturbation → verify variance_budget.json + report HTML exists with stacked bars
                                                                                                                                                                                        
### Part B: Output-side PCA for High-Dimensional Observables: COMPLETE                                                                                                                             

#### B1. uq quantify --output-pca K flag (uq/cli.py + uq/workflow.py)                                                                                                                      

- Before PCE fitting, if --output-pca K is set:
- Center Y, compute SVD → top-K principal components
- Y_pca shape (n_samples, K) replaces Y for all 4 strategies' PCE fitting
- Store: loadings (K, n_obs), explained variance ratios, mean vector
- PCE fits K surrogates instead of n_obs surrogates                                                                                                                                     
- Sobol indices are per-PC → interpretable as "which parameter drives PC1"                                                                                                              
                                                                                                                                                                                        
#### B2. Back-projection + biological interpretation                                                                                                                                         

- For each PC, compute top-N observable loadings → "PC1 ≈ growth signature (cell_mass 0.8, dry_mass 0.7, ...)"                                                                          
- Store pca_loadings.npy, pca_explained_variance.npy, pca_top_loadings.json in export
                                                                                                                                                                                        
#### B3. Report: "Principal Components" section (uq/report.py)                                                                                                                               

- Scree plot SVG: bar chart of variance explained per PC                                                                                                                                
- Per-PC Sobol bar chart: which parameters drive each PC                                                                                                                              
- Top loadings table: top-10 observables per PC with loading magnitudes                                                                                                                 
- Narrative caption: "PC1 of the transcriptome (X% variance) is driven Y% by parameter Z"                                                                                               
                                                                                                                                                                                        
#### B4. Tests                                                                                                                                                                               

- Unit test: synthetic Y (n_samples=50, n_obs=100), verify PCA reduces correctly, Sobol on PCs is sensible                                                                              
- Integration test: quantify --output-pca 5 on a cache with high-dim Y
                                                                                                                                                                                        
Files to modify                                                                                                                                                                       

┌───────────────────────────────┬───────────────────────────────────────────────────────────────────────────────────────────────────┐                                                 
│             File              │                                              Changes                                              │
├───────────────────────────────┼───────────────────────────────────────────────────────────────────────────────────────────────────┤                                                   
│ uq/cli.py                     │ Add --no-perturbation to sample + quantify, add --output-pca to quantify                          │
├───────────────────────────────┼───────────────────────────────────────────────────────────────────────────────────────────────────┤                                                   
│ uq/workflow.py                │ Add compute_variance_budget(), VarianceBudget dataclass, PCA reduction in quantify()              │                                                 
├───────────────────────────────┼───────────────────────────────────────────────────────────────────────────────────────────────────┤                                                   
│ uq/report.py                  │ New variance budget section (stacked bars + table), PCA section (scree + per-PC Sobol + loadings) │
├───────────────────────────────┼───────────────────────────────────────────────────────────────────────────────────────────────────┤                                                   
│ tests/test_variance_budget.py │ Unit + integration tests for Q1                                                                   │                                                 
├───────────────────────────────┼───────────────────────────────────────────────────────────────────────────────────────────────────┤                                                   
│ tests/test_output_pca.py      │ Unit + integration tests for PCA mode                                                             │                                                 
└───────────────────────────────┴───────────────────────────────────────────────────────────────────────────────────────────────────┘                                                   

### Execution order                                                                                                                                                                         

1. Part A first (A1→A2→A3→A4→A5) — it's the primary deliverable                                                                                                                         
2. Part B second (B1→B2→B3→B4) — builds on the same infrastructure
                                                                                                                                                                                        
### What I will NOT do                                                                                                                                                                    

- Modify the SMS-API                                                                                                                                                                    
- Change the existing Q2 (forward UQ) pipeline behavior — --no-perturbation is additive
- Add functional grouping or generalized Sobol (stakeholder said PCA first)                                                                                                             
- Touch the DAW/dashboard (report-only for now)  

--- 

## Feedback Session 2