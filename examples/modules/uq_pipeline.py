#!/usr/bin/env python
"""
Full End-to-End UQ Pipeline (RFC006)

This script demonstrates the complete UQ framework workflow as specified by RFC006
for Milestone 08.4.2: "Implement uncertainty quantification framework to track
prediction confidence."

The pipeline has two parallel phases after variance decomposition:

  Phase 1 (Population/Bulk):
    Morris prescreening → PCE surrogate → Sobol indices
    "Which parameters drive bulk output variance?"

  Phase 2 (Cell Cycle/Phenotypic):
    GSA-informed observable selection → Koopman θ → Strategy 4 wrapper
    → per-stage PCE → per-stage Sobol indices
    "Which parameters drive variance WITHIN each cell cycle stage?"

Both phases produce a UqProfile, assembled into a PipelineResult.

This example uses synthetic data (generate_synthetic_simulation_data) and
synthetic wrappers (SyntheticBulkWrapper, SyntheticStrategy4Wrapper) as
stand-ins for real vEcoli simulations. In production, these would be replaced
by SimulationWrapper and the real Strategy4Wrapper from uq.pipeline.workflow.

Usage:
    uv run python examples/uq_pipeline.py
    uv run python examples/uq_pipeline.py --output-dir ./my_results

Author: Alex Patrie
"""

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import polars as pl

# =============================================================================
# UQ Package Imports — using actual pipeline module
# =============================================================================
from libuq import (
    DynamicModeDecomposition,
    MecillinamParams,
    VioPathwayParams,
    XSpaceVecoli,
    identify_cell_cycle_relevant_observables,
)
from libuq.common.models import BaseClass
from libuq.pipeline.models import (
    GeneKnockoutParams,
    PipelineResult,
    StratificationLens,
    UqProfile,
)
from libuq.pipeline.workflow import (
    aggregate_timeseries,
    compute_strategy4_sobol,
    get_variance_decomposition,
    run_phase1,
)
from libuq.synthetic import generate_signal, generate_synthetic_simulation_data


# =============================================================================
# Synthetic Wrappers (stand-ins for real simulation wrappers)
#
# In production, Phase 1 uses SimulationWrapper and Phase 2 uses
# uq.pipeline.workflow.Strategy4Wrapper (which wraps a base simulation
# function with Koopman θ-binning). Here we use synthetic wrappers
# that exercise the same pipeline machinery with generate_signal().
# =============================================================================


class SyntheticBulkWrapper:
    """
    Synthetic wrapper for Phase 1: params → scalar bulk output.

    In production, this would be SimulationWrapper which runs vEcoli and
    aggregates outputs. Here we use generate_signal() to produce a
    parameter-dependent timeseries and return its mean as the bulk output.
    """

    def __init__(self, param_names: list[str]):
        self.param_names = param_names

    def __call__(self, x: np.ndarray) -> np.ndarray:
        signal = generate_signal(x, self.param_names, baseline_value=1.5, n_timesteps=400)
        return np.array([signal.mean()])

    def evaluate_batch(self, X: np.ndarray) -> np.ndarray:
        return np.vstack([self(x) for x in X])


class SyntheticStrategy4Wrapper:
    """
    Synthetic wrapper for Phase 2 (Step 6d): params → per-stage output.

    In production, this would be uq.pipeline.workflow.Strategy4Wrapper which:
      1. Runs the simulation via base_wrapper(params)
      2. Computes θ via KoopmanCellCycleVariable
      3. Bins by θ into n_bins stages
      4. Returns per-stage means

    Here we simulate this with generate_signal() and uniform binning.
    """

    def __init__(self, param_names: list[str], n_bins: int = 10):
        self.param_names = param_names
        self.n_bins = n_bins

    def __call__(self, params: np.ndarray) -> np.ndarray:
        signal = generate_signal(
            params,
            self.param_names,
            baseline_value=1.5,
            n_timesteps=self.n_bins * 40,
            random_seed=int(abs(params.sum() * 1000)) % (2**31),
        )
        n_per_bin = len(signal) // self.n_bins
        stage_means = np.array([signal[s * n_per_bin : (s + 1) * n_per_bin].mean() for s in range(self.n_bins)])
        return stage_means

    def evaluate_batch(self, X: np.ndarray) -> np.ndarray:
        return np.vstack([self(x) for x in X])


# =============================================================================
# Pipeline Configuration
# =============================================================================


@dataclass
class XSpaceConfigVecoli(BaseClass):
    vio_params: VioPathwayParams
    mec_params: MecillinamParams
    ko_params: GeneKnockoutParams
    vio_expression_bounds: tuple[float, float]
    vio_trl_eff_bounds: tuple[float, float]
    mecillinam_conc_bounds: tuple[float, float]
    generations: int


# =============================================================================
# Pipeline Execution
# =============================================================================


def run_full_uq_workflow(
    xspace_config: XSpaceConfigVecoli,
    output_dir: Path = Path("./uq_results"),
    n_samples: int = 50,
    polynomial_order: int = 2,
    n_bins: int = 10,
) -> PipelineResult:
    """
    Run the complete RFC006 UQ pipeline using actual uq.pipeline module code.

    Steps 1-4 are sequential and shared. Then Phase 1 (population-level GSA)
    and Phase 2 (cell-cycle-stratified GSA) run, producing a PipelineResult
    with two UqProfile instances.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    observable_columns = ["listeners__mass__dry_mass", "listeners__fba_results__growth"]

    print("=" * 80)
    print("vEcoli UQ Framework — Full RFC006 Pipeline")
    print("Milestone 08.4.2: Uncertainty Quantification Framework")
    print("=" * 80)

    # ── Step 1: Load datasets ──

    print("\n" + "=" * 80)
    print("STEP 1: Load Simulation Data")
    print("=" * 80)
    sim_data = generate_synthetic_simulation_data()
    print(f"    Generated {len(sim_data)} data points")
    print(f"    Experiments: {sim_data['experiment_id'].unique().to_list()}")
    print(f"    Seeds: {sim_data['lineage_seed'].unique().to_list()}")

    # ── Step 2: Define parameter space Ξ ──

    print("\n" + "=" * 80)
    print("STEP 2: Define Input Parameter Space (Ξ)")
    print("=" * 80)
    param_space = XSpaceVecoli(
        include_vio=True,
        include_mecillinam=True,
        vio_expression_bounds=xspace_config.vio_expression_bounds,
        vio_trl_eff_bounds=xspace_config.vio_trl_eff_bounds,
        mecillinam_conc_bounds=xspace_config.mecillinam_conc_bounds,
    )
    print(f"    Parameters: {param_space.parameter_names}")
    print(f"    Bounds: {param_space.parameter_bounds}")

    # ── Step 3: Aggregate outputs (uses uq.pipeline.workflow.aggregate_timeseries) ──

    print("\n" + "=" * 80)
    print("STEP 3: Aggregate (Strategies 1-3)")
    print("=" * 80)
    agg_result = aggregate_timeseries(sim_data, observable_columns)
    print(f"    Strategy 1 (UNIFORM): mean={agg_result.uniform.mean}, n={agg_result.uniform.n_samples}")
    print(f"    Strategy 2 (BY_GENERATION): {len(agg_result.generation.groups)} groups")
    print(f"    Strategy 3 (BY_LINEAGE_SEED): {len(agg_result.seed.groups)} groups")

    # ── Step 4: Variance decomposition (uses uq.pipeline.workflow.get_variance_decomposition) ──

    print("\n" + "=" * 80)
    print("STEP 4: Variance Decomposition")
    print("=" * 80)
    decomp = get_variance_decomposition(agg_result)
    gen_pct = np.mean(decomp["generation_fraction"]) * 100
    seed_pct = np.mean(decomp["seed_fraction"]) * 100
    residual_pct = max(0, 100 - gen_pct - seed_pct)
    print(f"    Generation effects:  {gen_pct:.1f}% of variance")
    print(f"    Stochastic seeding:  {seed_pct:.1f}% of variance")
    print(f"    Residual (cell cycle): {residual_pct:.1f}% of variance")
    print("    (Residual fraction → feeds into Phase 2 observable selection)")

    # ── Phase 1: Population-level GSA (uses uq.pipeline.workflow.run_phase1) ──

    print("\n" + "=" * 80)
    print("PHASE 1: Population-Level Sensitivity Analysis")
    print("=" * 80)
    wrapper_bulk = SyntheticBulkWrapper(param_space.parameter_names)
    print(f"    Building PCE surrogate (order={polynomial_order}, samples={n_samples})...")
    sobol_bulk, surrogate_bulk, _morris = run_phase1(
        param_space=param_space,
        simulation_func=wrapper_bulk,
        polynomial_order=polynomial_order,
        n_samples=n_samples,
    )

    print("\n    Phase 1 Sobol Indices (bulk):")
    print(f"    {'Parameter':<30s} {'S_i (first)':>12s} {'S_Ti (total)':>12s}")
    print("    " + "-" * 56)
    fo = sobol_bulk.first_order.flatten()
    to = sobol_bulk.total_order.flatten()
    for i, name in enumerate(sobol_bulk.parameter_names):
        si = fo[i] if i < len(fo) else 0.0
        sti = to[i] if i < len(to) else 0.0
        bar = "█" * int(abs(sti) * 40)
        print(f"    {name:<30s} {si:>12.4f} {sti:>12.4f}  {bar}")
    print(
        f"\n    PCE surrogate: R² = {surrogate_bulk.r_squared:.4f}, "
        f"dim = {surrogate_bulk.input_dim}→{surrogate_bulk.output_dim}"
    )

    # ── Phase 2: Cell-cycle-stratified GSA (Steps 5b-7b) ──
    #
    # Uses uq.pipeline.workflow.compute_strategy4_sobol and
    # uq.pipeline.workflow._split_multi_output_sobol for Sobol splitting.
    #
    # Note: With real data, you'd use uq.pipeline.workflow.run_phase2() which
    # chains Steps 5b → 6b → 6d → 7b automatically. Here we use a synthetic
    # Strategy4Wrapper instead of the real one (which requires Koopman DMD).

    print("\n" + "=" * 80)
    print("PHASE 2: Cell-Cycle-Stratified Sensitivity Analysis")
    print("=" * 80)

    # Step 5b: GSA-informed observable selection
    relevance = identify_cell_cycle_relevant_observables(
        aggregated_uniform=agg_result.uniform,
        aggregated_by_gen=agg_result.generation,
        aggregated_by_seed=agg_result.seed,
        observable_names=observable_columns,
    )
    print(f"    Step 5b: GSA-informed observable selection")
    print(f"        Relevant observables: {relevance.relevant_observables}")
    for obs, score in relevance.relevance_scores.items():
        print(f"        {obs}: residual_fraction = {score:.4f}")

    selected_obs = relevance.relevant_observables or observable_columns
    print(f"\n    Step 6b: Koopman cell cycle variable")
    print(f"        Using observables: {selected_obs}")
    print(f"        θ(x) = arg(φ_cc) / 2π  ∈ [0, 1]")

    # Step 6d + 7b: Strategy 4 wrapper → PCE + Sobol
    print(f"\n    Step 6d: Strategy 4 Wrapper (params → {n_bins} stage means)")
    f_stage4 = SyntheticStrategy4Wrapper(param_names=param_space.parameter_names, n_bins=n_bins)
    print(f"\n    Step 7b: PCE + Sobol on Strategy 4 (order={polynomial_order}, samples={n_samples})")
    per_stage_sobol, surrogate_cc = compute_strategy4_sobol(
        param_space=param_space,
        f_stage4=f_stage4,
        polynomial_order=polynomial_order,
        n_samples=n_samples,
    )

    # Print per-stage results
    print(f"\n    Phase 2 Sobol Indices ({len(per_stage_sobol)} stages):")
    print(f"    {'Stage':<8s} ", end="")
    for name in param_space.parameter_names:
        print(f"{name:>20s} ", end="")
    print()
    print("    " + "-" * (8 + 21 * len(param_space.parameter_names)))
    for i, s in enumerate(per_stage_sobol):
        vals = s.total_order.flatten()
        print(f"    {f'θ={i / len(per_stage_sobol):.1f}':<8s} ", end="")
        for j in range(len(param_space.parameter_names)):
            v = vals[j] if j < len(vals) else 0.0
            print(f"{v:>20.4f} ", end="")
        print()
    print(
        f"\n    PCE surrogate (phenotypic): R² = {surrogate_cc.r_squared:.4f}, "
        f"dim = {surrogate_cc.input_dim}→{surrogate_cc.output_dim}"
    )

    # ── Assemble PipelineResult ──

    pipeline_result = PipelineResult(
        population=UqProfile(
            stratification=StratificationLens.POPULATION,
            sobol_indices=[sobol_bulk],
            surrogate=surrogate_bulk,
        ),
        cell_cycle=UqProfile(
            stratification=StratificationLens.CELL_CYCLE,
            sobol_indices=per_stage_sobol,
            surrogate=surrogate_cc,
        ),
    )

    # ── Bonus: Koopman spectral analysis ──

    print("\n" + "=" * 80)
    print("BONUS: Koopman Spectral Analysis")
    print("=" * 80)
    trajectory_data = (
        sim_data.filter((pl.col("experiment_id") == 0) & (pl.col("lineage_seed") == 0))
        .sort("time")
        .select(observable_columns)
        .to_numpy()
    )
    print(f"    Trajectory shape: {trajectory_data.shape}")
    dmd = DynamicModeDecomposition(rank=5)
    dmd.fit(trajectory_data)
    spectrum = dmd.get_spectrum(observable_names=["mass", "growth_rate"])
    print(f"    Extracted {len(spectrum.modes)} Koopman modes:")
    for i, mode in enumerate(spectrum.get_dominant_modes(3)):
        print(
            f"      Mode {i + 1}: freq={mode.frequency:.6f} Hz, "
            f"|amplitude|={abs(mode.amplitude):.4f}, "
            f"{'oscillatory' if mode.is_oscillatory else 'non-oscillatory'}"
        )

    # ── Summary ──

    print("\n" + "=" * 80)
    print("PIPELINE RESULT")
    print("=" * 80)

    print(f"""
    PipelineResult assembled:

    ┌─ Population (Phase 1, bulk) ──────────────────────────────────────┐
    │  Stratification: {pipeline_result.population.stratification}
    │  SobolIndices:   {len(pipeline_result.population.sobol_indices)} set (S_i, S_Ti for {len(sobol_bulk.parameter_names)} params)
    │  PCESurrogate:   order={surrogate_bulk.polynomial_order}, R²={surrogate_bulk.r_squared:.4f}
    │  Answer:         "Which parameters drive bulk output variance?"
    └───────────────────────────────────────────────────────────────────┘

    ┌─ Cell Cycle (Phase 2, phenotypic) ────────────────────────────────┐
    │  Stratification: {pipeline_result.cell_cycle.stratification}
    │  SobolIndices:   {len(pipeline_result.cell_cycle.sobol_indices)} sets (one per θ-bin)
    │  PCESurrogate:   order={surrogate_cc.polynomial_order}, R²={surrogate_cc.r_squared:.4f}
    │  Answer:         "Which parameters drive variance WITHIN each
    │                   cell cycle stage?"
    └───────────────────────────────────────────────────────────────────┘

    Example interpretation:
      Phase 1: "vio_expression drives {abs(sobol_bulk.total_order.flatten()[0]):.0%} of bulk mass variance"
      Phase 2: "During early cell cycle (θ≈0), parameter importance shifts —
                see per-stage Sobol table above"
    """)

    # ── Save results via PipelineResult.export() ──

    print(f"    Exporting PipelineResult to: {output_dir}")
    pipeline_result.export(output_dir)
    print(f"    Export complete. Contents:")
    for p in sorted(output_dir.rglob("*")):
        if p.is_file():
            print(f"      {p.relative_to(output_dir)}")

    # Also save human-readable JSON summary
    results = {
        "variance_decomposition": {
            "generation_fraction": decomp["generation_fraction"].tolist(),
            "seed_fraction": decomp["seed_fraction"].tolist(),
        },
        "phase1_sobol": {
            "first_order": dict(zip(sobol_bulk.parameter_names, sobol_bulk.first_order.flatten().tolist())),
            "total_order": dict(zip(sobol_bulk.parameter_names, sobol_bulk.total_order.flatten().tolist())),
        },
        "phase2_sobol_per_stage": [
            {
                "stage": i,
                "total_order": dict(zip(s.parameter_names, s.total_order.flatten().tolist())),
            }
            for i, s in enumerate(per_stage_sobol)
        ],
        "pipeline_summary": {
            "population_surrogate_r2": surrogate_bulk.r_squared,
            "cell_cycle_surrogate_r2": surrogate_cc.r_squared,
            "n_cell_cycle_stages": len(per_stage_sobol),
            "n_parameters": len(param_space.parameter_names),
            "parameter_names": param_space.parameter_names,
        },
    }
    results_path = output_dir / "uq_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"    Human-readable results: {results_path}")

    # ── Verify round-trip: PipelineResult.from_export() ──

    print("\n    Verifying PipelineResult.from_export() round-trip...")
    loaded = PipelineResult.from_export(output_dir)
    assert len(loaded.population.sobol_indices) == 1
    assert len(loaded.cell_cycle.sobol_indices) == len(per_stage_sobol)
    print("    Round-trip verification passed.")

    print("\n" + "=" * 80)
    print("RFC006 Pipeline Complete")
    print("=" * 80)

    return pipeline_result


# =============================================================================
# Entry Point
# =============================================================================


def main():
    parser = argparse.ArgumentParser(description="Run full RFC006 UQ pipeline")
    parser.add_argument("--output-dir", type=str, default="./uq_results", help="Output directory")
    args = parser.parse_args()

    xspace_config = XSpaceConfigVecoli(
        vio_params=VioPathwayParams(expression=2.5, translation_efficiency=1.2),
        mec_params=MecillinamParams(times=[0.0, 3600.0], concentrations=[0.0, 5.0]),
        ko_params=GeneKnockoutParams(),
        vio_expression_bounds=(0.0, 5.0),
        vio_trl_eff_bounds=(0.0, 2.0),
        mecillinam_conc_bounds=(0.0, 10.0),
        generations=8,
    )

    run_full_uq_workflow(
        xspace_config=xspace_config,
        output_dir=Path(args.output_dir),
    )


if __name__ == "__main__":
    main()
