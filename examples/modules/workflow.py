import marimo

__generated_with = "0.21.1"
app = marimo.App(width="full")


@app.cell
def _():
    import json as _json
    from pathlib import Path
    from typing import Callable

    import marimo as mo

    from libuq import XSpaceVecoli
    from libuq.api.models import SystemConfig, Parameter, Observable
    from libuq.pipe import initialize_datasets, DatasetMultiExperiment
    from libuq.pipeline.models import SimDataParameter
    from libuq.pipeline.workflow import aggregate_timeseries
    from libuq.sampling import run_and_cache
    from libuq.wrappers import DataDrivenWrapper

    return (
        DatasetMultiExperiment,
        Observable,
        Parameter,
        Path,
        SystemConfig,
        XSpaceVecoli,
        initialize_datasets,
        mo,
    )


@app.cell
def _(mo):
    sim_base_path = mo.ui.text(label="Simulation Output Directory", debounce=True, value="/Users/alexanderpatrie/sms/vecoli_data/outputs")
    return (sim_base_path,)


@app.cell
def _(Path, mo, sim_base_path):
    paths = []
    if sim_base_path.value:
        paths = [{"experiment_id": p.name, "path": str(p)} for p in Path(sim_base_path.value).absolute().iterdir() if p.is_dir()]

    table = mo.ui.table(data=paths, initial_selection=list(range(len(paths))))
    return (table,)


@app.cell
def _(mo, sim_base_path, table):
    def experiment_ids():
        return [v['experiment_id'] for v in table.value]

    def base_outdir():
        return sim_base_path.value

    mo.vstack([sim_base_path, table])
    return base_outdir, experiment_ids


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### UQ Pipeline Stage 1: Precompute/generate samples
    """)
    return


@app.cell
def _(experiment_ids):
    experiment_ids()
    return


@app.cell
def _(
    DatasetMultiExperiment,
    Observable,
    Parameter,
    SystemConfig,
    XSpaceVecoli,
    base_outdir,
    experiment_ids,
    initialize_datasets,
):
    def initialize_data(config: SystemConfig) -> tuple[XSpaceVecoli, DatasetMultiExperiment]:
        """TODO: Put this in uq_simple.pipeline module"""
        hive_colnames = config.observable_column_names()
        ds = initialize_datasets(
            experiment_ids=experiment_ids(),
            sim_base_path=base_outdir(),
            observable_columns=hive_colnames,
        )

        sim_data_parameters = config.sim_data_parameters()
        param_space = ds.x[0].to_parameter_space(
            parameters=sim_data_parameters,
        )
        if param_space.n_parameters == 0:
            raise RuntimeError(
                "Parameter space is empty. Provide --params-file with SimDataParameter specs."
            )
        return param_space, ds


    # select specific params for UQ pipeline
    parameters: list[Parameter] = [
        Parameter(
            name="fraction_active_rnap_free",
            attr_path="process.transcription.fraction_active_rnap_free",
            bounds=(0.25, 0.47),
            description="Fraction of RNAP that is active when ppGpp-free (baseline ~0.36)",
        ),
        Parameter(
            name="fraction_active_rnap_bound",
            attr_path="process.transcription.fraction_active_rnap_bound",
            bounds=(0.12, 0.22),
            description="Fraction of RNAP that is active when ppGpp-bound (baseline ~0.17)"
        ),
        Parameter(
            name="cell_dry_mass_fraction",
            attr_path="mass.cell_dry_mass_fraction",
            bounds=(0.25, 0.35),
            description="Fraction of total cell mass that is dry mass (baseline ~0.30)",
        ),
        Parameter(
            name="kinetic_objective_weight",
            attr_path="process.metabolism.kinetic_objective_weight",
            bounds=(1e-8, 1e-6), 
            description="FBA kinetic vs homeostatic objective weight"
        )
    ]


    # select specific observables for UQ pipeline
    observable_columns = [
        "listeners.mass.dry_mass",
        "listeners.mass.cell_mass",
        "listeners.mass.volume",
        "listeners.mass.growth"
    ]

    observables: list[Observable] = [
        Observable(name=c.split(".")[-1], attr_path=c, description=c)
        for c in observable_columns
    ]


    # parameterize api dto (for convienience)
    config = SystemConfig(
        parameters=parameters, 
        observables=observables
    )

    # ---> define parameter space for uq and load experiment dataset(s)
    parameter_space, dataset = initialize_data(config)

    parameter_space.show()
    print(dataset.y)
    return dataset, observable_columns, parameter_space


@app.cell
def _(Path, dataset, observable_columns, parameter_space):
    # define timeseries generator as wrapper and generate samples

    from libuq.generators.vecoli import TimeseriesGeneratorVecoli
    from libuq.sampling import run_batch_and_cache

    seeds = 1
    generations = 1
    duration = 10800.0
    n_samples = 22
    cache_dir = Path("uq_results")
    max_workers = 4
    batch_dir = Path("samples")

    sim_func = TimeseriesGeneratorVecoli(
        baseline_sim_data=dataset.x[0].sim_data,
        param_space=parameter_space,
        max_duration=duration,
        generations=generations,
        n_init_sims=seeds,
        output_keys=[c.split(".")[-1] for c in observable_columns],
    )

    cache = run_batch_and_cache(
        parameter_space=parameter_space,
        simulation_func=sim_func,
        n_samples=n_samples,
        cache_dir=Path(cache_dir),
        seed=seeds,
        max_workers=max_workers,
        batch_dir=batch_dir
    )
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
