import marimo

__generated_with = "0.20.4"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo
    import uuid

    return mo, uuid


@app.cell
def _(mo, uuid):
    exp_id = mo.ui.text(label="Experiment ID: ", value="experiment-" + str(uuid.uuid1(node=1111223))[:22])
    exp_id
    return (exp_id,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## A. "Implement uncertainty quantification framework to track prediction confidence" (RFC006 §1, MS-08.4.2)

    #### The uq/ package as a whole implements this framework. The 7-step pipeline (parameter space -> aggregation -> variance decomposition -> Morris screening -> PCE surrogate -> Sobol indices) constitutes the UQ framework for tracking prediction confidence.
    """)
    return


@app.cell
def _(exp_id):
    from libuq import XSpaceVecoli, SensitivityAnalyzer, AggregationStrategy, SimulationWrapper, WrapperConfig

    class x:
        timesteps = 1111
        a = 0.22
        b = 3.0

    def f(_x):
        import numpy as np

        return np.random.random((_x.timesteps, 2)) * _x.a / _x.b

    param_space = XSpaceVecoli(include_vio=True, include_mecillinam=True, experiment_id=exp_id.value)
    param_space.show()
    wrapper = SimulationWrapper(
        parameter_space=param_space,
        config=WrapperConfig(
            sim_data_path="/Users/alexanderpatrie/sms/vEcoli/api_integration/sims/api_simulation_default/parca/kb/simData.cPickle"
        ),
    )
    analyzer = SensitivityAnalyzer(param_space, wrapper=wrapper)
    sobol_indices, pce_surrogate = analyzer.analyze_with_pce(polynomial_order=3, n_samples=100)
    # Track prediction confidence: surrogate predicts with R² quality metric
    print(f"Surrogate R²: {pce_surrogate.r_squared}")
    top = sobol_indices.select(n=5)
    print(f"Most influential parameters: {top}")
    return f, param_space, x


@app.cell
def _(f, x):
    f(x)
    return


@app.cell
def _(param_space):
    dir(param_space)
    return


@app.cell
def _():
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
