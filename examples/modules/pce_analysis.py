import marimo

__generated_with = "0.16.4"
app = marimo.App(width="full")


@app.cell
def _(mo):
    mo.md(
        r"""
    1. Sobol index extraction from the PCE coefficients
    2. Integration with aggregation strategies (the y should come from aggregated simulation outputs)
    3. Scientifically meaningful inputs (vio, mecillinam, knockouts - not generic x)

    ### How the UQ Package Uses This

      In uq/sensitivity.py, the following pattern is wrapped:

    ```python
      # Our wrapper around PyTUQ PCE
      def analyze_with_pce(self, polynomial_order=3, n_samples=100, use_uqpy=True):
          if not use_uqpy:
              # PyTUQ path - similar to your code
              from pytuq.pce import PCE

              X = self._generate_samples(n_samples)        # Your: x = scale01ToDom(...)
              Y = self.wrapper(X)                          # Your: y = true_model(x)

              pce = PCE(self.param_space.n_parameters, polynomial_order, 'LU')
              pce.set_training_data(X, Y)
              pce.build()

              # Extract Sobol indices from PCE coefficients (CONTEXT.md requirement)
              sobol_indices = self._compute_sobol_from_pce(pce)

              return sobol_indices, PCESurrogate(coefficients=pce.coeffs, ...)
    ```

    **Bottom line**: Your code handles the PCE surrogate construction requirement. It needs to be
    wrapped with the aggregation strategies and Sobol extraction to fully satisfy Milestone 08.4.2.
    """
    )
    return


@app.cell
def _():
    return


@app.cell
def _():
    import marimo as mo

    return (mo,)


if __name__ == "__main__":
    app.run()
