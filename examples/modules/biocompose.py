import marimo

__generated_with = "0.20.4"
app = marimo.App(width="full", layout_file="../assets/layouts/biocompose.grid.json")


@app.cell
def _():
    import dataclasses as dc
    import pprint
    import typing as ty
    from enum import StrEnum

    import marimo as mo
    import numpy as np
    import polars as pl

    return StrEnum, dc, mo, np, pl, pprint


@app.cell
def _(N, StrEnum, T, dc, dt, np, pl, pprint, t):
    class BinBand(StrEnum):
        LOW = "low"
        MID = "mid"
        HIGH = "high"

    @dc.dataclass
    class Bins:
        # TODO: instead of low, mid, high have continuous spectrum (with granularity)
        low: np.ndarray
        mid: np.ndarray
        high: np.ndarray

        def vectorize(self):
            bins = [getattr(self, binType) for binType in [BinBand.LOW, BinBand.MID, BinBand.HIGH]]
            vec = []
            for row in bins:
                for val in row:
                    vec.append(val)
            # self.vector = np.ndarray(vec)
            return np.arange(self.low.min(), self.high.max(), 1)

    class Spectrum:
        data: np.ndarray
        dt: float

        def __init__(self, data, dt: float):
            self.data = data
            self.dt = dt

        def __repr__(self) -> str:
            bins = self.bins

            def get_range(r: str):
                v = getattr(bins, r)
                return f"{int(v.min())} - {int(v.max())}"

            low, mid, high = list(map(lambda c: get_range(c), ["low", "mid", "high"]))
            return f"Spectrum:\n\n{pprint.pformat(self.data)}\n=====\nBins:\n\nlow: {low}\nmid: {mid}\nhigh: {high}"

        @property
        def frequencies(self):
            # freqs is now in Hz, same length as spectrum
            sample_rate: int = 44100
            # d=1/sample_rate
            return np.fft.rfftfreq(len(self.data), d=1 / sample_rate)

        @property
        def bins(self):
            mid_thresh = 300
            high_thresh = mid_thresh * 10
            freqs = self.frequencies
            low_bins = np.where(freqs < mid_thresh)[0]
            mid_bins = np.where((freqs >= mid_thresh) & (freqs < high_thresh))[0]
            high_bins = np.where(freqs >= high_thresh)[0]
            return Bins(low=low_bins, mid=mid_bins, high=high_bins)

        # def adjust(self, band: BinBand, dg: float, i_bin: int | None = None):
        #     selected_bins = getattr(self.bins, band)
        #     if i_bin is not None:
        #         selected_bins = selected_bins[i_bin]
        #     self.data[selected_bins] *= dg

        def adjust(self, dg: float, band: BinBand | None = None, i_bin: int | None = None):
            if i_bin is not None:
                selected_bin = self.bins.vectorize()[i_bin]
                self.data[selected_bin] *= dg
            else:
                selected_bins = getattr(self.bins, band)
                self.data[selected_bins] *= dg

        @property
        def magnitude(self):
            return 20 * np.log10(np.abs(self.data[1 : len(self.frequencies)]) + 1e-10)

        def plot(self, obs: str):
            import plotly.graph_objects as go

            freqs = self.frequencies[1:]
            magnitude_db = self.magnitude

            fig = go.Figure()
            fig.add_trace(go.Scatter(x=freqs, y=magnitude_db, fill="tozeroy", line=dict(color="cyan", width=1)))
            fig.update_xaxes(type="log", range=[np.log10(20), np.log10(20000)])
            fig.update_layout(template="plotly_dark", title=f"Spectrum Analyzer: {obs}")
            return fig

    class TimeseriesDataset:
        df: pl.DataFrame
        t: np.ndarray[float]
        observables: list[str]

        def __init__(self, data: pl.DataFrame, t_colname: str = "time"):
            self.df = data
            self.t = self.df.select(t_colname).to_numpy()
            self.dt = self.t[1] - self.t[0]
            self.observables = [c for c in self.df.columns if not c == t_colname]

        @property
        def timeseries(self) -> np.ndarray:
            return self.df.select(self.observables)

        def to_spectrum(self, observable_name: str) -> Spectrum:
            if observable_name not in self.observables:
                raise ValueError(f"{observable_name} not found in data cols. Valid options: {self.observables}")
            ts = self.df.select(observable_name).to_numpy().flatten()
            return Spectrum(data=np.fft.fft(ts), dt=self.dt)  # shape (T, N)

        def from_spectrum(self, s: Spectrum) -> np.ndarray:
            return np.fft.ifft(s.data).real

        def __repr__(self) -> str:
            return pprint.pformat(self.df)

    def test_spectral_transform():
        _T = 1111  # total duration
        _dt = 1.0  # global timestep/interval
        _t = (lambda T, dt: np.arange(start=0.0, stop=float(T), step=dt))(T, dt)

        _N = 3  # num observables
        _observable_names = ["a", "b", "c"]
        _timeseries_dataset = pl.DataFrame({"time": t, **dict(zip(observable_names, np.random.random((N, T))))})
        ds = TimeseriesDataset(data=_timeseries_dataset)
        _spectrum = ds.to_spectrum("a")
        _ts_a = ds.df.select("a").to_numpy()
        ds.from_spectrum(_spectrum) == _ts_a
        assert np.all(_ts_a == ds.df.select("a").to_numpy())

    # loading fake data example
    # T = 1111  # total duration
    # dt = 1.0  # global timestep/interval
    # t = (lambda T, dt: np.arange(start=0.0, stop=float(T), step=dt))(T, dt)
    # N = 3  # num observables
    # observable_names = ['a', 'b', 'c']
    # timeseries_dataset = pl.DataFrame({
    #     "time": t,
    #     **dict(zip(
    #         observable_names,
    #         np.random.random((N, T))
    #     ))
    # })

    # instead, load the real deal :)
    from pathlib import Path

    from libuq.inputs import load_dataset

    def load_trajectory():
        import json

        with open("baseline_observables.json") as f:
            obs = [col for col in json.load(f) if col.startswith("listener")]
        obs.append("time")
        X = load_dataset(
            experiment_id="api_simulation_default",
            outdir_root=Path("/Users/alexanderpatrie/sms/sms-api/artifacts/sims"),
            observables=obs,
        )
        observables = []
        schema = X.collect_schema()
        for obs_i in obs:
            coltype = schema[obs_i]
            if isinstance(coltype, pl.Float64):
                observables.append(obs_i)
        X = X.select(observables)
        return X

    X = load_trajectory()
    t_range = X.select("time").to_numpy()

    # timeseries = TimeseriesDataset(data=timeseries_dataset)
    timeseries = TimeseriesDataset(data=X, t_colname="time")
    observable_names = timeseries.observables
    return BinBand, Spectrum, timeseries


@app.cell
def _(timeseries):
    spectrum = timeseries.to_spectrum("listeners__mass__cell_mass")
    return (spectrum,)


@app.cell
def _(spectrum, timeseries):
    ts_a = timeseries.from_spectrum(spectrum)
    return


@app.cell
def _(mo):
    low_slider = mo.ui.slider(label="LOW", value=1.0, start=0.1, stop=10.0, step=0.1, show_value=True)
    mid_slider = mo.ui.slider(label="MID", value=1.0, start=0.1, stop=10.0, step=0.1, show_value=True)
    high_slider = mo.ui.slider(label="HIGH", value=1.0, start=0.1, stop=10.0, step=0.1, show_value=True)
    return


@app.cell
def _(mo, observable_names):
    obs_dropdown = mo.ui.dropdown(
        label="observable name:", options=observable_names, value="listeners__mass__cell_mass"
    )
    return (obs_dropdown,)


@app.cell
def _(BinBand, Spectrum, mo, spectrum):
    class AllSliders(dict):
        def flatten(self):
            s = []
            for row in self.values():
                for r in row:
                    s.append(r)
            return s

    class BinSliders:
        def generate_range_sliders(self, band: BinBand, spectrum: Spectrum) -> list[mo.ui.slider]:
            sliders = []
            bins = getattr(spectrum.bins, band)
            for i, b in enumerate(bins):
                freqs = spectrum.frequencies[bins]
                bin_min = -freqs.min()
                bin_max = freqs.max() * 4
                on_change = lambda val: spectrum
                b_slider = mo.ui.slider(
                    full_width=True,
                    label=f"     {band.value}:{b}     ",
                    start=bin_min,
                    stop=bin_max,
                    step=1.0,
                    show_value=True,
                    value=spectrum.frequencies[bins][i],
                )
                sliders.append(b_slider)
            return sliders

        def get_all_sliders(self):
            bands = [BinBand.LOW, BinBand.MID, BinBand.HIGH]
            return dict(
                zip(
                    bands,
                    [self.generate_range_sliders(band, spectrum) for band in bands],
                )
            )

        @property
        def all(self):
            return AllSliders(self.get_all_sliders())

        def get_ui(self):
            return list(
                map(
                    # lambda band: mo.accordion({band: self.generate_range_sliders(band, spectrum)}),
                    lambda band: self.generate_range_sliders(band, spectrum),
                    [BinBand.LOW, BinBand.MID, BinBand.HIGH],
                )
            )

        @property
        def ui(self):
            return mo.vstack(self.get_ui(), justify="start")

    sliders = BinSliders()
    return


@app.cell
def _(mo, np):
    from scipy.interpolate import interp1d

    # Create 12 gain sliders for control points across the frequency spectrum
    N_CONTROL_POINTS = 22
    control_freqs = np.logspace(np.log10(20), np.log10(20000), N_CONTROL_POINTS)

    # Create labels for each frequency band
    def _get_freq_label(freq):
        if freq >= 1000:
            return f"{freq / 1000:.1f}kHz"
        return f"{freq:.0f}Hz"

    # Use mo.ui.array for proper reactivity - this makes the whole array reactive
    gain_sliders = mo.ui.array([
        mo.ui.slider(
            orientation="vertical",
            start=0.1,
            stop=4.0,
            step=0.05,
            value=1.0,
            label=_get_freq_label(freq),
            show_value=True,
        )
        for freq in control_freqs
    ])
    return control_freqs, gain_sliders, interp1d


@app.cell
def _(
    control_freqs,
    gain_sliders,
    interp1d,
    mo,
    np,
    obs_dropdown,
    pprint,
    timeseries,
):
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    class InteractiveSpectrum:
        """Spectrum with continuous gain control via interpolated control points."""

        def __init__(self, spectrum, control_freqs: np.ndarray, control_gains: np.ndarray):
            self.spectrum = spectrum
            self.freqs = spectrum.frequencies[1:]  # exclude DC
            self.control_freqs = control_freqs
            self.control_gains = control_gains

        def get_gain_curve(self) -> np.ndarray:
            """Interpolate control points to get gain for every frequency bin."""
            log_control = np.log10(self.control_freqs)
            log_freqs = np.log10(np.clip(self.freqs, 1, None))

            interp_func = interp1d(
                log_control,
                self.control_gains,
                kind="cubic",
                bounds_error=False,
                fill_value=(self.control_gains[0], self.control_gains[-1]),
            )
            return interp_func(log_freqs)

        def apply_gains(self) -> np.ndarray:
            """Apply interpolated gains to spectrum data."""
            gain_curve = self.get_gain_curve()
            modified_data = self.spectrum.data.copy()
            modified_data[1 : len(self.freqs) + 1] *= gain_curve
            return modified_data

        def get_magnitude_db(self, data: np.ndarray | None = None) -> np.ndarray:
            """Get magnitude in dB."""
            if data is None:
                data = self.apply_gains()
            return 20 * np.log10(np.abs(data[1 : len(self.freqs) + 1]) + 1e-10)

        def to_timeseries(self, modified_data: np.ndarray) -> np.ndarray:
            """Reconstruct timeseries from modified spectrum via inverse FFT."""
            return np.fft.ifft(modified_data).real

    # Read current gain values from sliders (this triggers reactivity!)
    current_gains = np.array(gain_sliders.value)

    # Get original timeseries
    _obs_name = obs_dropdown.value
    _original_ts = timeseries.df.select(_obs_name).to_numpy().flatten()
    _time = timeseries.t.flatten()

    # Create spectrum and apply gains
    _spectrum = timeseries.to_spectrum(_obs_name)
    interactive = InteractiveSpectrum(_spectrum, control_freqs, current_gains)

    # Get original and modified magnitudes
    original_mag = _spectrum.magnitude
    modified_spectrum_data = interactive.apply_gains()
    modified_mag = interactive.get_magnitude_db(modified_spectrum_data)
    freqs = interactive.freqs

    # Reconstruct modified timeseries via inverse FFT
    _modified_ts = interactive.to_timeseries(modified_spectrum_data)

    # =========================================================================
    # BUILD UNIFIED FIGURE with spectrum on top, timeseries on bottom
    # =========================================================================
    combined_fig = make_subplots(
        rows=2,
        cols=1,
        row_heights=[0.5, 0.5],
        subplot_titles=["Timeseries (reconstructed from spectrum)", "Frequency Spectrum (drag sliders to modify)"],
        vertical_spacing=0.12,
    )

    # ----- ROW 2: SPECTRUM -----
    # Original spectrum (dimmed)
    combined_fig.add_trace(
        go.Scatter(
            x=freqs,
            y=original_mag,
            fill="tozeroy",
            name="Original Spectrum",
            line=dict(color="rgba(100, 100, 100, 0.4)", width=1),
            fillcolor="rgba(100, 100, 100, 0.2)",
            legendgroup="spectrum",
        ),
        row=2,
        col=1,
    )

    # Modified spectrum (bright cyan)
    combined_fig.add_trace(
        go.Scatter(
            x=freqs,
            y=modified_mag,
            fill="tozeroy",
            name="Modified Spectrum",
            line=dict(color="cyan", width=1),
            fillcolor="rgba(0, 255, 255, 0.3)",
            legendgroup="spectrum",
        ),
        row=2,
        col=1,
    )

    # Control points overlaid on spectrum
    marker_y = np.interp(control_freqs, freqs, modified_mag)
    combined_fig.add_trace(
        go.Scatter(
            x=control_freqs,
            y=marker_y,
            mode="markers+lines",
            name="EQ Control Points",
            marker=dict(size=10, color="yellow", symbol="circle", line=dict(color="orange", width=2)),
            line=dict(color="rgba(255, 255, 0, 0.3)", width=1, dash="dot"),
            hovertemplate="%{x:.0f} Hz<br>Gain: %{customdata:.2f}<extra></extra>",
            customdata=current_gains,
            legendgroup="spectrum",
        ),
        row=2,
        col=1,
    )

    # ----- ROW 1: TIMESERIES -----
    # Original timeseries
    combined_fig.add_trace(
        go.Scatter(
            x=_time,
            y=_original_ts,
            name="Original Timeseries",
            line=dict(color="rgba(0, 200, 255, 0.8)", width=1.5),
            legendgroup="timeseries",
        ),
        row=1,
        col=1,
    )

    # Modified timeseries (bright magenta)
    combined_fig.add_trace(
        go.Scatter(
            x=_time,
            y=_modified_ts,
            name="Modified Timeseries",
            line=dict(color="magenta", width=1),
            legendgroup="timeseries",
        ),
        row=1,
        col=1,
    )

    # ----- LAYOUT -----
    combined_fig.update_xaxes(type="log", range=[np.log10(20), np.log10(20000)], title="Frequency (Hz)", row=2, col=1)
    combined_fig.update_yaxes(title="Magnitude (dB)", row=2, col=1)

    combined_fig.update_xaxes(title="Time", row=1, col=1)
    combined_fig.update_yaxes(title="Value", row=1, col=1)

    combined_fig.update_layout(
        template="plotly_dark",
        height=700,
        showlegend=True,
        legend=dict(x=1.02, y=1, xanchor="left"),
        title=f"Interactive Spectrum Analyzer: {_obs_name}",
    )

    # =========================================================================
    # UNIFIED LAYOUT with sliders on left, plots on right
    # =========================================================================
    _slider_panel = mo.vstack(
        [
            mo.md("### EQ Controls"),
            obs_dropdown,
            mo.md("**Frequency Gains:**"),
            # *list(gain_sliders),  # All sliders stacked vertically
            mo.md("**Gains:**\n"),
            mo.md(f"{pprint.pformat([f'{g:.2f}' for g in current_gains])}"),
        ],
        justify="start",
    )

    _slider_panel
    return (combined_fig,)


@app.cell
def _(mo):
    mo.md("""
    ## Interactive Spectrum Analyzer

    **How it works:**
    1. The **top plot** shows the timeseries (original and reconstructed via inverse FFT)
    2. Drag the **EQ sliders** to boost/cut different frequency bands
    3. The **bottom plot** shows the frequency spectrum (FFT of the timeseries)
    4. Changes propagate instantly: spectrum modifications → IFFT → timeseries

    **Try it:** Boost low frequencies to see smoother trends, or cut high frequencies to remove noise.
    """)
    return


@app.cell
def _(gain_sliders, mo):
    mo.hstack(list(gain_sliders))
    return


@app.cell
def _(combined_fig, mo):
    mo.hstack(
        [
            # _slider_panel,
            combined_fig,
        ],
        widths=[1, 4],
        gap=2,
    )
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
