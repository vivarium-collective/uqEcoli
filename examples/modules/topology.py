import marimo

__generated_with = "0.20.4"
app = marimo.App(width="full")


@app.cell
def _():
    import numpy as np
    import polars as pl
    import dataclasses as dc
    from scipy.spatial.distance import euclidean
    import plotly.graph_objects as go

    @dc.dataclass
    class Point:
        x: float
        y: float
        id: str | None = None

        def to_coordinates(self) -> np.ndarray[float, (2,)]:
            return np.array([self.x, self.y])

    @dc.dataclass
    class Pair:
        a: Point
        b: Point
        id: str = None
        m: Point = None
        d: Point = None

        def __post_init__(self):
            if self.id is None:
                self.id = f"a({self.a.to_coordinates()}), b({self.b.to_coordinates()}"
            self.m: Point = self.calculate_middle()
            self.d: float | int = self.calculate_distance()

        def calculate_middle(self):
            pos = (self.a.to_coordinates() + self.b.to_coordinates()) / 2
            return Point(*pos, id=f"midpoint-{self.id}")

        def calculate_distance(self):
            return np.linalg.norm(self.b.to_coordinates() - self.a.to_coordinates())

        def to_coordinates(self) -> np.ndarray[float, (3,)]:
            # mid.x, mid.y, self.distance
            return (self.m.x, self.m.y, self.d)

        def to_df(self):
            return pl.DataFrame(dict(zip(["x", "y", "d"], self.to_coordinates())))

        def plot(self) -> go.Figure:
            fig = go.Figure()
            a, b = list(map(lambda p: p.to_coordinates(), [self.a, self.b]))
            label_a, label_b = ["a", "b"]
            for point, label, color in zip(
                [a, b],
                [label_a, label_b],
                ["magenta", "cyan"],
            ):
                fig.add_trace(
                    go.Scatter(
                        x=[point[0]],
                        y=[point[1]],
                        mode="markers+text",
                        name=label,
                        text=[label],
                        textposition="top center",
                        textfont=dict(color=color, size=13),
                        marker=dict(color=color, size=12, symbol="circle"),
                    )
                )

            # --- midpoint from to_df() ---
            row = self.to_df().row(0, named=True)
            fig.add_trace(
                go.Scatter(
                    x=[row["x"]],
                    y=[row["y"]],
                    mode="markers+text",
                    name="midpoint",
                    text=["midpoint"],
                    textposition="top center",
                    textfont=dict(color="limegreen", size=13),
                    marker=dict(color="limegreen", size=12, symbol="diamond"),
                )
            )

            fig.update_layout(
                plot_bgcolor="#0a0a0a",
                paper_bgcolor="#0a0a0a",
                yaxis=dict(scaleanchor="x", showgrid=False, zeroline=False, visible=False),
                xaxis=dict(showgrid=False, zeroline=False, visible=False),
                margin=dict(l=20, r=20, t=20, b=20),
                showlegend=False,
            )

            return fig

    pair1 = Pair(a=Point(22, 1.111), b=Point(-0.3, 0.22**3))
    pair2 = Pair(
        a=Point(
            *np.random.random(
                2,
            )
        ),
        b=Point(
            *np.random.random(
                2,
            )
        ),
    )
    return Pair, Point, dc, go, np, pair2


@app.cell
def _(pair2):
    pair2.plot()
    return


@app.cell
def _():
    return


@app.cell
def _(Pair, Point, dc, go, np):
    def generate_loop(
        *,
        granularity: int = 200,
        radius: float = 1.0,
        roughness: float = 0.35,
        n_blobs: int = 6,
        random_seed: int = 42,
    ) -> np.ndarray:
        """
        Generates a closed 2D splat/slime shape as an array of points.

        Parameters
        ----------
        granularity : int
            Number of points in the closed loop.
        radius : float
            Base radius of the splat.
        roughness : float
            Amplitude of stochastic radial distortion (fraction of radius).
        n_blobs : int
            Number of large organic lobes in the shape.
        random_seed : int
            Reproducibility seed.

        Returns
        -------
        np.ndarray of shape (granularity, 2)
            Closed loop of (x, y) coords. First and last point are the same.
        """
        rng = np.random.default_rng(random_seed)

        # Angles for each point around the circle (closed: last = first)
        angles = np.linspace(0, 2 * np.pi, granularity, endpoint=False)

        # --- Layer 1: large organic lobes (low-frequency) ---
        lobe_phases = rng.uniform(0, 2 * np.pi, n_blobs)
        lobe_amps = rng.uniform(0.1, 0.3, n_blobs) * radius
        lobe_freqs = rng.uniform(0.8, 2.5, n_blobs)

        radial_distortion = np.zeros(granularity)
        for amp, freq, phase in zip(lobe_amps, lobe_freqs, lobe_phases):
            radial_distortion += amp * np.sin(freq * angles + phase)

        # --- Layer 2: mid-frequency bumpiness ---
        for _ in range(4):
            freq = rng.uniform(4, 9)
            amp = rng.uniform(0.02, 0.08) * radius
            phase = rng.uniform(0, 2 * np.pi)
            radial_distortion += amp * np.sin(freq * angles + phase)

        # --- Layer 3: high-frequency surface roughness (Perlin-like sum) ---
        for _ in range(6):
            freq = rng.uniform(10, 25)
            amp = rng.uniform(0.005, 0.02) * radius * roughness
            phase = rng.uniform(0, 2 * np.pi)
            radial_distortion += amp * np.sin(freq * angles + phase)

        # Final radii — clamp so shape never collapses inward
        r = np.clip(radius + radial_distortion, 0.15 * radius, None)

        # Cartesian coords
        x = r * np.cos(angles)
        y = r * np.sin(angles)

        # Close the loop
        x = np.append(x, x[0])
        y = np.append(y, y[0])

        return np.column_stack([x, y])

    PAIR_COLORS = [
        ("magenta", "rgba(255,0,255,0.4)"),
        ("cyan", "rgba(0,255,255,0.4)"),
    ]

    @dc.dataclass
    class ClosedLoop:
        granularity: int = 200
        roughness: float = float(np.random.random())
        n_blobs: int = 7
        id: str | None = None
        line: np.ndarray | None = None
        points: list[Point] | None = None

        def __post_init__(self):
            self.line = generate_loop(granularity=self.granularity)
            if self.id is None:
                l = self.line.flatten()
                self.id = f"loop-min({np.min(l)}_max({np.max(l)}"
            self.points = [Point(*p, id=f"point-{i}") for i, p in enumerate(self.line)]

        def get_pairs(self):
            from itertools import combinations

            return [Pair(a=a, b=b) for a, b in combinations(self.points, 2)]

        def get_pair_combinations(self) -> list[Pair]:
            from itertools import combinations

            pairs = self.get_pairs()
            return list(combinations(pairs, 2))

        def plot_pairs(self, pair1: Pair, pair2: Pair) -> go.Figure:
            pts = self.line
            fig = go.Figure(
                go.Scatter(
                    name=self.id,
                    x=pts[:, 0],
                    y=pts[:, 1],
                    mode="lines",
                    fill="toself",
                    fillcolor="rgba(0, 255, 100, 0.2)",
                    line=dict(color="limegreen", width=2),
                )
            )

            for pair, (color, color_rgba) in zip([pair1, pair2], PAIR_COLORS):
                # --- connector line a → b ---
                fig.add_trace(
                    go.Scatter(
                        x=[pair.a.x, pair.b.x],
                        y=[pair.a.y, pair.b.y],
                        mode="lines",
                        line=dict(color=color, width=1.5, dash="solid"),
                        showlegend=False,
                    )
                )

                # --- a, b, midpoint dots (all same color per pair) ---
                for point, label in zip(
                    [pair.a, pair.b, pair.m],
                    ["a", "b", "m"],
                ):
                    fig.add_trace(
                        go.Scatter(
                            x=[point.x],
                            y=[point.y],
                            mode="markers+text",
                            text=[label],
                            textposition="top center",
                            textfont=dict(color=color, size=12),
                            marker=dict(color=color, size=12, symbol="circle" if label != "m" else "diamond"),
                            showlegend=False,
                        )
                    )

            fig.update_layout(
                yaxis=dict(scaleanchor="x"),
                plot_bgcolor="#0a0a0a",
                paper_bgcolor="#0a0a0a",
                margin=dict(l=20, r=20, t=20, b=20),
                showlegend=False,
                xaxis=dict(showgrid=False, zeroline=False, visible=False),
                yaxis_showgrid=False,
                yaxis_zeroline=False,
                yaxis_visible=False,
            )
            return fig

        def get_pair(self, i_a: int, i_b: int):
            return Pair(
                a=self.points[i_a],
                b=self.points[i_b],
            )

        def get_random_pair_combinations(self, n: int, seed: int = 42) -> list[tuple[Pair, Pair]]:
            import random

            rng = random.Random(seed)
            pairs = self.get_pairs()
            return [(rng.choice(pairs), rng.choice(pairs)) for _ in range(n)]

        def plot_pair(self, pair: Pair = None) -> go.Figure:
            pts = self.line
            fig = go.Figure(
                go.Scatter(
                    name=self.id,
                    x=pts[:, 0],
                    y=pts[:, 1],
                    mode="lines",
                    fill="toself",
                    fillcolor="rgba(0, 255, 100, 0.2)",
                    line=dict(color="limegreen", width=2),
                )
            )

            if pair is not None:
                for trace in pair.plot().data:  # extract traces from Pair fig and superimpose
                    fig.add_trace(trace)

            fig.update_layout(
                yaxis=dict(scaleanchor="x"),
                plot_bgcolor="#0a0a0a",
                paper_bgcolor="#0a0a0a",
                margin=dict(l=20, r=20, t=20, b=20),
                showlegend=False,
                xaxis=dict(showgrid=False, zeroline=False, visible=False),
                yaxis_showgrid=False,
                yaxis_zeroline=False,
                yaxis_visible=False,
            )
            return fig

        def verify_inscribed_rectangle(self, p1: Pair, p2: Pair, tol: float = 1e-6) -> bool:
            m1, m2 = p1.m.to_coordinates(), p2.m.to_coordinates()
            d1, d2 = p1.d, p2.d
            return np.allclose(m1, m2, atol=tol) and abs(d1 - d2) < tol

        def get_verified_pair_combinations(self, tol: float = 1e-6) -> list[tuple[Pair, Pair]]:
            return [
                (p1, p2) for p1, p2 in self.get_pair_combinations() if self.verify_inscribed_rectangle(p1, p2, tol=tol)
            ]

        def plot_pairs_movie(self, pair_combinations: list[tuple[Pair, Pair]]) -> go.Figure:
            pts = self.line

            def pair_traces(pair1: Pair, pair2: Pair) -> list:
                traces = []
                for pair, (color, _) in zip([pair1, pair2], PAIR_COLORS):
                    traces.append(
                        go.Scatter(
                            x=[pair.a.x, pair.b.x],
                            y=[pair.a.y, pair.b.y],
                            mode="lines",
                            line=dict(color=color, width=1.5, dash="solid"),
                            showlegend=False,
                        )
                    )
                    for point, label in zip([pair.a, pair.b, pair.m], ["a", "b", "m"]):
                        traces.append(
                            go.Scatter(
                                x=[point.x],
                                y=[point.y],
                                mode="markers+text",
                                text=[label],
                                textposition="top center",
                                textfont=dict(color=color, size=12),
                                marker=dict(color=color, size=12, symbol="circle" if label != "m" else "diamond"),
                                showlegend=False,
                            )
                        )
                return traces

            def verified_trace(p1: Pair, p2: Pair) -> go.Scatter:
                result = self.verify_inscribed_rectangle(p1, p2)
                label = f"VERIFIED: {result}"
                color = "limegreen" if result else "red"
                x_anchor = np.min(pts[:, 0])
                y_anchor = np.max(pts[:, 1])
                return go.Scatter(
                    x=[x_anchor],
                    y=[y_anchor],
                    mode="text",
                    text=[label],
                    textposition="bottom right",
                    textfont=dict(color=color, size=15, family="monospace"),
                    showlegend=False,
                )

            loop_trace = go.Scatter(
                name=self.id,
                x=pts[:, 0],
                y=pts[:, 1],
                mode="lines",
                fill="toself",
                fillcolor="rgba(0, 255, 100, 0.2)",
                line=dict(color="limegreen", width=2),
            )

            frames = []
            for i, (p1, p2) in enumerate(pair_combinations):
                frames.append(
                    go.Frame(
                        data=[loop_trace] + pair_traces(p1, p2) + [verified_trace(p1, p2)],
                        name=str(i),
                    )
                )

            fig = go.Figure(
                data=frames[0].data,
                frames=frames,
            )

            fig.update_layout(
                yaxis=dict(scaleanchor="x"),
                plot_bgcolor="#0a0a0a",
                paper_bgcolor="#0a0a0a",
                margin=dict(l=20, r=20, t=40, b=20),
                showlegend=False,
                xaxis=dict(showgrid=False, zeroline=False, visible=False),
                yaxis_showgrid=False,
                yaxis_zeroline=False,
                yaxis_visible=False,
                updatemenus=[
                    dict(
                        type="buttons",
                        showactive=False,
                        y=1.08,
                        x=0.5,
                        xanchor="center",
                        buttons=[
                            dict(
                                label="▶  Play",
                                method="animate",
                                args=[
                                    None,
                                    dict(
                                        frame=dict(duration=400, redraw=True),
                                        fromcurrent=True,
                                        transition=dict(duration=0),
                                    ),
                                ],
                            ),
                            dict(
                                label="⏸  Pause",
                                method="animate",
                                args=[
                                    [None],
                                    dict(
                                        frame=dict(duration=0, redraw=False),
                                        mode="immediate",
                                        transition=dict(duration=0),
                                    ),
                                ],
                            ),
                        ],
                    )
                ],
                sliders=[
                    dict(
                        currentvalue=dict(prefix="frame: ", font=dict(color="limegreen")),
                        pad=dict(t=10),
                        steps=[
                            dict(
                                method="animate",
                                args=[
                                    [str(i)],
                                    dict(
                                        mode="immediate",
                                        frame=dict(duration=400, redraw=True),
                                        transition=dict(duration=0),
                                    ),
                                ],
                                label=str(i),
                            )
                            for i in range(len(frames))
                        ],
                    )
                ],
            )

            return fig

    return (ClosedLoop,)


@app.cell
def _(ClosedLoop, np):
    loop = ClosedLoop(granularity=1111, n_blobs=22, roughness=np.random.random() ** np.cos(np.random.random()))
    return (loop,)


@app.cell
def _(loop, np):
    def i_rand():
        return np.random.randint(low=0, high=len(loop.points) - 1)

    pair_a, pair_b = list(map(lambda pair: loop.get_pair(i_a=i_rand(), i_b=i_rand()), list(range(2))))
    return


@app.cell
def _():
    # combos = loop.get_random_pair_combinations(100)
    return


@app.cell
def _():
    # loop.plot_pairs_movie(combos)
    return


@app.cell
def _(loop):
    for tol in [1e-6, 1e-4, 1e-2, 0.05, 0.1]:
        results = loop.get_verified_pair_combinations(tol=tol)
        print(f"tol={tol:.0e}  →  {len(results)} verified pairs")
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
