# /// script
# requires-python = ">=3.12"
# dependencies = ["marimo>=0.23", "plotly", "numpy"]
# ///

import marimo

__generated_with = "0.23.0"
app = marimo.App(width="full")


# ════════════════════════════════════════════════════════════════════════════
#  IMPORTS
# ════════════════════════════════════════════════════════════════════════════


@app.cell
def _imports():
    import marimo as mo
    import numpy as np
    import json
    from pathlib import Path
    import plotly.graph_objects as go

    return Path, go, json, mo, np


# ════════════════════════════════════════════════════════════════════════════
#  THEME — DAW dark palette injected as global CSS
# ════════════════════════════════════════════════════════════════════════════


@app.cell
def _theme(mo):
    mo.Html("""
<style>
  :root {
    --bg:         #0d0d0d;
    --surface:    #1a1a2e;
    --surface2:   #252545;
    --border:     #2a2a4a;
    --text:       #e0e0e0;
    --text-dim:   #888888;
    --accent:     #00f0ff;
    --accent2:    #aa66ff;
    --accent3:    #33ff99;
    --accent4:    #ffaa00;
    --accent5:    #ff3366;
    --font:       'Inter', 'Helvetica Neue', sans-serif;
    --mono:       'Menlo', 'Fira Code', monospace;
  }

  .uq-achievement {
    background: var(--surface);
    border-left: 4px solid var(--accent);
    padding: 12px 16px;
    margin: 10px 0;
    border-radius: 4px;
    font-family: var(--font);
  }
  .uq-achievement .label {
    color: var(--accent);
    font-weight: bold;
    font-size: 0.85em;
    letter-spacing: 0.08em;
    text-transform: uppercase;
  }
  .uq-achievement .body {
    color: var(--text);
    margin-top: 4px;
    font-size: 0.95em;
  }

  .uq-boss {
    background: linear-gradient(135deg, #1a0a2e, #0a1a2e);
    border: 2px solid var(--accent2);
    padding: 16px;
    border-radius: 8px;
    margin: 12px 0;
    font-family: var(--font);
  }

  .uq-cmd {
    background: #0a0a14;
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: 12px 16px;
    font-family: var(--mono);
    font-size: 0.9em;
    color: var(--accent3);
    margin: 8px 0;
    white-space: pre;
    overflow-x: auto;
  }

  .uq-badge {
    display: inline-block;
    background: var(--surface2);
    border: 1px solid var(--accent);
    color: var(--accent);
    border-radius: 12px;
    padding: 2px 10px;
    font-size: 0.8em;
    font-family: var(--mono);
    margin: 2px;
  }

  .uq-tip {
    background: var(--surface);
    border-left: 4px solid var(--accent4);
    padding: 10px 14px;
    margin: 8px 0;
    border-radius: 4px;
    color: var(--text-dim);
    font-size: 0.9em;
    font-family: var(--font);
  }
  .uq-tip .tip-label { color: var(--accent4); font-weight: bold; }

  .uq-warn {
    background: var(--surface);
    border-left: 4px solid var(--accent5);
    padding: 10px 14px;
    margin: 8px 0;
    border-radius: 4px;
    color: var(--text-dim);
    font-size: 0.9em;
    font-family: var(--font);
  }

  .uq-skill-tree {
    display: grid;
    grid-template-columns: repeat(5, 1fr);
    gap: 8px;
    margin: 12px 0;
  }
  .uq-skill {
    background: var(--surface2);
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: 8px;
    text-align: center;
    font-size: 0.8em;
    font-family: var(--font);
    color: var(--text-dim);
    transition: all 0.2s;
  }
  .uq-skill.unlocked {
    border-color: var(--accent);
    color: var(--accent);
    background: var(--surface);
  }
</style>
""")
    return


# ════════════════════════════════════════════════════════════════════════════
#  MISSION START
# ════════════════════════════════════════════════════════════════════════════


@app.cell
def _title(mo):
    mo.Html("""
<div style="
  background: linear-gradient(135deg, #0a0a14 0%, #1a0a2e 50%, #0a1a2e 100%);
  border: 1px solid #2a2a4a;
  border-radius: 12px;
  padding: 32px 40px;
  text-align: center;
  font-family: 'Inter', sans-serif;
  margin-bottom: 8px;
">
  <div style="font-size: 0.75em; letter-spacing: 0.3em; color: #00f0ff; text-transform: uppercase; margin-bottom: 8px;">
    ▶ MISSION START
  </div>
  <div style="font-size: 2.2em; font-weight: 900; color: #e0e0e0; letter-spacing: -0.02em; margin-bottom: 4px;">
    UQ Framework
  </div>
  <div style="font-size: 1.1em; color: #aa66ff; font-weight: 600; margin-bottom: 16px;">
    Uncertainty Quantification for vEcoli — Interactive Tutorial
  </div>
  <div style="font-size: 0.9em; color: #888888; max-width: 600px; margin: 0 auto; line-height: 1.6;">
    Every concept is computed live using the real UQ pipeline.<br/>
    Complete all 10 levels to master the full <code style="color: #00f0ff; background: #1a1a2e; padding: 1px 6px; border-radius: 3px;">uq</code> CLI.
  </div>
</div>
""")
    return


# ════════════════════════════════════════════════════════════════════════════
#  LEVEL SELECTOR
# ════════════════════════════════════════════════════════════════════════════


@app.cell
def _level_slider(mo):
    level = mo.ui.slider(
        start=0,
        stop=9,
        step=1,
        value=0,
        label="Progress Level",
        show_value=True,
        full_width=True,
    )
    return (level,)


@app.cell
def _level_display(level, mo):
    _levels = [
        ("🎯", "The Mission",         "Why UQ — the core problem"),
        ("📐", "Parameter Space",     "What we perturb and how"),
        ("🎲", "Sampling",            "PCRV germ sampling + uq sample"),
        ("📦", "PCE Surrogate",       "Fitting the polynomial chaos model"),
        ("📊", "Sobol Indices",       "Analytical sensitivity from PCE"),
        ("🔬", "4 Strategies",        "Aggregation deep-dive (RFC006)"),
        ("🚀", "Full Pipeline",       "sample → quantify — live command builder"),
        ("⚡", "Advanced Features",   "PCA, baseline, remote, multi-condition"),
        ("⚔️", "CLI Arsenal",         "Every uq command, with examples"),
        ("👾", "Boss Level",          "Interactive PCE challenge — prove mastery"),
    ]
    _lv = int(level.value)
    _emoji, _title, _desc = _levels[_lv]

    _bars = "".join(
        f'<span style="display:inline-block;width:28px;height:8px;border-radius:4px;'
        f'background:{"#00f0ff" if i <= _lv else "#2a2a4a"};margin:0 2px;"></span>'
        for i in range(10)
    )
    mo.Html(f"""
<div style="
  background: #1a1a2e;
  border: 1px solid #2a2a4a;
  border-radius: 8px;
  padding: 14px 20px;
  display: flex;
  align-items: center;
  gap: 16px;
  font-family: 'Inter', sans-serif;
  margin: 4px 0;
">
  <div style="font-size: 2em;">{_emoji}</div>
  <div style="flex: 1;">
    <div style="font-size: 0.7em; letter-spacing: 0.15em; color: #888; text-transform: uppercase;">
      Level {_lv + 1} of 10
    </div>
    <div style="font-size: 1.1em; font-weight: 700; color: #e0e0e0;">{_title}</div>
    <div style="font-size: 0.85em; color: #888888;">{_desc}</div>
  </div>
  <div style="text-align: right;">
    <div style="margin-bottom: 4px;">{_bars}</div>
    <div style="font-size: 0.7em; color: #00f0ff; letter-spacing: 0.1em;">{_lv * 10 + 10}% COMPLETE</div>
  </div>
</div>
""")
    return


# ════════════════════════════════════════════════════════════════════════════
#  LEVEL 0 — THE MISSION
# ════════════════════════════════════════════════════════════════════════════


@app.cell
def _l0_chart(go, np):
    _rng = np.random.default_rng(42)
    _t = np.linspace(0, 10, 200)
    _k_vals = np.linspace(0.3, 0.7, 16)

    _fig0 = go.Figure()
    for _i, _k in enumerate(_k_vals):
        _y = np.exp(_k * _t) * (1 + 0.04 * np.sin(2 * _t))
        _fig0.add_trace(go.Scatter(
            x=_t, y=_y, mode="lines",
            line=dict(color="#00f0ff" if _i == 8 else "#aa66ff", width=2 if _i == 8 else 1),
            opacity=0.7 if _i == 8 else 0.18,
            showlegend=_i == 8,
            name="k = 0.50 (nominal)" if _i == 8 else None,
        ))
    _fig0.update_layout(
        title=dict(text="Output Uncertainty from Parameter Spread", x=0.5,
                   font=dict(color="#e0e0e0", size=14)),
        xaxis_title="Time (hours)", yaxis_title="Dry Mass (pg)",
        template="plotly_dark",
        height=320,
        paper_bgcolor="#0d0d0d", plot_bgcolor="#0d0d0d",
        legend=dict(x=0.01, y=0.99, bgcolor="rgba(0,0,0,0)", font=dict(color="#e0e0e0")),
        margin=dict(l=50, r=20, t=50, b=40),
    )
    return (_fig0,)


@app.cell
def _l0_content(_fig0, mo):
    mo.vstack([
        _fig0,
        mo.md(r"""
### The Core Problem

Every vEcoli simulation depends on **kinetic parameters** — rate constants, mass
fractions, transcription efficiencies. These come from experiments with
measurement uncertainty. The chart above shows what happens when a single
growth rate constant spans its plausible range: a **fan of possible futures**.

**The UQ framework answers four questions:**

| Question | Tool |
|---|---|
| Which parameters actually drive output variance? | **Sobol sensitivity indices** |
| How much variance does each explain? | **Polynomial Chaos Expansion (PCE)** |
| Does sensitivity shift across the cell cycle? | **Growth-stratified GSA (Strategy 4)** |
| Is the finding robust to stochastic noise? | **Multi-seed aggregation (Strategy 3)** |

**Two commands. That's the whole pipeline:**

```bash
uv run uq sample  /path/to/simData.cPickle   # perturb + run vEcoli + cache
uv run uq quantify /path/to/simData.cPickle  # fit PCE + Sobol → report.html
```

> Advance the **Progress Level** slider to unlock each concept.
"""),
    ])
    return


# ════════════════════════════════════════════════════════════════════════════
#  LEVEL 1 — PARAMETER SPACE
# ════════════════════════════════════════════════════════════════════════════


@app.cell
def _l1_gate(level, mo):
    mo.stop(int(level.value) < 1)
    return


@app.cell
def _l1_data():
    default_params = [
        ("kinetic_obj_weight",   5e-8,  5e-7,  "Metabolism FBA objective blending"),
        ("secretion_penalty",    1e-4,  1e-2,  "Overflow metabolism penalty"),
        ("rnap_free_frac",       0.25,  0.47,  "ppGpp-free RNAP active fraction"),
        ("rnap_bound_frac",      0.25,  0.47,  "ppGpp-bound RNAP active fraction"),
        ("basal_elong_rate",     10.0,  22.0,  "Ribosome elongation speed (aa/s)"),
        ("dry_mass_frac",        0.25,  0.35,  "Cell dry mass fraction"),
    ]
    return (default_params,)


@app.cell
def _l1_chart(default_params, go, np):
    _names = [p[0] for p in default_params]
    _lows  = [p[1] for p in default_params]
    _highs = [p[2] for p in default_params]
    _ranges = [(h - l) / ((h + l) / 2) * 100 for l, h in zip(_lows, _highs)]

    _fig1 = go.Figure(go.Bar(
        y=_names, x=_ranges, orientation="h",
        marker=dict(color=np.linspace(0.3, 0.9, 6), colorscale="Tealgrn", showscale=False),
        text=[f"±{r:.0f}%" for r in _ranges], textposition="outside",
        textfont=dict(color="#00f0ff", size=12),
    ))
    _fig1.update_layout(
        title=dict(text="Default 6 Parameters — Relative Uncertainty Range", x=0.5,
                   font=dict(color="#e0e0e0", size=13)),
        xaxis_title="Relative Range (%)",
        template="plotly_dark", height=240,
        paper_bgcolor="#0d0d0d", plot_bgcolor="#0d0d0d",
        margin=dict(l=160, r=80, t=50, b=40),
        yaxis=dict(autorange="reversed"),
    )
    return (_fig1,)


@app.cell
def _l1_content(default_params, _fig1, mo):
    _rows = "\n".join(
        f"| `{n}` | {l:.2g} | {h:.2g} | {(h-l)/((h+l)/2)*100:.0f}% | {d} |"
        for n, l, h, d in default_params
    )
    mo.vstack([
        _fig1,
        mo.md(f"""
### Default 6 Parameters (`DEFAULT_SIM_DATA_PARAMETERS`)

Each parameter is a **dot-path** into `SimulationDataEcoli`:

| Parameter | Lower | Upper | Range | Description |
|---|---|---|---|---|
{_rows}

**How they map to vEcoli internals:**

```
process.metabolism.kinetic_objective_weight     →  FBA kinetic vs homeostatic blend
process.transcription.fraction_active_rnap_free →  RNAP activation (ppGpp-free)
process.translation.basal_elongation_rate       →  ribosome speed
mass.cell_dry_mass_fraction                     →  mass conversion factor
```

During sampling each LHS point becomes a **vEcoli variant**:
```json
{{"value": [{{"process.metabolism.kinetic_objective_weight": 3.2e-7}}]}}
```
`sim_data_setattr` mutates `simData.cPickle` before each run.

**Custom parameters:** `--params-file params.json` overrides the defaults.
```json
[
  {{"name": "my_param", "attr_path": "process.metabolism.kinetic_objective_weight",
    "bounds": [1e-8, 1e-6], "description": "FBA objective weight"}}
]
```
"""),
        mo.Html("""
<div class="uq-achievement">
  <div class="label">⚡ Skill Unlocked — Parameter Space</div>
  <div class="body">You can now define what the UQ framework perturbs. Six physiologically-motivated
  defaults are built in; override with any scalar <code>simData</code> attribute.</div>
</div>
"""),
    ])
    return


# ════════════════════════════════════════════════════════════════════════════
#  LEVEL 2 — SAMPLING
# ════════════════════════════════════════════════════════════════════════════


@app.cell
def _l2_gate(level, mo):
    mo.stop(int(level.value) < 2)
    return


@app.cell
def _l2_sliders(mo):
    n_samples_slider = mo.ui.slider(10, 200, step=10, value=50, label="N samples", show_value=True, full_width=True)
    n_params_slider  = mo.ui.slider(2,  6,   step=1,  value=2,  label="d parameters", show_value=True, full_width=True)
    return n_params_slider, n_samples_slider


@app.cell
def _l2_lhs(go, n_params_slider, n_samples_slider, np):
    from math import comb as _comb

    _N = int(n_samples_slider.value)
    _d = int(n_params_slider.value)
    _p = 3

    _rng2 = np.random.RandomState(42)
    _lhs = np.zeros((_N, _d))
    for _dim in range(_d):
        _perm = _rng2.permutation(_N)
        _lhs[:, _dim] = (_perm + _rng2.random(_N)) / _N

    _n_terms = _comb(_d + _p, _p)

    _fig2 = go.Figure()
    if _d == 2:
        _fig2.add_trace(go.Scatter(
            x=_lhs[:, 0], y=_lhs[:, 1], mode="markers",
            marker=dict(size=8, color="#00f0ff", opacity=0.7),
            name=f"LHS (N={_N})",
        ))
        _fig2.update_layout(
            xaxis=dict(title="Parameter 1 [0,1]", range=[0, 1], gridcolor="#1a1a2e"),
            yaxis=dict(title="Parameter 2 [0,1]", range=[0, 1], gridcolor="#1a1a2e"),
        )
    else:
        _fig2.add_trace(go.Scatter3d(
            x=_lhs[:, 0], y=_lhs[:, 1], z=_lhs[:, 2],
            mode="markers",
            marker=dict(size=5, color=_lhs[:, 0], colorscale="Tealgrn", opacity=0.8, showscale=False),
        ))
        _fig2.update_layout(scene=dict(
            xaxis_title="p1", yaxis_title="p2", zaxis_title="p3",
            bgcolor="#0d0d0d",
        ))

    _fig2.update_layout(
        title=dict(text=f"PCRV Germ Samples — N={_N}, d={_d}", x=0.5,
                   font=dict(color="#e0e0e0", size=13)),
        template="plotly_dark", height=360,
        paper_bgcolor="#0d0d0d", plot_bgcolor="#0d0d0d",
        margin=dict(l=30, r=30, t=50, b=40),
    )
    return _fig2, _n_terms


@app.cell
def _l2_content(_fig2, _n_terms, mo, n_params_slider, n_samples_slider):
    _N2 = int(n_samples_slider.value)
    _d2 = int(n_params_slider.value)
    mo.vstack([
        _fig2,
        mo.md(f"""
### PCRV Sampling — Step 2 of UQPC

**PyTUQ PCRV** (Polynomial Chaos Random Variable) generates samples in *germ space* `ξ ∈ [-1, 1]ᵈ`:

```python
from pytuq.rv.pcrv import PCRV
input_pc = PCRV(dom=bounds, pctype="LU", order=1)  # Legendre uniform
germ = input_pc.sampleGerm({_N2})     # ← NOT LHS — PyTUQ native random sampling
X    = input_pc.evalPC(germ)          # map to physical space
```

**Physical space:** `x(ξ) = ½(lb + ub) + ½(ub - lb) · ξ`

| Setting | Value |
|---|---|
| Parameters (d) | **{_d2}** |
| Samples (N) | **{_N2}** |
| PCE basis terms (p=3) | **{_n_terms}** |
| Recommended minimum | **{2 * _n_terms}** (= 2×basis) |

```bash
uv run uq sample /path/to/simData.cPickle \\
    --cache-dir ./uq_cache \\
    --n-samples {_N2} \\
    --observables higher_order \\
    --generations 2 \\
    --n-init-sims 2
```

**What `uq sample` does internally:**
1. Load `simData.cPickle` → build `XSpaceVecoli` (parameter space)
2. `PCRV.sampleGerm(N)` → germ samples
3. Build sim_data_setattr variants → vEcoli config JSON
4. Run `runscripts/workflow.py` (subprocess, Nextflow)
5. Collect Parquet → extract observables (cd1 analysis)
6. Cache `(X, Y, timeseries)` → `./uq_cache/`
"""),
        mo.Html("""
<div class="uq-tip">
  <span class="tip-label">SAMPLING TIP:</span>
  Use <code>--n-test 10</code> to reserve held-out samples for out-of-sample PCE
  validation (UQPC --ntst). These are run through vEcoli alongside training samples
  and cached as <code>X_test.npy / Y_test.npy</code>.
</div>
"""),
    ])
    return


# ════════════════════════════════════════════════════════════════════════════
#  LEVEL 3 — PCE SURROGATE
# ════════════════════════════════════════════════════════════════════════════


@app.cell
def _l3_gate(level, mo):
    mo.stop(int(level.value) < 3)
    return


@app.cell
def _l3_pce(go, np):
    from libuq.inputs import XSpaceVecoli as _XS3
    from libuq.pipeline.models import SimDataParameter as _P3
    from uq.workflow import run_uqpc as _run3

    _rng3 = np.random.RandomState(42)
    _X3   = _rng3.uniform(0, 1, (80, 3))
    _Y3   = (3*_X3[:,0] + 1.5*_X3[:,1] + 0.3*_X3[:,2]
             + 0.5*_X3[:,0]*_X3[:,1] + 0.2*_X3[:,0]**2).reshape(-1, 1)

    _ps3 = _XS3(
        parameter_names=[], parameter_bounds=[], parameter_types=[],
        experiment_id="tutorial3",
        parameters=[_P3(name=f"p{i}", attr_path=f"x.p{i}", bounds=(0.0, 1.0)) for i in range(3)],
    )
    _res3 = _run3(param_space=_ps3, Y_train=_Y3, X_train=_X3,
                  polynomial_order=2, regression="lsq", seed=42)

    _sweep3 = np.linspace(0, 1, 100)
    _Xsw3   = np.tile(np.array([[0.5, 0.5, 0.5]]), (100, 1))
    _Xsw3[:, 0] = _sweep3
    _Ysw3 = _res3.surrogate.predict(_Xsw3)

    _fig3 = go.Figure()
    _fig3.add_trace(go.Scatter(
        x=_X3[:, 0], y=_Y3.flatten(), mode="markers",
        marker=dict(size=6, color="#aa66ff", opacity=0.5), name="Training samples",
    ))
    _fig3.add_trace(go.Scatter(
        x=_sweep3, y=_Ysw3.flatten(), mode="lines",
        line=dict(color="#00f0ff", width=3), name="PCE fit (p=2)",
    ))
    _fig3.update_layout(
        title=dict(text="PCE Surrogate — Fitted Response (sweep p₁, fix p₂=p₃=0.5)", x=0.5,
                   font=dict(color="#e0e0e0", size=13)),
        xaxis_title="p₁", yaxis_title="Output Ŷ",
        template="plotly_dark", height=330,
        paper_bgcolor="#0d0d0d", plot_bgcolor="#0d0d0d",
        margin=dict(l=50, r=20, t=50, b=40),
        legend=dict(x=0.01, y=0.99, bgcolor="rgba(0,0,0,0)"),
    )

    _relerr3 = float(np.mean(_res3.relerr_train)) if _res3.relerr_train is not None else 0.0
    _n_terms3 = len(_res3.surrogate.multi_indices)

    return _fig3, _n_terms3, _relerr3


@app.cell
def _l3_content(_fig3, _n_terms3, _relerr3, mo):
    mo.vstack([
        _fig3,
        mo.md(rf"""
### PCE Surrogate — Step 4 of UQPC

A **Legendre Polynomial Chaos Expansion** is fitted to the cached (X, Y) data:

$$\hat{{Y}}(\xi) = \sum_{{|\alpha| \le p}} c_\alpha \, \Phi_\alpha(\xi), \quad \xi \in [-1, 1]^d$$

- `Φ_α` = multivariate Legendre polynomial (orthogonal on [-1, 1]ᵈ)
- `c_α` = coefficients fitted by **least squares** (or BCS, ANL)
- `α` = multi-index, e.g. `[1, 0, 0]` → linear in p₁

**Live result:** {_n_terms3} basis terms, mean train error = `{_relerr3:.2e}`

**Regression methods** (`--regression`):

| Method | Flag | When to use |
|---|---|---|
| Least squares | `lsq` | Default, N ≥ 2×terms |
| Bayesian Compressed Sensing | `bcs` | Sparse, high-d, fewer samples |
| Analytical Bayes | `anl` | Fast, assumes Gaussian prior |

**Why PCE?** 80 samples → analytical Sobol indices. Monte Carlo would need 10,000+
samples for the same accuracy.

**Polynomial order** (`--polynomial-order`): order 2 = quadratic interactions.
Higher order → more terms → more samples needed.
"""),
        mo.Html(f"""
<div class="uq-achievement">
  <div class="label">⚡ Skill Unlocked — PCE Surrogate</div>
  <div class="body">The PCE is the core math object. Once fitted, Sobol indices are
  computed <em>analytically</em> from the coefficients — no additional simulation runs needed.
  Train error: <code>{_relerr3:.2e}</code>  |  Basis terms: <code>{_n_terms3}</code></div>
</div>
"""),
    ])
    return


# ════════════════════════════════════════════════════════════════════════════
#  LEVEL 4 — SOBOL INDICES
# ════════════════════════════════════════════════════════════════════════════


@app.cell
def _l4_gate(level, mo):
    mo.stop(int(level.value) < 4)
    return


@app.cell
def _l4_sobol(go, np):
    from libuq.inputs import XSpaceVecoli as _XS4
    from libuq.pipeline.models import SimDataParameter as _P4
    from uq.workflow import run_uqpc as _run4

    _rng4 = np.random.RandomState(123)
    _X4   = _rng4.uniform(0, 1, (100, 3))
    # p1 dominates, p2 moderate, p3 negligible
    _Y4   = (5*_X4[:,0] + 2*_X4[:,1] + 0.1*_X4[:,2]
             + 1.5*_X4[:,0]*_X4[:,1]).reshape(-1, 1)

    _ps4 = _XS4(
        parameter_names=[], parameter_bounds=[], parameter_types=[],
        experiment_id="tutorial4",
        parameters=[_P4(name=f"p{i}", attr_path=f"x.p{i}", bounds=(0.0, 1.0)) for i in range(3)],
    )
    _res4 = _run4(param_space=_ps4, Y_train=_Y4, X_train=_X4,
                  polynomial_order=3, regression="lsq", seed=42)

    _s1_4 = _res4.sobol.first_order
    _st_4 = _res4.sobol.total_order
    _names4 = ["p₁ (dominant)", "p₂ (moderate)", "p₃ (negligible)"]

    _fig4a = go.Figure()
    _x4 = np.arange(3)
    _w4 = 0.35
    _fig4a.add_trace(go.Bar(x=_x4 - _w4/2, y=_s1_4, width=_w4,
        name="Sᵢ (first-order)", marker_color="#aa66ff"))
    _fig4a.add_trace(go.Bar(x=_x4 + _w4/2, y=_st_4, width=_w4,
        name="S_Ti (total-order)", marker_color="#00f0ff"))
    _fig4a.update_layout(
        barmode="group",
        title=dict(text="Sobol Indices — Analytical from PCE Coefficients", x=0.5,
                   font=dict(color="#e0e0e0", size=13)),
        xaxis=dict(ticktext=_names4, tickvals=[0, 1, 2], title="Parameter"),
        yaxis=dict(title="Variance Fraction", range=[0, max(_st_4.max(), _s1_4.max()) * 1.25]),
        template="plotly_dark", height=320,
        paper_bgcolor="#0d0d0d", plot_bgcolor="#0d0d0d",
        margin=dict(l=50, r=20, t=50, b=80),
        legend=dict(x=0.01, y=0.99, bgcolor="rgba(0,0,0,0)"),
    )

    # Interaction bar: S_Ti - S_i
    _interact = _st_4 - _s1_4
    _fig4b = go.Figure(go.Bar(
        x=_names4, y=_interact,
        marker_color=["#ffaa00", "#ffaa00", "#ffaa00"],
        text=[f"{v:.3f}" for v in _interact], textposition="outside",
    ))
    _fig4b.update_layout(
        title=dict(text="Interaction Effects: S_Ti − Sᵢ", x=0.5,
                   font=dict(color="#e0e0e0", size=12)),
        yaxis=dict(title="Interaction Fraction"),
        template="plotly_dark", height=220,
        paper_bgcolor="#0d0d0d", plot_bgcolor="#0d0d0d",
        margin=dict(l=50, r=20, t=45, b=60),
    )

    return _fig4a, _fig4b, _s1_4, _st_4


@app.cell
def _l4_content(_fig4a, _fig4b, _s1_4, _st_4, mo):
    mo.vstack([
        mo.hstack([_fig4a, _fig4b]),
        mo.md(rf"""
### Sobol Indices — Step 5 of UQPC

```
S_i   = Var[E[Y | x_i]]  /  Var[Y]
S_Ti  = 1 - Var[E[Y | x_-i]]  /  Var[Y]
```

| Index | Meaning | Value (this example) |
|---|---|---|
| **Sᵢ** (first-order) | Main effect of parameter i alone | p₁={_s1_4[0]:.3f}, p₂={_s1_4[1]:.3f}, p₃={_s1_4[2]:.3f} |
| **S_Ti** (total-order) | Main + ALL interactions involving i | p₁={_st_4[0]:.3f}, p₂={_st_4[1]:.3f}, p₃={_st_4[2]:.3f} |
| **S_Ti − Sᵢ** | Higher-order interaction contribution | shown above |

**In `uq_results.json`:**
```json
{{"phase1_population": {{
  "sobol_first_order":  {{"p0": {_s1_4[0]:.4f}, "p1": {_s1_4[1]:.4f}, ...}},
  "sobol_total_order":  {{"p0": {_st_4[0]:.4f}, "p1": {_st_4[1]:.4f}, ...}}
}}}}
```

**Interpretation in vEcoli context:**
- High S_Ti → measure this parameter more precisely
- Low S_Ti (< 0.02) → candidate for fixing at nominal value
- S_Ti >> Sᵢ → strong nonlinear coupling with other parameters
"""),
    ])
    return


# ════════════════════════════════════════════════════════════════════════════
#  LEVEL 5 — 4 AGGREGATION STRATEGIES
# ════════════════════════════════════════════════════════════════════════════


@app.cell
def _l5_gate(level, mo):
    mo.stop(int(level.value) < 5)
    return


@app.cell
def _l5_selector(mo):
    strategy_selector = mo.ui.dropdown(
        options={
            "Strategy 1: Uniform (bulk population)": 1,
            "Strategy 2: By Generation": 2,
            "Strategy 3: By Lineage Seed": 3,
            "Strategy 4: Growth-Stratified (θ)": 4,
        },
        value=1, label="Aggregation Strategy",
    )
    return (strategy_selector,)


@app.cell
def _l5_content(go, mo, np, strategy_selector):
    _strat_colors = {1: "#00f0ff", 2: "#aa66ff", 3: "#ffaa00", 4: "#33ff99"}
    _s_id = int(strategy_selector.value)
    _clr  = _strat_colors[_s_id]

    # Illustrative timeseries for selected strategy
    _rng5 = np.random.RandomState(_s_id * 7)
    _t5   = np.linspace(0, 6, 120)
    _fig5 = go.Figure()

    if _s_id == 1:
        for _j in range(8):
            _y5 = np.cumsum(_rng5.randn(120) * 0.15) + 2 + _j * 0.08
            _fig5.add_trace(go.Scatter(x=_t5, y=_y5, mode="lines",
                line=dict(color=_clr, width=1), opacity=0.35, showlegend=False))
        _fig5.add_trace(go.Scatter(x=_t5, y=np.ones(120) * 2.35, mode="lines",
            line=dict(color=_clr, width=3, dash="dot"), name="Uniform mean"))
    elif _s_id == 2:
        for _gen in range(3):
            _y5 = np.cumsum(_rng5.randn(120) * 0.1) + 1.8 + _gen * 0.4
            _fig5.add_trace(go.Scatter(x=_t5, y=_y5, mode="lines",
                line=dict(color=_clr, width=2), name=f"Generation {_gen+1}"))
    elif _s_id == 3:
        for _seed in range(4):
            _y5 = np.cumsum(_rng5.randn(120) * 0.2) + 2.0 + _seed * 0.1
            _fig5.add_trace(go.Scatter(x=_t5, y=_y5, mode="lines",
                line=dict(color=_clr, width=2), name=f"Seed {1000 + _seed * 100}"))
    else:
        _theta = np.linspace(0, 1, 120)
        for _stage in range(5):
            _lo, _hi = _stage / 5, (_stage + 1) / 5
            _mask = (_theta >= _lo) & (_theta < _hi)
            _y5 = 0.5 * _theta + 0.1 * np.sin(8 * np.pi * _theta) + _rng5.randn(120) * 0.05
            _fig5.add_trace(go.Scatter(x=_t5[_mask], y=_y5[_mask], mode="lines",
                line=dict(color=_clr, width=3), name=f"θ {int(_lo*100)}–{int(_hi*100)}%"))
        _fig5.update_xaxes(title_text="θ (cell cycle position, 0=birth → 1=division)")

    _fig5.update_layout(
        title=dict(text=f"Strategy {_s_id} — Aggregation Illustration", x=0.5,
                   font=dict(color="#e0e0e0", size=13)),
        xaxis_title="Time" if _s_id != 4 else "θ",
        yaxis_title="Observable", template="plotly_dark", height=280,
        paper_bgcolor="#0d0d0d", plot_bgcolor="#0d0d0d",
        margin=dict(l=50, r=20, t=50, b=40),
        legend=dict(x=0.01, y=0.99, bgcolor="rgba(0,0,0,0)"),
    )

    _descs = {
        1: """
**Uniform (bulk population)** — average over all cells, all times, all seeds.

Answers: *Which parameters matter for the population as a whole?*

This is the Strategy 1 result (`population_surrogate/`, `population_sobol/`).
No special CLI flags needed — it always runs.
""",
        2: """
**By Generation** — compute Sobol per cell generation.

Controls for transient startup dynamics. Early generations (gen 0-1) often
show different sensitivity patterns as the model converges to steady-state.

**Enable:** `uq sample --generations 2` (or more)

Output: `strategy2_by_generation/gen_0_sobol/`, `gen_1_sobol/`, ...
""",
        3: """
**By Lineage Seed** — compute Sobol per stochastic seed.

Controls for gene expression noise and stochastic molecule partitioning.
If Sobol indices vary significantly across seeds, stochastic effects dominate.

**Enable:** `uq sample --n-init-sims 2` (or more)

Output: `strategy3_by_seed/seed_0_sobol/`, `seed_1000_sobol/`, ...
""",
        4: """
**Growth-Stratified (θ)** — bin cells by cell cycle position.

Cell cycle coordinate:
`θ = [log(m) − log(m_birth)] / [log(m_div) − log(m_birth)]`

Each θ-bin gets its own PCE surrogate + Sobol indices. Answers:
*When during the cell cycle is each parameter most influential?*

**Enable:** `uq quantify --n-bins 10` (default)

Output: `strategy4_growth_stratified/stage_0/`, ..., `stage_9/`
""",
    }

    mo.vstack([
        _fig5,
        mo.md(f"""
### Strategy {_s_id}

{_descs[_s_id]}

---

| Strategy | Groups By | Purpose | CLI Requirement |
|---|---|---|---|
| 1 | None | Population average | None (always runs) |
| 2 | Generation | Convergence check | `--generations >= 2` |
| 3 | Lineage seed | Stochastic variance | `--n-init-sims >= 2` |
| 4 | θ-bin (cell cycle) | Phase-specific GSA | `--n-bins N` (default 10) |

**All 4 strategies run automatically in a single `uq quantify` call.**
The variance decomposition step reports how much total variance comes from
each source *before* any PCE is fit.
"""),
    ])
    return


# ════════════════════════════════════════════════════════════════════════════
#  LEVEL 6 — FULL PIPELINE
# ════════════════════════════════════════════════════════════════════════════


@app.cell
def _l6_gate(level, mo):
    mo.stop(int(level.value) < 6)
    return


@app.cell
def _l6_diagram(mo):
    mo.Html("""
<div style="font-family: 'Menlo', monospace; font-size: 0.82em; color: #e0e0e0;
     background: #0d0d0d; border: 1px solid #2a2a4a; border-radius: 8px; padding: 20px; margin: 8px 0;">
<pre style="margin:0;color:#e0e0e0;">
┌─────────────────────────────────────────────────────────────────────┐
│  <span style="color:#00f0ff;">uq sample</span>  (compute-intensive — runs vEcoli, hours)               │
│                                                                     │
│  Step 1: Load simData.cPickle  →  XSpaceVecoli (parameter space)   │
│  Step 2: PCRV.sampleGerm(N)   →  germ samples ξ ∈ [-1,1]ᵈ         │
│  Step 3: evalPC(ξ)            →  physical samples X                │
│  Step 4: Build sim_data_setattr variants (one per LHS point)       │
│  Step 5: Run runscripts/workflow.py  (Nextflow + vEcoli subproc)   │
│  Step 6: Collect Parquet → extract observables (cd1 analysis)      │
│  Step 7: Cache  X.npy, Y.npy, timeseries/  →  ./uq_cache/         │
└─────────────────────────────────────────┬───────────────────────────┘
                                           │  hours later
                                           ▼
┌─────────────────────────────────────────────────────────────────────┐
│  <span style="color:#33ff99;">uq quantify</span>  (fast — PCE math only, seconds)                      │
│                                                                     │
│  Step 1: Load ./uq_cache  →  X (n×d),  Y (n×m)                     │
│  Step 2: Optional PCA: Y (n×m) → Y_pca (n×K)                       │
│  Step 3: For each of 4 strategies:                                  │
│           a. Aggregate Y by strategy (uniform / gen / seed / θ)    │
│           b. Scale X to germ space [-1, 1]                          │
│           c. Build Legendre basis matrix  Φ (N × n_terms)          │
│           d. Fit PCE coefficients c via lsq / bcs / anl            │
│           e. setCfs(c) → pcrv.computeSens() → S_i, S_Ti            │
│  Step 4: Export uq_results.json + surrogate binaries + report.html │
└─────────────────────────────────────────────────────────────────────┘
</pre>
</div>
""")
    return


@app.cell
def _l6_sliders(mo):
    sample_n    = mo.ui.slider(10,  200, step=10,  value=50,  label="--n-samples",       show_value=True)
    sample_gens = mo.ui.slider(1,   5,   step=1,   value=2,   label="--generations",     show_value=True)
    sample_seed = mo.ui.slider(1,   5,   step=1,   value=2,   label="--n-init-sims",     show_value=True)
    sample_obs  = mo.ui.dropdown(
        options=["mass", "higher_order", "exchange_fluxes", "transcriptome", "proteome", "fluxome"],
        value="higher_order", label="--observables",
    )
    q_order = mo.ui.slider(1, 5, step=1, value=2, label="--polynomial-order", show_value=True)
    q_reg   = mo.ui.dropdown(
        options={"lsq (least squares)": "lsq", "bcs (Bayesian sparse)": "bcs", "anl (analytical)": "anl"},
        value="lsq", label="--regression",
    )
    q_bins  = mo.ui.slider(5, 20, step=5, value=10, label="--n-bins", show_value=True)
    return q_bins, q_order, q_reg, sample_gens, sample_n, sample_obs, sample_seed


@app.cell
def _l6_commands(mo, q_bins, q_order, q_reg, sample_gens, sample_n, sample_obs, sample_seed):
    from math import comb as _comb6
    _N6 = int(sample_n.value)
    _d6 = 6
    _p6 = int(q_order.value)
    _n_terms6 = _comb6(_d6 + _p6, _p6)
    _ok = "✓" if _N6 >= 2 * _n_terms6 else "⚠ N < 2×terms — consider increasing"

    mo.vstack([
        mo.md("### Live Command Builder — Adjust Sliders"),
        mo.Html(f"""
<div class="uq-cmd">uv run uq sample /path/to/simData.cPickle \\
    --cache-dir ./uq_cache \\
    --n-samples {_N6} \\
    --generations {int(sample_gens.value)} \\
    --n-init-sims {int(sample_seed.value)} \\
    --observables {sample_obs.value} \\
    --n-test 10</div>
<div class="uq-cmd">uv run uq quantify /path/to/simData.cPickle \\
    --cache-dir ./uq_cache \\
    --export-path ./uq_results \\
    --polynomial-order {_p6} \\
    --regression {q_reg.value} \\
    --n-bins {int(q_bins.value)}</div>
"""),
        mo.Html(f"""
<div class="uq-tip">
  <span class="tip-label">PCE SIZING:</span>
  d={_d6} params, p={_p6} order → <strong>{_n_terms6} basis terms</strong>.
  N={_N6} samples {_ok}
</div>
"""),
        mo.md(f"""
**After `uq quantify` completes:**

```
uq_results/
├── uq_results.json          # All Sobol indices (all 4 strategies)
├── report.html              # Self-contained interactive HTML report
├── manifest.json            # Provenance: git SHAs, packages, CLI args
├── population_surrogate/    # PCE coefficients + multi-indices (.npy)
├── strategy2_by_generation/ # Per-generation Sobol
├── strategy3_by_seed/       # Per-seed Sobol
├── strategy4_growth_stratified/  # Per-θ-bin Sobol + surrogates
└── variance_decomposition.json   # σ² budget (gen / seed / θ / residual)
```

Open the report: `uv run uq report --results-path ./uq_results`
"""),
    ])
    return


# ════════════════════════════════════════════════════════════════════════════
#  LEVEL 7 — ADVANCED FEATURES
# ════════════════════════════════════════════════════════════════════════════


@app.cell
def _l7_gate(level, mo):
    mo.stop(int(level.value) < 7)
    return


@app.cell
def _l7_selector(mo):
    feature_selector = mo.ui.dropdown(
        options={
            "🔬  Output PCA  (--output-pca K)":                "pca",
            "📊  Q1 Baseline  (--no-perturbation)":            "baseline",
            "☁️   Remote SMS-API  (--api-url)":                "remote",
            "🔄  Multi-Condition  (--conditions)":             "multicond",
            "📐  Held-out Validation  (--n-test)":             "validation",
            "🗃️   HPC Batch  (export-configs + collect-results)": "hpc",
        },
        value="pca", label="Advanced Feature",
    )
    return (feature_selector,)


@app.cell
def _l7_content(feature_selector, go, mo, np):
    _features7 = {
        "pca": {
            "color": "#00f0ff",
            "title": "Output-Side PCA — `--output-pca K`",
            "content": """
When Y has thousands of columns (transcriptome ~4300, proteome ~4300, fluxome ~2800),
fitting individual PCE surrogates is expensive and noisy. PCA collapses them first.

```bash
uv run uq quantify /path/to/simData.cPickle \\
    --cache-dir ./uq_cache \\
    --observables transcriptome \\
    --output-pca 5
```

**What happens:**
1. Center Y (n × 4300) → SVD → top-5 principal components
2. Fit 5 PCE surrogates (one per PC) instead of 4300
3. Sobol indices are per-PC: "which parameter drives PC1?"
4. Top loadings decode biology: "PC1 ≈ global growth signature"

**Export files:**
- `pca/pca_loadings.npy` — (K × n_obs) loading matrix
- `pca/pca_explained_variance.npy` — variance fraction per PC
- `pca/pca_top_loadings.json` — top-10 observables per PC

**Recommended:** `--observables transcriptome --output-pca 5`
""",
        },
        "baseline": {
            "color": "#33ff99",
            "title": "Q1 Baseline Variance Budget — `--no-perturbation`",
            "content": """
Run vEcoli at **fixed baseline** (no parameter perturbations) to decompose
output variance into natural biological sources.

```bash
# Step 1: run baseline (no variants)
uv run uq sample /path/to/simData.cPickle \\
    --no-perturbation --n-init-sims 5 --generations 3

# Step 2: compute σ² budget
uv run uq quantify /path/to/simData.cPickle --no-perturbation
```

**4-way decomposition:**

| Component | Meaning |
|---|---|
| σ²_gen | Drift across generations (convergence to steady-state) |
| σ²_seed | Stochastic seed-to-seed variation |
| σ²_θ | Cell cycle phase variation |
| σ²_residual | Unexplained (numerical noise) |

**Why this matters:** σ²_total from baseline is the *denominator* for
perturbation sensitivity — tells you whether parameter effects are
detectable above biological noise.
""",
        },
        "remote": {
            "color": "#aa66ff",
            "title": "Remote Execution via SMS-API — `--api-url`",
            "content": """
Run vEcoli on a remote cluster instead of locally:

```bash
uv run uq sample /path/to/simData.cPickle \\
    --api-url https://sms.cam.uchc.edu \\
    --simulator-id 11 \\
    --n-samples 50 \\
    --observables transcriptome
```

**What happens:**
1. Build N sim_data_setattr variants from LHS samples
2. Submit ONE simulation with N variants to AWS Batch
3. Poll until complete (progress bar)
4. Download cd1 analysis TSVs (one archive)
5. Parse → cache (identical format to local)

**Fetch outputs from an existing sim:**
```bash
uv run uq fetch 48 --api-url https://sms.cam.uchc.edu
```

**Key advantage:** No local vEcoli installation needed. `uq quantify` is
identical regardless of how samples were generated.
""",
        },
        "multicond": {
            "color": "#ffaa00",
            "title": "Cross-Condition GSA — `--conditions`",
            "content": """
Run the same UQ analysis under multiple growth conditions simultaneously:

```bash
uv run uq sample /path/to/simData.cPickle \\
    --conditions vecoli_m9_glucose_minus_aas \\
    --conditions vecoli_m9_glucose_plus_aas

uv run uq quantify /path/to/simData.cPickle  # auto-detects multi-condition
```

**Auto-detection:** `quantify` reads `conditions.json` from the cache
directory and runs per-condition + cross-condition analysis.

| Metric | Meaning |
|---|---|
| Rank stability | How consistent is parameter ranking across conditions? |
| Universal drivers | S_Ti > 10% in ALL conditions |
| Condition-specific | Parameters that matter in only one condition |
| Differential Sobol | ΔS_Ti = |S_Ti(A) − S_Ti(B)| |

**Output:** Cross-condition comparison table with stability indicators
(`●●●○○` = 3/5 conditions show this parameter as a top driver).
""",
        },
        "validation": {
            "color": "#ff3366",
            "title": "Held-out Validation — `--n-test`",
            "content": """
Reserve samples for out-of-sample PCE quality assessment (UQPC `--ntst`):

```bash
uv run uq sample /path/to/simData.cPickle \\
    --n-samples 50 --n-test 10
```

**What gets cached:** `X_test.npy`, `Y_test.npy`, `timeseries_test/`

**In `uq quantify` output:**
```
SURROGATE QUALITY // RELATIVE ERRORS
┌──────────┬───────────┬──────────┐
│ OBSERV.  │   TRAIN   │   TEST   │
├──────────┼───────────┼──────────┤
│ output[0]│  1.2e-03  │  1.8e-03 │
│ output[1]│  2.1e-03  │  2.9e-03 │
└──────────┴───────────┴──────────┘
```

**Diagnostics:**
- Test error ≈ train error → good generalization
- Test >> train → overfitting → lower `--polynomial-order`
- Both > 0.1 → insufficient N → add more samples
""",
        },
        "hpc": {
            "color": "#888888",
            "title": "HPC Batch Workflow — `export-configs` + `collect-results`",
            "content": """
For large-scale runs (N=200+) on Slurm/Nextflow clusters:

```bash
# 1. Export per-sample configs (no vEcoli runs yet)
uv run uq export-configs /path/to/simData.cPickle ./batch --n-samples 200

# 2. Run on cluster  (each job: one pickled simData + one config JSON)
#    e.g. via Nextflow, Slurm array, or manual dispatch

# 3. Collect outputs into cache
uv run uq collect-results ./batch ./batch_outputs --cache-dir ./uq_cache

# 4. Analyze identically to local mode
uv run uq quantify /path/to/simData.cPickle --cache-dir ./uq_cache
```

**Key design:** Variants are baked into each pickled `simData` at export time.
HPC jobs only need vEcoli — no UQ codebase required.

**Export structure:**
```
batch/
├── simData/0.cPickle   ← simData with LHS variant 0 applied
├── simData/1.cPickle   ...
├── config/0.json       ← vEcoli workflow config for variant 0
├── metadata.json       ← maps index → parameter values
└── ...
```
""",
        },
    }

    _feat = _features7[feature_selector.value]
    mo.vstack([
        mo.Html(f"""
<div style="border-left: 4px solid {_feat['color']}; padding: 12px 16px; background: #1a1a2e;
     border-radius: 4px; margin: 8px 0;">
  <div style="color: {_feat['color']}; font-weight: bold; font-size: 1.0em; margin-bottom: 4px;">
    {_feat['title']}
  </div>
</div>
"""),
        mo.md(_feat["content"]),
        mo.Html("""
<div class="uq-achievement">
  <div class="label">⚡ Skill Unlocked — Advanced Features</div>
  <div class="body">You now know every production-grade feature of the UQ framework.
  Combine them freely: e.g. remote + multi-condition + output-pca + held-out validation
  in a single campaign.</div>
</div>
"""),
    ])
    return


# ════════════════════════════════════════════════════════════════════════════
#  LEVEL 8 — CLI ARSENAL
# ════════════════════════════════════════════════════════════════════════════


@app.cell
def _l8_gate(level, mo):
    mo.stop(int(level.value) < 8)
    return


@app.cell
def _l8_header(mo):
    mo.Html("""
<div style="background: linear-gradient(90deg, #1a0a2e, #0a1a2e);
     border: 2px solid #aa66ff; border-radius: 8px; padding: 20px 24px; margin: 8px 0;
     font-family: 'Inter', sans-serif;">
  <div style="font-size: 0.75em; letter-spacing: 0.3em; color: #aa66ff; text-transform: uppercase;">
    ⚔️  LEVEL 8 — CLI ARSENAL
  </div>
  <div style="font-size: 1.3em; font-weight: 700; color: #e0e0e0; margin: 4px 0;">
    Every <code style="color: #00f0ff;">uq</code> command, with examples
  </div>
  <div style="font-size: 0.85em; color: #888;">
    The two core commands (<code>sample</code>, <code>quantify</code>) were covered in Levels 2–7.
    Here are all the supporting commands that complete the toolkit.
  </div>
</div>
""")
    return


@app.cell
def _l8_selector(mo):
    cmd_selector = mo.ui.dropdown(
        options={
            "uq dashboard  — interactive visualization": "dashboard",
            "uq tui        — terminal UI (Textual)": "tui",
            "uq gui        — browser GUI (marimo)": "gui",
            "uq tutorial   — this tutorial": "tutorial",
            "uq report     — generate/re-generate HTML report": "report",
            "uq show-config — preview vEcoli config JSON": "show_config",
            "uq compare    — side-by-side Sobol comparison": "compare",
            "uq export-figures — publication PDFs + LaTeX": "export_figures",
            "uq suggest-experiment — where to measure next": "suggest",
            "uq fetch      — download SMS-API simulation outputs": "fetch",
            "uq init       — guided project setup wizard": "init",
            "uq help       — help for any subcommand": "help_cmd",
        },
        value="dashboard", label="Command",
    )
    return (cmd_selector,)


@app.cell
def _l8_ref(cmd_selector, mo):
    _cmds8 = {
        "dashboard": {
            "sig": "uq dashboard [--results-path PATH] [--run-mode tk|mo|web]",
            "desc": "Interactive visualization of a completed UQ result.",
            "detail": """
**Three modes:**

| Flag | Mode | Description |
|---|---|---|
| *(default)* | `tk` | Tkinter DAW — draggable parameter markers, sensitivity spectrogram, tracking dot |
| `--run-mode mo` | marimo | Slider-reactive browser notebook |
| `--run-mode web` | Dash | Standalone web app with shareable URL |

**Panels (all modes):**
- PCE Response Curves — Y(θ) as each parameter varies
- Sensitivity Spectrogram — S_Ti heatmap (params × stages)
- Observable Waveform — baseline vs modulated

```bash
uv run uq dashboard                          # auto-detect ./uq_results
uv run uq dashboard --results-path ./my_run
uv run uq dashboard --run-mode mo            # marimo browser
```
""",
        },
        "tui": {
            "sig": "uq tui",
            "desc": "Textual terminal UI — full workflow in the terminal.",
            "detail": """
A keyboard-driven terminal UI (built with Textual) that guides you through
the full `sample → quantify` workflow without leaving the terminal.

**Keyboard shortcuts:**
- `Tab` / `Shift+Tab` — navigate panels
- `Enter` — run selected action
- `q` — quit
- `d` — toggle dark/light theme

**Panels:**
1. Configure — set simData path, n_samples, observables
2. Sample — run `uq sample` with live progress bar
3. Quantify — run `uq quantify` and see Rich report inline
4. Results — browse exported Sobol tables

```bash
uv run uq tui
```
""",
        },
        "gui": {
            "sig": "uq gui",
            "desc": "Full browser GUI — configure, sample, quantify, explore in marimo.",
            "detail": """
A self-contained marimo notebook GUI that provides the complete workflow
in the browser. No command line required after launch.

**Features:**
- Step-by-step parameter configuration
- Real-time sampling progress
- Post-quantify PCE explorer
- Sobol bar charts + spectrogram
- Export to PDF/HTML

```bash
uv run uq gui
# Opens browser at http://localhost:2718
```
""",
        },
        "tutorial": {
            "sig": "uq tutorial",
            "desc": "Launch this interactive tutorial.",
            "detail": """
Spins up this marimo app (`app/tutorial.py`) in the browser.

```bash
uv run uq tutorial
# Opens browser at http://localhost:2718
```

This tutorial covers all 10 concepts with live computations. Complete
all 10 levels to fully understand the UQ CLI.
""",
        },
        "report": {
            "sig": "uq report [--results-path PATH] [--output PATH]",
            "desc": "Generate (or re-generate) a self-contained HTML report.",
            "detail": """
Creates `report.html` from `uq_results.json` and `manifest.json`.
Already called automatically at the end of `uq quantify`, but useful for:
- Re-styling an existing result
- Sharing a single-file artifact

```bash
uv run uq report                               # default: ./uq_results
uv run uq report --results-path ./my_run
uv run uq report --output ./report_v2.html

# For Q1 baseline results:
uv run uq report --results-path ./baseline_results
```

**Report sections:**
- Sobol rankings (all 4 strategies)
- PCE response curves (interactive, client-side JS)
- Sensitivity spectrogram SVG
- Variance decomposition bar chart
- PCA scree plot (if --output-pca was used)
- Provenance / manifest table
""",
        },
        "show_config": {
            "sig": "uq show-config SIMDATA [--n-samples N] [--seed N]",
            "desc": "Preview the vEcoli workflow config JSON without running anything.",
            "detail": """
Generates the complete vEcoli config JSON that `uq sample` would write and pass
to `runscripts/workflow.py`. Useful for reviewing variants before a long run.

```bash
uv run uq show-config /path/to/simData.cPickle \\
    --n-samples 3 --generations 2 --n-init-sims 2

# Write to file for manual inspection / HPC submission
uv run uq show-config /path/to/simData.cPickle \\
    --output config_preview.json
```

**Output includes:**
- `variants` section — all N sim_data_setattr mutations
- `n_init_sims`, `generations`, `max_duration`
- Experiment ID, output directory
- Parameter values in physical space

**Total sims printed:** `(N + 1) × n_init_sims × generations`
(+1 = baseline variant 0)
""",
        },
        "compare": {
            "sig": "uq compare DIR1 DIR2 [DIR3 ...]",
            "desc": "Side-by-side Sobol comparison across multiple UQ experiments.",
            "detail": """
Loads `uq_results.json` from each directory and prints a comparison table
with Δ highlighting for two-experiment case.

```bash
uv run uq compare ./results_glucose ./results_minimal_aa

# Three-way comparison
uv run uq compare ./results_a ./results_b ./results_c
```

**Output (two-experiment example):**
```
S_Ti COMPARISON (Population)
┌───────────────────────┬────────────┬────────────┬────────┐
│ PARAMETER             │ glucose    │ minimal_aa │ Δ      │
├───────────────────────┼────────────┼────────────┼────────┤
│ kinetic_obj_weight    │ 0.412      │ 0.389      │ -0.023 │
│ rnap_free_frac        │ 0.234      │ 0.301      │ +0.067 │
└───────────────────────┴────────────┴────────────┴────────┘
```
""",
        },
        "export_figures": {
            "sig": "uq export-figures [--results-path PATH] [--output-dir PATH]",
            "desc": "Generate publication-ready PDFs and LaTeX from UQ results.",
            "detail": """
Creates static, print-quality figures from a completed UQ result.
Requires `kaleido` for Plotly PDF export.

```bash
uv pip install kaleido  # one-time
uv run uq export-figures --results-path ./uq_results
```

**Outputs:**
```
uq_results/figures/
├── sobol_bar_chart.pdf   — grouped bars (S_Ti per strategy, all params)
├── spectrogram.pdf       — print-quality sensitivity heatmap
├── response_curves.pdf   — PCE response curves at midpoint
└── sobol_table.tex       — LaTeX table, top-K params per strategy
```
""",
        },
        "suggest": {
            "sig": "uq suggest-experiment [--results-path PATH] [--n-grid N]",
            "desc": "Identify where to measure next to reduce prediction uncertainty most.",
            "detail": """
Uses the fitted PCE surrogate to scan the parameter space and find the
region of maximum prediction variance. Points to where new experiments
would be most informative.

```bash
uv run uq suggest-experiment --results-path ./uq_results --n-grid 5000
```

**Output:**
```
SUGGESTED NEXT EXPERIMENT
┌───────────────────────┬──────────┬──────────────────────┐
│ PARAMETER             │ VALUE    │ RANGE                │
├───────────────────────┼──────────┼──────────────────────┤
│ kinetic_obj_weight    │ 4.12e-7  │ [5e-8, 5e-7]         │
│ rnap_free_frac        │ 0.43     │ [0.25, 0.47]         │
└───────────────────────┴──────────┴──────────────────────┘
Recommendation: Measure rnap_free_frac more precisely in [0.38, 0.47]
```
""",
        },
        "fetch": {
            "sig": "uq fetch SIM_ID [--api-url URL] [--observables ...]",
            "desc": "Download and parse cd1 analysis outputs from a completed SMS-API simulation.",
            "detail": """
Downloads the simulation archive, parses cd1 TSV files, and prints a
summary. Use to inspect what the SMS-API produces before running a full
UQ campaign.

```bash
uv run uq fetch 48 --api-url https://sms.cam.uchc.edu

# Inspect specific observable modules
uv run uq fetch 48 --observables higher_order --observables transcriptome
```

**Output:**
```
cd1 Analysis Outputs
┌──────────────────┬──────────────────────┬───────┐
│ PRESET           │ MODULE               │ ROWS  │
├──────────────────┼──────────────────────┼───────┤
│ higher_order     │ cd1_growth_analysis  │ 6     │
│ transcriptome    │ cd1_transcriptomics  │ 4345  │
│ proteome         │ cd1_proteomics       │ 4345  │
│ exchange_fluxes  │ cd1_metabolism       │ 87    │
└──────────────────┴──────────────────────┴───────┘
Total observables: 8783
```
""",
        },
        "init": {
            "sig": "uq init",
            "desc": "Guided interactive project setup wizard.",
            "detail": """
Detects vEcoli, validates simData, lets you choose observable presets,
and writes `uq_config.json` ready to paste into `uq sample`.

```bash
uv run uq init
```

**Wizard steps:**
1. Auto-detect `simData.cPickle` (searches common paths)
2. Validate vEcoli is importable
3. Choose observable preset (with descriptions + feature counts)
4. Choose N samples (with time estimates)
5. Choose regression method
6. Write `uq_config.json` + print exact `uq sample` command

**Best for:** First-time setup or new projects.
""",
        },
        "help_cmd": {
            "sig": "uq help [COMMAND]",
            "desc": "Show help for a subcommand, or the main CLI.",
            "detail": """
Every subcommand also accepts a trailing `help` word as an alias for `--help`.

```bash
uv run uq help             # list all commands
uv run uq help sample      # sample flags
uv run uq help quantify    # quantify flags
uv run uq sample help      # same as uq help sample
uv run uq quantify help    # same as uq help quantify
```

**Pro tip:** All flags are documented inline. Run `uq <cmd> --help` to see
the full flag reference with default values and descriptions.
""",
        },
    }

    _cd = _cmds8[cmd_selector.value]
    mo.vstack([
        mo.Html(f"""
<div style="background: #1a1a2e; border-left: 4px solid #aa66ff; padding: 12px 16px;
     border-radius: 4px; margin: 4px 0; font-family: 'Menlo', monospace;">
  <span style="color: #00f0ff; font-weight: bold;">{_cd['sig']}</span><br/>
  <span style="color: #888; font-size: 0.9em;">{_cd['desc']}</span>
</div>
"""),
        mo.md(_cd["detail"]),
    ])
    return


@app.cell
def _l8_summary(mo):
    mo.Html("""
<div style="background: #1a1a2e; border: 1px solid #2a2a4a; border-radius: 8px;
     padding: 16px 20px; margin: 12px 0; font-family: 'Inter', sans-serif;">
  <div style="color: #aa66ff; font-weight: bold; margin-bottom: 8px;">
    ⚔️  COMPLETE CLI REFERENCE
  </div>
  <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 4px 16px; font-size: 0.85em;">
    <div><span style="color: #00f0ff; font-family: monospace;">uq sample</span>
         <span style="color: #888;"> — UQPC steps 1–3 (parameterize + run vEcoli)</span></div>
    <div><span style="color: #00f0ff; font-family: monospace;">uq quantify</span>
         <span style="color: #888;"> — UQPC steps 4–5 (PCE + Sobol)</span></div>
    <div><span style="color: #33ff99; font-family: monospace;">uq dashboard</span>
         <span style="color: #888;"> — tk/marimo/web visualization</span></div>
    <div><span style="color: #33ff99; font-family: monospace;">uq tui</span>
         <span style="color: #888;"> — Textual terminal UI</span></div>
    <div><span style="color: #33ff99; font-family: monospace;">uq gui</span>
         <span style="color: #888;"> — marimo browser GUI</span></div>
    <div><span style="color: #33ff99; font-family: monospace;">uq tutorial</span>
         <span style="color: #888;"> — this tutorial</span></div>
    <div><span style="color: #ffaa00; font-family: monospace;">uq report</span>
         <span style="color: #888;"> — generate HTML report</span></div>
    <div><span style="color: #ffaa00; font-family: monospace;">uq show-config</span>
         <span style="color: #888;"> — preview vEcoli config JSON</span></div>
    <div><span style="color: #ffaa00; font-family: monospace;">uq compare</span>
         <span style="color: #888;"> — side-by-side Sobol comparison</span></div>
    <div><span style="color: #ffaa00; font-family: monospace;">uq export-figures</span>
         <span style="color: #888;"> — publication PDFs + LaTeX</span></div>
    <div><span style="color: #aa66ff; font-family: monospace;">uq suggest-experiment</span>
         <span style="color: #888;"> — max-uncertainty parameter region</span></div>
    <div><span style="color: #aa66ff; font-family: monospace;">uq fetch</span>
         <span style="color: #888;"> — download SMS-API simulation data</span></div>
    <div><span style="color: #aa66ff; font-family: monospace;">uq init</span>
         <span style="color: #888;"> — guided project setup wizard</span></div>
    <div><span style="color: #aa66ff; font-family: monospace;">uq help</span>
         <span style="color: #888;"> — help for any subcommand</span></div>
  </div>
</div>
""")
    return


# ════════════════════════════════════════════════════════════════════════════
#  LEVEL 9 — BOSS LEVEL: Interactive PCE Explorer
# ════════════════════════════════════════════════════════════════════════════


@app.cell
def _l9_gate(level, mo):
    mo.stop(int(level.value) < 9)
    return


@app.cell
def _l9_boss_intro(mo):
    mo.Html("""
<div class="uq-boss">
  <div style="font-size: 0.7em; letter-spacing: 0.3em; color: #aa66ff; text-transform: uppercase; margin-bottom: 6px;">
    👾 BOSS LEVEL — PROVE MASTERY
  </div>
  <div style="font-size: 1.2em; font-weight: 800; color: #e0e0e0; margin-bottom: 8px;">
    Interactive PCE Explorer
  </div>
  <div style="font-size: 0.85em; color: #888; line-height: 1.6;">
    A live PCE surrogate (order 3, 3 parameters, 100 samples) is fitted below.
    The sliders control parameter values. The response curves update in real time,
    and the Sobol bars show each parameter's contribution to total variance.<br/><br/>
    <strong style="color: #00f0ff;">Challenge:</strong> Identify which parameter dominates variance,
    and find the region where the output changes fastest.
    This is exactly what <code>uq dashboard</code> provides after a real vEcoli run.
  </div>
</div>
""")
    return


@app.cell
def _l9_fit(np):
    from libuq.inputs import XSpaceVecoli as _XS9
    from libuq.pipeline.models import SimDataParameter as _P9
    from uq.workflow import run_uqpc as _run9

    _rng9 = np.random.RandomState(42)
    _N9   = 100
    _d9   = 3
    _X9   = _rng9.uniform(0, 1, (_N9, _d9))
    _Y9   = (
        3.0 * _X9[:, 0]
        + 1.5 * _X9[:, 1]
        + 0.3 * _X9[:, 2]
        + 0.5 * _X9[:, 0] * _X9[:, 1]
        + 0.2 * _X9[:, 0] ** 2
    ).reshape(-1, 1)

    _ps9 = _XS9(
        parameter_names=[], parameter_bounds=[], parameter_types=[],
        experiment_id="boss",
        parameters=[_P9(name=f"p{i}", attr_path=f"x.p{i}", bounds=(0.0, 1.0)) for i in range(_d9)],
    )

    live_result = _run9(
        param_space=_ps9,
        Y_train=_Y9,
        X_train=_X9,
        polynomial_order=3,
        regression="lsq",
        seed=42,
    )
    return (live_result,)


@app.cell
def _l9_sliders(mo):
    p1 = mo.ui.slider(0, 1, step=0.01, value=0.5, label="p₁", show_value=True, full_width=True)
    p2 = mo.ui.slider(0, 1, step=0.01, value=0.5, label="p₂", show_value=True, full_width=True)
    p3 = mo.ui.slider(0, 1, step=0.01, value=0.5, label="p₃", show_value=True, full_width=True)
    return p1, p2, p3


@app.cell
def _l9_explore(go, live_result, mo, np, p1, p2, p3):
    _x_curr = np.array([[p1.value, p2.value, p3.value]])
    _y_curr = float(live_result.surrogate.predict(_x_curr)[0, 0])

    _s1_live = live_result.sobol.first_order
    _st_live = live_result.sobol.total_order

    # Response curves — sweep each param, fix others at current
    _n_sw = 60
    _sweep9 = np.linspace(0, 1, _n_sw)
    _colors9 = ["#00f0ff", "#aa66ff", "#33ff99"]
    _param_labels9 = ["p₁", "p₂", "p₃"]

    _fig9a = go.Figure()
    for _pi9 in range(3):
        _Xsw9 = np.tile(_x_curr, (_n_sw, 1))
        _Xsw9[:, _pi9] = _sweep9
        _Ysw9 = live_result.surrogate.predict(_Xsw9).flatten()
        _fig9a.add_trace(go.Scatter(
            x=_sweep9, y=_Ysw9, mode="lines",
            name=f"Sweep {_param_labels9[_pi9]}",
            line=dict(color=_colors9[_pi9], width=2),
        ))
        _fig9a.add_trace(go.Scatter(
            x=[_x_curr[0, _pi9]], y=[_y_curr], mode="markers",
            marker=dict(symbol="diamond", size=14, color=_colors9[_pi9],
                        line=dict(color="#ffffff", width=1)),
            showlegend=False,
        ))

    _fig9a.update_layout(
        title=dict(text="PCE Response Curves — Move Sliders to Explore", x=0.5,
                   font=dict(color="#e0e0e0", size=13)),
        xaxis_title="Parameter Value",
        yaxis_title="Predicted Output  Ŷ",
        template="plotly_dark", height=360,
        paper_bgcolor="#0d0d0d", plot_bgcolor="#0d0d0d",
        margin=dict(l=50, r=20, t=50, b=40),
        legend=dict(x=0.01, y=0.99, bgcolor="rgba(0,0,0,0)"),
    )

    # Sobol bar chart
    _xpos9 = np.arange(3)
    _w9 = 0.35
    _fig9b = go.Figure()
    _fig9b.add_trace(go.Bar(x=_xpos9 - _w9/2, y=_s1_live, width=_w9,
        name="Sᵢ (first-order)", marker_color="#aa66ff"))
    _fig9b.add_trace(go.Bar(x=_xpos9 + _w9/2, y=_st_live, width=_w9,
        name="S_Ti (total-order)", marker_color="#00f0ff"))
    _fig9b.update_layout(
        barmode="group",
        title=dict(text="Sobol Indices (analytical)", x=0.5,
                   font=dict(color="#e0e0e0", size=12)),
        xaxis=dict(ticktext=_param_labels9, tickvals=[0, 1, 2]),
        yaxis_title="Variance Fraction",
        template="plotly_dark", height=260,
        paper_bgcolor="#0d0d0d", plot_bgcolor="#0d0d0d",
        margin=dict(l=50, r=20, t=45, b=40),
        legend=dict(x=0.01, y=0.99, bgcolor="rgba(0,0,0,0)"),
    )

    _err9 = float(np.mean(live_result.relerr_train)) if live_result.relerr_train is not None else 0.0
    _n_terms9 = len(live_result.surrogate.multi_indices)
    _top_idx9 = int(np.argmax(_st_live))

    _stat_box = mo.Html(f"""
<div style="background: #1a1a2e; border: 1px solid #2a2a4a; border-radius: 6px;
     padding: 12px 16px; font-family: 'Menlo', monospace; font-size: 0.85em; color: #e0e0e0;">
  <div style="color: #888; margin-bottom: 6px;">CURRENT POINT</div>
  <div>Ŷ(p₁={p1.value:.2f}, p₂={p2.value:.2f}, p₃={p3.value:.2f}) =
    <span style="color: #00f0ff; font-weight: bold;">{_y_curr:.4f}</span>
  </div>
  <div style="margin-top: 8px; color: #888;">SURROGATE</div>
  <div>Order 3 | {_n_terms9} terms | train err: {_err9:.2e}</div>
  <div style="margin-top: 8px; color: #888;">DOMINANT PARAM</div>
  <div style="color: #ffaa00; font-weight: bold;">
    p{_top_idx9 + 1}  (S_Ti = {_st_live[_top_idx9]:.3f})
  </div>
</div>
""")

    mo.vstack([
        mo.hstack([
            mo.vstack([
                mo.md("**Parameter Controls**"),
                p1, p2, p3,
                _stat_box,
            ]),
            mo.vstack([_fig9a, _fig9b]),
        ]),
    ])
    return


# ════════════════════════════════════════════════════════════════════════════
#  MISSION COMPLETE
# ════════════════════════════════════════════════════════════════════════════


@app.cell
def _l9_complete(level, mo):
    mo.stop(int(level.value) < 9)
    mo.Html("""
<div style="
  background: linear-gradient(135deg, #0a1a0a 0%, #0a0a1a 50%, #1a0a0a 100%);
  border: 2px solid #33ff99;
  border-radius: 12px;
  padding: 32px 40px;
  text-align: center;
  font-family: 'Inter', sans-serif;
  margin: 16px 0;
">
  <div style="font-size: 0.7em; letter-spacing: 0.35em; color: #33ff99; text-transform: uppercase; margin-bottom: 8px;">
    ✓ ALL LEVELS COMPLETE
  </div>
  <div style="font-size: 2em; font-weight: 900; color: #e0e0e0; margin-bottom: 6px;">
    MISSION COMPLETE
  </div>
  <div style="font-size: 0.95em; color: #888; max-width: 640px; margin: 0 auto 20px; line-height: 1.7;">
    You now understand the complete UQ framework for vEcoli whole-cell simulations —
    from parameter space design through sampling, PCE fitting, Sobol analysis,
    and all 14 CLI commands.
  </div>

  <div style="
    display: grid; grid-template-columns: repeat(5, 1fr); gap: 8px;
    max-width: 700px; margin: 0 auto 20px;
  ">
    <div style="background:#1a1a2e;border:1px solid #00f0ff;border-radius:6px;padding:8px;font-size:0.75em;color:#00f0ff;">🎯 Problem</div>
    <div style="background:#1a1a2e;border:1px solid #00f0ff;border-radius:6px;padding:8px;font-size:0.75em;color:#00f0ff;">📐 Params</div>
    <div style="background:#1a1a2e;border:1px solid #00f0ff;border-radius:6px;padding:8px;font-size:0.75em;color:#00f0ff;">🎲 Sampling</div>
    <div style="background:#1a1a2e;border:1px solid #00f0ff;border-radius:6px;padding:8px;font-size:0.75em;color:#00f0ff;">📦 PCE</div>
    <div style="background:#1a1a2e;border:1px solid #00f0ff;border-radius:6px;padding:8px;font-size:0.75em;color:#00f0ff;">📊 Sobol</div>
    <div style="background:#1a1a2e;border:1px solid #33ff99;border-radius:6px;padding:8px;font-size:0.75em;color:#33ff99;">🔬 Strategies</div>
    <div style="background:#1a1a2e;border:1px solid #33ff99;border-radius:6px;padding:8px;font-size:0.75em;color:#33ff99;">🚀 Pipeline</div>
    <div style="background:#1a1a2e;border:1px solid #33ff99;border-radius:6px;padding:8px;font-size:0.75em;color:#33ff99;">⚡ Advanced</div>
    <div style="background:#1a1a2e;border:1px solid #33ff99;border-radius:6px;padding:8px;font-size:0.75em;color:#33ff99;">⚔️ Arsenal</div>
    <div style="background:#1a1a2e;border:1px solid #33ff99;border-radius:6px;padding:8px;font-size:0.75em;color:#33ff99;">👾 Boss</div>
  </div>

  <div style="font-size: 0.85em; color: #888; margin-bottom: 4px;">NEXT STEPS</div>
  <div style="
    display: grid; grid-template-columns: 1fr 1fr; gap: 6px;
    max-width: 560px; margin: 0 auto;
    font-family: 'Menlo', monospace; font-size: 0.8em;
  ">
    <div style="background:#0d0d0d;border:1px solid #2a2a4a;border-radius:4px;padding:8px;color:#00f0ff;">uv run uq init</div>
    <div style="background:#0d0d0d;border:1px solid #2a2a4a;border-radius:4px;padding:8px;color:#33ff99;">uv run uq sample ...</div>
    <div style="background:#0d0d0d;border:1px solid #2a2a4a;border-radius:4px;padding:8px;color:#ffaa00;">uv run uq quantify ...</div>
    <div style="background:#0d0d0d;border:1px solid #2a2a4a;border-radius:4px;padding:8px;color:#aa66ff;">uv run uq dashboard</div>
  </div>
</div>
""")
    return


if __name__ == "__main__":
    app.run()
