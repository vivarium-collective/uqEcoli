# Marimo Interactive Notebooks

Interactive notebooks exported as WASM-powered HTML. These run entirely in
your browser via Pyodide (WebAssembly) — no Python server needed.

## Available Notebooks

| Notebook | Description | WASM HTML |
|----------|-------------|-----------|
| `03c_reactive_sensitivity_generalized` | Multi-parameter reactive sensitivity | [reactive.wasm.html](reactive.wasm.html) (76 KB, generated) |

## How to Regenerate

```bash
# From the repo root, export each notebook to WASM HTML:
uv run marimo export html-wasm tutorials/01_introduction.py -o demo/marimo/intro.wasm.html
uv run marimo export html-wasm tutorials/03c_reactive_sensitivity_generalized.py -o demo/marimo/reactive.wasm.html
```

**Note:** WASM-exported notebooks run on Pyodide, which supports most (but not
all) Python packages. Notebooks that depend on native extensions (e.g., PyTUQ)
may not be fully functional in WASM. The exported HTML files are still useful
as interactive demos of the notebook interface and workflow structure.

## Screenshot Guide

### 3a. `uq gui` Parameter Sliders
```bash
uv run uq gui
```
In the marimo browser tab that opens:
1. Drag the parameter sliders (PCE response curves update in real time)
2. Capture the slider panel + response curve area together
3. Save as `demo/marimo/gui_sliders.png`

### 3b. Reactive Sensitivity Notebook
```bash
uv run marimo run tutorials/03c_reactive_sensitivity_generalized.py
```
1. Drag the multi-parameter interaction widgets
2. Capture the full notebook view showing the reactive charts
3. Save as `demo/marimo/reactive_sensitivity.png`

### 3c. Screen Recording (GIF)
Use your preferred screen capture tool to record a short GIF (5-10 seconds)
showing slider dragging and live updates:
- **macOS:** QuickTime Player (screen recording) + ffmpeg to convert to GIF
- **OBS Studio:** Free, cross-platform
- **Peek (Linux):** Simple GIF screen recorder
- Save as `demo/marimo/slider_demo.gif`

### 3d. Tutorial Level Select
```bash
uv run uq tutorial
```
Capture the CLI output showing the available tutorial topics/levels.
Save as `demo/marimo/tutorial_cli.png`.

## WASM Deployment

The WASM HTML files are self-contained and served statically via GitHub Pages
at https://vivarium-collective.github.io/uqEcoli/demo/marimo/.
