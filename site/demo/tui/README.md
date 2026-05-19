# Terminal UI (TUI)

The UQPC interactive terminal UI built with [Textual](https://textual.textualize.io/).

## Usage

```bash
# From the repo root, with cached results available:
uv run uq tui

# Or from an installed package:
pip install uqecoli
uq tui
```

## Features

- **Parameter importance table** — ranked Sobol indices with color bars
- **Observable selector** — cycle through tracked observables with keyboard or mouse
- **Live mode** — periodic refresh for real-time monitoring
- **Keyboard shortcuts** — full keyboard navigation (Textual framework)

## Screenshots

To capture TUI screenshots:

### Option 1: `screencapture` (macOS)
```bash
# Run the TUI in one terminal, then in another:
screencapture -w demo/tui/tui_dashboard.png
```

### Option 2: Textual Devtools
```bash
# Run with devtools for SVG output:
textual run --dev uq/tui.py
# Then use the web browser devtools at http://localhost:8080
```

### Option 3: `script(1)` + HTML rendering (fallback)
```bash
script -q demo/tui/tui_session.log
uq tui
# Press Ctrl+D to exit script
# Then convert to HTML using ansi2html:
ansi2html demo/tui/tui_session.log > demo/tui/tui_session.html
```

## Key Screenshots

| # | File | Subject |
|---|------|---------|
| 4a | `tui_dashboard.png` | TUI showing parameter importance table and observable selector |
| 4b | `tui_session.gif` | Optional: asciinema terminal recording |
| 4c | `tui_session.html` | Fallback: script(1) output rendered as HTML |
