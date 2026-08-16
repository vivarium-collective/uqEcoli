"""UQPC TUI — Textual terminal interface for ``uq.workflow``.

Launch:
    uv run uq tui
    uv run python -m uq.tui

ANSI-only colors via ``textual-ansi`` theme — adapts to any terminal.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
from pathlib import Path
from typing import Any

import numpy as np
from rich.syntax import Syntax
from rich.text import Text
from textual import work
from textual.app import App, ComposeResult
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.screen import ModalScreen
from textual.widgets import (
    Button,
    Checkbox,
    DataTable,
    Footer,
    Header,
    Input,
    Label,
    ProgressBar,
    RichLog,
    Select,
    Static,
    Switch,
    TabbedContent,
    TabPane,
)

from libuq.pipeline.models import SimDataParameter
from libuq.pipeline.param_loader import DEFAULT_SIM_DATA_PARAMETERS

# ── Constants ────────────────────────────────────────────────────────

BANNER = (
    "[bold cyan]"
    " ██╗   ██╗ ██████╗ ██████╗  ██████╗\n"
    " ██║   ██║██╔═══██╗██╔══██╗██╔════╝\n"
    " ██║   ██║██║   ██║██████╔╝██║     \n"
    " ██║   ██║██║▄▄ ██║██╔═══╝ ██║     \n"
    " ╚██████╔╝╚██████╔╝██║     ╚██████╗\n"
    "  ╚═════╝  ╚══▀▀═╝ ╚═╝      ╚═════╝"
    "[/bold cyan]"
)

REGRESSION_OPTIONS = [
    ("lsq  (Least Squares)", "lsq"),
    ("bcs  (Bayesian Compressed Sensing)", "bcs"),
    ("anl  (Analytical Bayesian)", "anl"),
]

_DEFAULT_PARAMS = list(DEFAULT_SIM_DATA_PARAMETERS)

DEFAULT_OBS = [
    "listeners__mass__dry_mass",
    "listeners__mass__cell_mass",
    "listeners__mass__volume",
    "listeners__mass__growth",
]

# Observable presets (cd1 analysis modules) — see uq/observables.py
try:
    from uq.observables import PRESETS as _OBS_PRESETS
except ImportError:
    _OBS_PRESETS = {}  # type: ignore[assignment]


def _json_markup(data: Any) -> Syntax:
    return Syntax(
        json.dumps(data, indent=2, default=str),
        "json",
        theme="native",
        word_wrap=True,
    )


# vEcoli workflow-config helpers live in uq/vecoli_config.py (CLI-owned).
# The TUI consumes them; no re-export — new callers should import from
# uq.vecoli_config directly.
from uq.vecoli_config import (
    _build_config,
    _build_variants_from_samples,
    _collect_variant_timeseries,
    _count_completed_variants,
    _get_vecoli_root,
)


# ── Modal ────────────────────────────────────────────────────────────


class TextInputScreen(ModalScreen[str | None]):
    BINDINGS = [("escape", "cancel", "Cancel")]
    CSS = """
    TextInputScreen { align: center middle; }
    #modal-container {
        width: 60; height: auto; max-height: 10;
        border: thick ansi_cyan; padding: 1 2;
    }
    #modal-container Input { margin-top: 1; }
    #modal-container .modal-buttons { margin-top: 1; height: 3; }
    """

    def __init__(self, prompt: str, placeholder: str = "", default: str = "") -> None:
        super().__init__()
        self.prompt = prompt
        self.placeholder = placeholder
        self.default = default

    def compose(self) -> ComposeResult:
        with Vertical(id="modal-container"):
            yield Label(self.prompt)
            yield Input(placeholder=self.placeholder, value=self.default, id="modal-input")
            with Horizontal(classes="modal-buttons"):
                yield Button("OK", variant="primary", id="modal-ok")
                yield Button("Cancel", id="modal-cancel")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "modal-ok":
            self._submit()
        else:
            self.dismiss(None)

    def on_input_submitted(self, _event: Input.Submitted) -> None:
        self._submit()

    def _submit(self) -> None:
        val = self.query_one("#modal-input", Input).value.strip()
        self.dismiss(val if val else None)

    def action_cancel(self) -> None:
        self.dismiss(None)


# ── Main App ─────────────────────────────────────────────────────────


class UQPCApp(App[None]):
    """UQPC workflow for vEcoli — RFC006 global sensitivity analysis."""

    TITLE = "UQPC"
    SUB_TITLE = "RFC006 · PyTUQ + vEcoli"
    BINDINGS = [
        ("q", "quit", "Quit"),
        ("d", "toggle_dark", "Theme"),
    ]

    CSS = """
    #sidebar {
        width: 38; border-right: solid ansi_bright_black; padding: 1;
    }
    #sidebar Button { width: 100%; }
    #sidebar Input { width: 100%; margin: 0 0 1 0; }
    #sidebar Select { width: 100%; margin: 0 0 1 0; }
    .nav-section {
        text-style: bold; color: ansi_cyan; margin: 1 0 0 0;
    }
    .cfg-label {
        color: ansi_yellow; margin: 0;
    }
    .param-row {
        height: auto; margin: 0;
    }
    .param-row Checkbox {
        width: 100%; color: ansi_cyan;
    }
    .bounds-row {
        height: 3; margin: 0 0 1 0;
    }
    .bounds-row Label {
        width: 3; color: ansi_bright_black;
    }
    .bounds-row Input {
        width: 1fr;
    }
    #main-content { padding: 1 2; }
    #result-log { height: 1fr; border: round ansi_bright_black; }
    DataTable { height: 1fr; border: round ansi_bright_black; }

    #progress-bar { margin: 0 2; }
    #progress-status { color: ansi_cyan; text-style: bold; padding: 0 2; }
    """

    def __init__(self) -> None:
        super().__init__()
        self.theme = "textual-ansi"
        self._result: Any = None
        self._sampling_cancel = threading.Event()

    def compose(self) -> ComposeResult:
        yield Header()
        with Horizontal():
            with VerticalScroll(id="sidebar"):
                yield Label("CONFIG", classes="nav-section")
                yield Label("simData", classes="cfg-label")
                yield Input(
                    value="/Users/alexanderpatrie/sms/uqEcoli/sim_data/baseline/kb/simData.cPickle",
                    id="cfg-simdata",
                )
                yield Label("Cache dir", classes="cfg-label")
                yield Input(value="./uq_cache", id="cfg-cache")
                yield Label("Samples", classes="cfg-label")
                yield Input(value="20", id="cfg-n-samples", type="integer")
                yield Label("Generations", classes="cfg-label")
                yield Input(value="1", id="cfg-gens", type="integer")
                yield Label("Seeds", classes="cfg-label")
                yield Input(value="1", id="cfg-seeds", type="integer")
                yield Label("PCE order", classes="cfg-label")
                yield Input(value="2", id="cfg-order", type="integer")
                yield Label("Growth bins", classes="cfg-label")
                yield Input(value="10", id="cfg-bins", type="integer")
                yield Label("Regression", classes="cfg-label")
                yield Select(REGRESSION_OPTIONS, value="lsq", id="cfg-reg")

                yield Label("Conditions", classes="cfg-label")
                yield Input(
                    value="",
                    id="cfg-conditions",
                    placeholder="comma-separated dataset IDs (multi-parca)",
                )

                yield Label("VARIANT PARAMETERS", classes="nav-section")
                for _p in _DEFAULT_PARAMS:
                    with Horizontal(classes="param-row"):
                        yield Checkbox(
                            _p.name,
                            value=True,
                            id=f"chk-{_p.name}",
                        )
                    with Horizontal(classes="bounds-row"):
                        yield Label("lo", classes="bounds-label")
                        yield Input(
                            value=str(_p.bounds[0]),
                            id=f"lo-{_p.name}",
                        )
                        yield Label("hi", classes="bounds-label")
                        yield Input(
                            value=str(_p.bounds[1]),
                            id=f"hi-{_p.name}",
                        )

                yield Label("SAMPLE (Steps 1-3)", classes="nav-section")
                yield Button("Run Sampling", id="run-sample", variant="success")
                yield Button("Cancel Sampling", id="cancel-sample", variant="error")

                yield Label("QUANTIFY (Steps 4-5)", classes="nav-section")
                yield Button("Run Quantify", id="run-quantify", variant="success")

                yield Label("RESULTS", classes="nav-section")
                yield Button("S1 Population", id="show-s1", variant="primary")
                yield Button("S2 By Generation", id="show-s2", variant="primary")
                yield Button("S3 By Seed", id="show-s3", variant="primary")
                yield Button("S4 Growth-Stratified", id="show-s4", variant="primary")

                yield Label("EXPORT", classes="nav-section")
                yield Button("Export All", id="export-all")

            with Vertical(id="main-content"):
                yield Static("", id="progress-status")
                yield ProgressBar(total=100, show_eta=True, id="progress-bar")
                yield self._build_tabs()

        yield Footer()

    @staticmethod
    def _build_tabs() -> TabbedContent:
        tabs = TabbedContent(id="tabs")
        log_pane = TabPane("Log", id="tab-log")
        log_pane.compose_add_child(RichLog(id="result-log", highlight=True, markup=True, wrap=True))
        table_pane = TabPane("Table", id="tab-table")
        table_pane.compose_add_child(DataTable(id="data-table"))
        tabs.compose_add_child(log_pane)
        tabs.compose_add_child(table_pane)
        return tabs

    def on_mount(self) -> None:
        self.query_one("#progress-bar", ProgressBar).display = False
        self.query_one("#progress-status", Static).update("")
        self.write_log(BANNER)
        self.write_log("[dim]RFC006 Global Sensitivity Analysis — PyTUQ UQPC Workflow[/dim]")
        self.write_log(
            "[dim]Sidebar: run workflow steps.  Config bar: set parameters.  "
            "[bold]q[/bold]=quit  [bold]d[/bold]=theme[/dim]\n"
        )

    # ── Helpers ──────────────────────────────────────────────────────

    def write_log(self, msg: str | Text | Syntax) -> None:
        self.query_one("#result-log", RichLog).write(msg)

    def _show_json(self, data: Any, title: str = "") -> None:
        if title:
            self.write_log(f"[bold cyan]{title}[/]")
        self.write_log(_json_markup(data))
        self.write_log("")

    def _populate_table(self, columns: list[str], rows: list[list[str]]) -> None:
        table = self.query_one("#data-table", DataTable)
        table.clear(columns=True)
        for col in columns:
            table.add_column(col)
        for row in rows:
            table.add_row(*row)
        self.query_one("#tabs", TabbedContent).active = "tab-table"

    def _switch_to_log(self) -> None:
        self.query_one("#tabs", TabbedContent).active = "tab-log"

    def _cfg(self, widget_id: str) -> str:
        return self.query_one(f"#{widget_id}", Input).value.strip()

    def _set_progress(self, current: int, total: int, label: str) -> None:
        bar = self.query_one("#progress-bar", ProgressBar)
        bar.display = True
        bar.update(total=total, progress=current)
        self.query_one("#progress-status", Static).update(f"[ansi_cyan]{label}[/]  [ansi_yellow]{current}/{total}[/]")

    def _hide_progress(self) -> None:
        self.query_one("#progress-bar", ProgressBar).display = False
        self.query_one("#progress-status", Static).update("")

    # ── Button dispatch ──────────────────────────────────────────────

    def on_button_pressed(self, event: Button.Pressed) -> None:
        bid = event.button.id
        if bid == "run-sample":
            self._do_sample()
        elif bid == "cancel-sample":
            self._sampling_cancel.set()
            self.write_log("[ansi_yellow]Cancellation requested...[/]")
        elif bid == "run-quantify":
            self._do_quantify()
        elif bid in ("show-s1", "show-s2", "show-s3", "show-s4"):
            self._show_strategy(int(bid[-1]))
        elif bid == "export-all":
            self._do_export()

    # ── Sampling: single workflow.py call + live progress ────────────

    @work(thread=True)
    def _do_sample(self) -> None:
        from libuq.pipeline.param_loader import ParameterDataset
        from uq.workflow import _setup_input_pc

        self._sampling_cancel.clear()
        sim_path = self._cfg("cfg-simdata")
        if not sim_path:
            self.call_from_thread(self.write_log, "[ansi_red]Set simData.cPickle path first[/]")
            return

        # Resolve to absolute paths — workflow.py runs with cwd=vecoli_root
        sim_path = str(Path(sim_path).resolve())
        cache_dir = Path(self._cfg("cfg-cache") or "./uq_cache").resolve()
        n_samples = int(self._cfg("cfg-n-samples") or "20")
        generations = int(self._cfg("cfg-gens") or "1")
        n_init_sims = int(self._cfg("cfg-seeds") or "1")

        self.call_from_thread(self._switch_to_log)
        self.call_from_thread(
            self.write_log,
            f"[bold cyan]SAMPLING[/] {n_samples} variants x {n_init_sims} seeds x {generations} gens",
        )

        # ── Step 1: Setup inputs ──
        self.call_from_thread(self.write_log, "[ansi_bright_black]Step 1: loading simData...[/]")

        # Read active parameters from sidebar checkboxes + bounds inputs
        active_params: list[SimDataParameter] = []
        for p in _DEFAULT_PARAMS:
            chk = self.query_one(f"#chk-{p.name}", Checkbox)
            if chk.value:
                lo_str = self.query_one(f"#lo-{p.name}", Input).value.strip()
                hi_str = self.query_one(f"#hi-{p.name}", Input).value.strip()
                lo = float(lo_str) if lo_str else p.bounds[0]
                hi = float(hi_str) if hi_str else p.bounds[1]
                active_params.append(
                    SimDataParameter(
                        name=p.name,
                        attr_path=p.attr_path,
                        bounds=(lo, hi),
                        description=p.description,
                    )
                )

        if not active_params:
            self.call_from_thread(self.write_log, "[ansi_red]No parameters selected — check at least one[/]")
            return

        try:
            ds = ParameterDataset(sim_data_path=sim_path)
            param_space = ds.to_parameter_space(parameters=active_params)
        except Exception as e:
            self.call_from_thread(self.write_log, f"[ansi_red]Failed to load simData: {e}[/]")
            return

        bounds = np.array(param_space.parameter_bounds)
        self.call_from_thread(
            self.write_log,
            f"[ansi_bright_black]  {param_space.n_parameters} params: {param_space.parameter_names}[/]",
        )
        for ap in active_params:
            self.call_from_thread(
                self.write_log,
                f"[ansi_bright_black]    {ap.name}: [{ap.bounds[0]}, {ap.bounds[1]}][/]",
            )

        # ── Step 2: Generate samples via PCRV ──
        self.call_from_thread(self.write_log, "[ansi_bright_black]Step 2: PCRV.sampleGerm()...[/]")
        input_pc, _, _ = _setup_input_pc(bounds)
        np.random.seed(42)
        germ_train = input_pc.sampleGerm(n_samples)
        X_train = input_pc.evalPC(germ_train)
        self.call_from_thread(
            self.write_log,
            f"[ansi_bright_black]  X shape: {X_train.shape} (physical space)[/]",
        )

        # ── Step 3: Build config + run workflow.py with live output ──
        self.call_from_thread(self.write_log, "[ansi_bright_black]Step 3: running vEcoli workflow.py...[/]")

        import shutil

        batch_dir = cache_dir / "_batch"
        # Clean stale output from previous runs — workflow.py refuses
        # to write into an existing experiment directory.
        if batch_dir.exists():
            shutil.rmtree(batch_dir)
        batch_dir.mkdir(parents=True, exist_ok=True)
        output_dir = batch_dir / "output"
        output_dir.mkdir(exist_ok=True)
        experiment_id = "uqpc_batch"

        # Use the user-provided simData path directly in the config.
        # This tells workflow.py to skip parca — it already has the
        # pre-computed simData pickle.
        variants = _build_variants_from_samples(X_train, param_space._sim_data_parameters)

        # Parse conditions from sidebar
        cond_str = self._cfg("cfg-conditions").strip()
        conditions = [c.strip() for c in cond_str.split(",") if c.strip()] if cond_str else None

        config = _build_config(
            sim_data_path=sim_path,
            output_dir=str(output_dir),
            variants_section=variants,
            experiment_id=experiment_id,
            n_init_sims=n_init_sims,
            generations=generations,
            conditions=conditions,
        )
        config_path = batch_dir / "workflow_config.json"
        config_path.write_text(json.dumps(config, indent=2))

        if conditions:
            cond_meta = {"conditions": conditions, "n_samples": n_samples}
            (cache_dir / "conditions.json").write_text(json.dumps(cond_meta))
            self.call_from_thread(
                self.write_log,
                f"[ansi_cyan]Multi-condition: {len(conditions)} parca_variants[/]",
            )

        self.call_from_thread(
            self.write_log,
            f"[ansi_bright_black]  Config: {config_path}[/]",
        )
        self.call_from_thread(
            self.write_log,
            f"[ansi_bright_black]  Output: {output_dir}[/]",
        )

        import re
        import time as _time

        # Total sims = (n_variants + baseline) * n_init_sims * generations
        total_sims = (n_samples + 1) * n_init_sims * generations
        self.call_from_thread(self._set_progress, 0, total_sims, "Launching Nextflow")

        history_base = output_dir / experiment_id / "history"

        vecoli_root = _get_vecoli_root()
        nf_temp = Path(vecoli_root) / "nextflow_temp" / experiment_id
        if nf_temp.exists():
            shutil.rmtree(nf_temp)

        workflow_script = os.path.join(vecoli_root, "runscripts", "workflow.py")
        cmd = [sys.executable, workflow_script, "--config", str(config_path)]
        env = os.environ.copy()
        existing = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = vecoli_root + (os.pathsep + existing if existing else "")
        env["PYTHONUNBUFFERED"] = "1"

        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            cwd=vecoli_root,
            env=env,
        )

        # Shared state — daemon threads write, main loop reads + updates UI
        poll_stop = threading.Event()
        start_time = _time.monotonic()
        phase = ["launching"]
        pq_count = [0]
        log_queue: list[tuple[str, str]] = []  # (message, style) pairs
        log_lock = threading.Lock()

        def _enqueue_log(msg: str, style: str = "") -> None:
            with log_lock:
                log_queue.append((msg, style))

        # ── Background: poll output directory for .pq files ──
        def _poll_outputs() -> None:
            while not poll_stop.is_set():
                n = _count_completed_variants(history_base)
                if n != pq_count[0]:
                    pq_count[0] = n
                poll_stop.wait(2.0)

        poll_thread = threading.Thread(target=_poll_outputs, daemon=True)
        poll_thread.start()

        # ── Background: read stdout chunks, enqueue log lines ──
        # Regex to strip ALL ANSI escape sequences (cursor movement,
        # colors, bold/reset, 256-color, truecolor, erase-line, etc.)
        _ansi_re = re.compile(r"\x1b\[[\d;]*[A-Za-z]|\x1b\[\d*[A-GJK]|\x07")

        def _read_stdout() -> None:
            if proc.stdout is None:
                return
            fd = proc.stdout.fileno()
            buf = b""
            while True:
                # os.read returns as soon as ANY bytes are available
                # (unlike file.read(N) which blocks until N bytes or EOF)
                try:
                    chunk = os.read(fd, 4096)
                except OSError:
                    break
                if not chunk:
                    break
                buf += chunk
                while b"\n" in buf or b"\r" in buf:
                    idx_n = buf.find(b"\n")
                    idx_r = buf.find(b"\r")
                    if idx_n == -1:
                        idx = idx_r
                    elif idx_r == -1:
                        idx = idx_n
                    else:
                        idx = min(idx_n, idx_r)
                    raw = buf[:idx].decode("utf-8", errors="replace").strip()
                    buf = buf[idx + 1 :]
                    if not raw:
                        continue
                    clean = _ansi_re.sub("", raw).strip()
                    if not clean:
                        continue

                    low = clean.lower()

                    # Detect phase transitions
                    if "createvariants" in low and phase[0] == "launching":
                        phase[0] = "variants"
                        _enqueue_log("Phase: creating variant sim_data pickles...", "ansi_cyan")
                    elif "sim" in low and ("gen" in low or "agent" in low) and phase[0] != "simulating":
                        phase[0] = "simulating"
                        _enqueue_log(f"Phase: simulating {total_sims} cells...", "ansi_cyan")

                    # Classify and enqueue — ALL lines shown, color by type
                    if any(kw in low for kw in ["error", "fail", "exception", "traceback"]):
                        _enqueue_log(clean, "ansi_red")
                    elif any(kw in low for kw in ["completed at", "duration", "succeeded"]):
                        _enqueue_log(clean, "ansi_green")
                    elif any(kw in low for kw in ["warn", "note"]):
                        _enqueue_log(clean, "ansi_yellow")
                    else:
                        # Show ALL other lines (Nextflow progress, executor, etc.)
                        _enqueue_log(clean, "ansi_bright_black")

        stdout_thread = threading.Thread(target=_read_stdout, daemon=True)
        stdout_thread.start()

        # ── Main @work loop: drain queues, update UI every second ──
        _last_prelim_count = [0]
        try:
            while proc.poll() is None:
                if self._sampling_cancel.is_set():
                    proc.terminate()
                    proc.wait(timeout=10)
                    self.call_from_thread(self.write_log, "[ansi_yellow]Sampling cancelled[/]")
                    poll_stop.set()
                    self.call_from_thread(self._hide_progress)
                    return

                # Drain log queue
                with log_lock:
                    pending = list(log_queue)
                    log_queue.clear()
                for msg, style in pending:
                    if style:
                        self.call_from_thread(self.write_log, f"[{style}]  {msg}[/{style}]")
                    else:
                        self.call_from_thread(self.write_log, f"  {msg}")

                # Update progress bar
                elapsed = int(_time.monotonic() - start_time)
                label = phase[0].capitalize()
                self.call_from_thread(
                    self._set_progress,
                    pq_count[0],
                    total_sims,
                    f"{label}  {pq_count[0]}/{total_sims}  [{elapsed}s]",
                )

                # Preliminary Sobol every 5 completed variants
                _cur = pq_count[0]
                if _cur >= 5 and _cur - _last_prelim_count[0] >= 5:
                    _last_prelim_count[0] = _cur
                    self._show_preliminary_sobol(
                        history_base, _cur, X_train, germ_train, param_space,
                    )

                threading.Event().wait(1.0)

            # Process finished — drain remaining logs
            with log_lock:
                pending = list(log_queue)
                log_queue.clear()
            for msg, style in pending:
                if style:
                    self.call_from_thread(self.write_log, f"[{style}]  {msg}[/{style}]")
                else:
                    self.call_from_thread(self.write_log, f"  {msg}")
        finally:
            poll_stop.set()
            stdout_thread.join(timeout=10)
            poll_thread.join(timeout=5)

        exit_code = proc.returncode
        if exit_code != 0 and not self._sampling_cancel.is_set():
            self.call_from_thread(
                self.write_log,
                f"[ansi_yellow]workflow.py exited with code {exit_code} (some variants may have failed)[/]",
            )

        # ── Step 4: Collect + preprocess ──
        self.call_from_thread(self._set_progress, total_sims, total_sims, "Collecting outputs")
        self.call_from_thread(self.write_log, "[ansi_bright_black]Collecting Parquet outputs...[/]")

        try:
            # Find history_base (workflow.py may nest differently)
            if not history_base.exists():
                candidates = list(output_dir.glob("*/history"))
                if candidates:
                    history_base = candidates[0]

            Y_agg, Y_ts, Y_meta = _collect_variant_timeseries(
                history_base,
                n_samples,
                DEFAULT_OBS,
            )
        except Exception as e:
            self.call_from_thread(self.write_log, f"[ansi_red]Collection failed: {e}[/]")
            self.call_from_thread(self._hide_progress)
            return

        # ── Save cache ──
        from libuq.sampling import PrecomputedCache as PC

        cache_dir.mkdir(parents=True, exist_ok=True)
        cache = PC(
            cache_dir=cache_dir,
            X=X_train,
            Y=Y_agg,
            parameter_names=param_space.parameter_names,
            metadata={
                "bounds": bounds.tolist(),
                "seed": 42,
                "observable_columns": DEFAULT_OBS,
            },
            Y_timeseries=Y_ts,
            Y_timeseries_meta=Y_meta,
        )
        cache.save()
        np.save(cache_dir / "germ_train.npy", germ_train)

        self.call_from_thread(self._hide_progress)
        self.call_from_thread(
            self.write_log,
            f"[ansi_green]Cached {Y_agg.shape[0]} samples ({Y_agg.shape[1]} obs) to {cache_dir}[/]",
        )
        if Y_ts:
            self.call_from_thread(
                self.write_log,
                f"[ansi_bright_black]  Timeseries: {len(Y_ts)} samples, shape {Y_ts[0].shape}[/]",
            )
        self.call_from_thread(self.write_log, "")

    def _show_preliminary_sobol(
        self,
        history_base: Path,
        n_completed: int,
        X_train: np.ndarray,
        germ_train: np.ndarray,
        param_space: Any,
    ) -> None:
        """Fit a preliminary PCE on completed samples and show live Sobol."""
        if n_completed < 5:
            return
        try:
            Y_agg, _, _ = _collect_variant_timeseries(
                history_base, min(n_completed, X_train.shape[0]), DEFAULT_OBS,
            )
            k = min(n_completed, Y_agg.shape[0], X_train.shape[0])
            if k < 3:
                return

            from uq.workflow import _compute_sobol, _fit_surrogate

            germ_k = germ_train[:k]
            Y_k = Y_agg[:k]
            pcrv, _ = _fit_surrogate(germ_k, Y_k, polynomial_order=2, regression="lsq")
            sobol = _compute_sobol(pcrv, param_space.parameter_names, Y_k)

            lines = [f"[ansi_cyan]── Preliminary Sobol ({k} samples) ──[/]"]
            for i, nm in enumerate(param_space.parameter_names):
                st = sobol.total_order[i]
                bar = "█" * int(st * 20)
                lines.append(f"  {nm:<35s} {st:.3f} {bar}")

            for line in lines:
                self.call_from_thread(self.write_log, line)
        except Exception:
            pass  # Don't let preliminary analysis break sampling

    # ── Quantify ─────────────────────────────────────────────────────

    @work(thread=True)
    def _do_quantify(self) -> None:
        from uq.multi_condition import is_multi_condition_cache, quantify_multi_condition
        from uq.workflow import quantify

        cache_dir = self._cfg("cfg-cache") or "./uq_cache"
        sim_path = self._cfg("cfg-simdata")
        if not sim_path:
            self.call_from_thread(self.write_log, "[ansi_red]Set simData path first[/]")
            return

        order = int(self._cfg("cfg-order") or "2")
        bins = int(self._cfg("cfg-bins") or "10")
        reg = self.query_one("#cfg-reg", Select).value

        self.call_from_thread(self._switch_to_log)
        self.call_from_thread(self._set_progress, 0, 4, "Quantify: Strategy 1")
        self.call_from_thread(
            self.write_log,
            f"[bold cyan]QUANTIFY[/] order={order} bins={bins} reg={reg}",
        )

        # Auto-detect multi-condition cache
        if is_multi_condition_cache(cache_dir):
            self.call_from_thread(
                self.write_log,
                "[ansi_magenta]Multi-condition cache detected — cross-condition GSA[/]",
            )
            try:
                mc_result = quantify_multi_condition(
                    cache_dir=cache_dir,
                    sim_data_path=sim_path,
                    polynomial_order=order,
                    n_bins=bins,
                    regression=reg,
                )
                self.call_from_thread(self._set_progress, 4, 4, "Done")
                for cond_id, cr in mc_result.per_condition.items():
                    self.call_from_thread(self.write_log, f"\n[ansi_cyan]── {cond_id} ──[/]")
                    for i, nm in enumerate(mc_result.parameter_names):
                        st = cr.strategy1.sobol.total_order[i]
                        bar = "█" * int(st * 30)
                        self.call_from_thread(self.write_log, f"  {nm:<35s} {st:.4f} {bar}")

                if mc_result.universal_drivers:
                    self.call_from_thread(
                        self.write_log,
                        f"\n[ansi_green]Universal drivers: {', '.join(mc_result.universal_drivers)}[/]",
                    )
                for cond_id, params in mc_result.condition_specific.items():
                    self.call_from_thread(
                        self.write_log,
                        f"[ansi_yellow]Condition-specific ({cond_id}): {', '.join(params)}[/]",
                    )
                self.call_from_thread(self._hide_progress)
            except Exception as e:
                self.call_from_thread(self.write_log, f"[ansi_red]Error: {e}[/]\n")
                self.call_from_thread(self._hide_progress)
            return

        try:
            result = quantify(
                cache_dir=cache_dir,
                sim_data_path=sim_path,
                polynomial_order=order,
                n_bins=bins,
                regression=reg,  # type: ignore[arg-type]
            )
            self._result = result

            self.call_from_thread(self._set_progress, 4, 4, "Done")
            self.call_from_thread(
                self.write_log,
                f"[ansi_green]Done: {len(result.parameter_names)} params, "
                f"S2={len(result.strategy2)} gens, "
                f"S3={len(result.strategy3)} seeds, "
                f"S4={len(result.strategy4_per_stage)} stages[/]",
            )

            # Inline S1 summary
            self.call_from_thread(self.write_log, "\n[bold cyan]Strategy 1 — Population[/]")
            s1 = result.strategy1
            for i, nm in enumerate(result.parameter_names):
                st = s1.sobol.total_order[i]
                bar = "\u2588" * int(st * 30)
                self.call_from_thread(self.write_log, f"  {nm:<35s} {st:.4f} {bar}")
            self.call_from_thread(
                self.write_log,
                f"[ansi_bright_black]  relerr: {s1.relerr_train.tolist()}[/]\n",
            )
            self.call_from_thread(self._hide_progress)

        except Exception as e:
            self.call_from_thread(self.write_log, f"[ansi_red]Error: {e}[/]\n")
            self.call_from_thread(self._hide_progress)

    # ── Show strategy results ────────────────────────────────────────

    def _show_strategy(self, n: int) -> None:
        if self._result is None:
            self.write_log("[ansi_yellow]Run quantify first[/]")
            return

        r = self._result
        names = r.parameter_names

        if n == 1:
            self._populate_table(
                ["Parameter", "S1", "ST", ""],
                [
                    [
                        nm,
                        f"{r.strategy1.sobol.first_order[i]:.4f}",
                        f"{r.strategy1.sobol.total_order[i]:.4f}",
                        "\u2588" * int(r.strategy1.sobol.total_order[i] * 40),
                    ]
                    for i, nm in enumerate(names)
                ],
            )
        elif n == 2:
            if not r.strategy2:
                self.write_log("[ansi_yellow]No generation metadata[/]")
                return
            cols = ["Parameter"] + [f"Gen {g}" for g in sorted(r.strategy2)]
            rows = [
                [nm] + [f"{r.strategy2[g].sobol.total_order[i]:.4f}" for g in sorted(r.strategy2)]
                for i, nm in enumerate(names)
            ]
            self._populate_table(cols, rows)
        elif n == 3:
            if not r.strategy3:
                self.write_log("[ansi_yellow]No seed metadata[/]")
                return
            cols = ["Parameter"] + [f"Seed {s}" for s in sorted(r.strategy3)]
            rows = [
                [nm] + [f"{r.strategy3[s].sobol.total_order[i]:.4f}" for s in sorted(r.strategy3)]
                for i, nm in enumerate(names)
            ]
            self._populate_table(cols, rows)
        elif n == 4:
            if not r.strategy4_per_stage:
                self.write_log("[ansi_yellow]No timeseries in cache[/]")
                return
            nb = len(r.strategy4_per_stage)
            cols = ["Parameter"] + [f"{j / nb:.0%}-{(j + 1) / nb:.0%}" for j in range(nb)]
            rows = [
                [nm] + [f"{r.strategy4_per_stage[j].sobol.total_order[i]:.4f}" for j in range(nb)]
                for i, nm in enumerate(names)
            ]
            self._populate_table(cols, rows)

        self.write_log(f"[ansi_green]Strategy {n} → Table tab[/]")

    # ── Export ───────────────────────────────────────────────────────

    @work(thread=True)
    def _do_export(self) -> None:
        if self._result is None:
            self.call_from_thread(self.write_log, "[ansi_yellow]Run quantify first[/]")
            return
        export_path = Path("./uq_results")
        self.call_from_thread(self.write_log, f"[ansi_cyan]Exporting to {export_path}...[/]")
        try:
            self._result.export(export_path)
            self.call_from_thread(self.write_log, f"[ansi_green]Exported to {export_path}[/]")
            self.call_from_thread(
                self._show_json,
                json.loads((export_path / "uq_results.json").read_text()),
                "uq_results.json",
            )
        except Exception as e:
            self.call_from_thread(self.write_log, f"[ansi_red]Error: {e}[/]\n")

    # ── Theme toggle ─────────────────────────────────────────────────

    def action_toggle_dark(self) -> None:
        cycle = ["textual-ansi", "textual-dark", "textual-light"]
        idx = cycle.index(self.theme) if self.theme in cycle else 0
        self.theme = cycle[(idx + 1) % len(cycle)]
        self.write_log(f"[dim]Theme: {self.theme}[/dim]")


def main() -> None:
    UQPCApp().run()


if __name__ == "__main__":
    main()
