"""
uq CLI — scientifically transparent UQ for vEcoli.

Three commands:

    uv run uq sample ...     # Stage 1: generate + cache samples (PCRV germ sampling)
    uv run uq quantify ...   # Stage 2: all 4 RFC006 strategies via PyTUQ PCE
    uv run uq dashboard ...  # Interactive visualization (tk or marimo)

The ``sample`` command is identical to ``uq sample`` — uses PCRV.sampleGerm()
(PyTUQ-native random sampling), subprocess-based vEcoli execution, and
PrecomputedCache.  Also caches
per-row generation/lineage_seed metadata for strategies 2-3.

The ``quantify`` command fits PCE surrogates via ``pytuq.surrogates.pce.PCE``
(following the UQPC workflow) and computes Sobol indices across all 4
RFC006 aggregation strategies:
    1. Uniform (bulk) — population-averaged sensitivity
    2. By generation — controls for convergence toward steady-state
    3. By lineage seed — controls for stochastic variance
    4. Growth-stratified — θ = normalized log(dry_mass), no spectral decomposition

Regression backends (--regression): lsq (default), bcs (sparse), anl (analytical).
"""

from __future__ import annotations

from enum import StrEnum
from pathlib import Path
from typing import Any, cast

import numpy as np
import typer
from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from uq.models import CliType

console = Console()
app = typer.Typer(
    help=(
        "Scientifically transparent UQ for vEcoli.\n\n"
        "Tip: every subcommand accepts a trailing ``help`` word as an alias "
        "for ``--help``, e.g. ``uq sample help``, ``uq quantify help``."
    ),
)


# ── sample: reuse uq sample directly ────────────────────────────────


@app.command()
def sample(
    sim_data_path: str = typer.Argument(..., help="Path to simData.cPickle"),
    cache_dir: str = typer.Option("./uq_cache", help="Cache output directory"),
    n_samples: int = 20,
    n_test: int = typer.Option(
        0,
        help="Extra held-out validation samples (UQPC --ntst). Evaluated via vEcoli "
        "alongside training samples; used by `quantify` to compute test relative errors.",
    ),
    seed: int = 42,
    generations: int = 1,
    n_init_sims: int = 1,
    max_duration: float = 10800.0,
    params_file: str | None = None,
    observables: list[str] = typer.Option(
        ["mass"],
        help="Observable presets to extract (cd1 analysis modules). "
        "Options: mass, higher_order, exchange_fluxes, transcriptome, "
        "proteome, fluxome. Pass multiple: --observables higher_order "
        "--observables exchange_fluxes",
    ),
    generation_lower_bound: int = typer.Option(
        0,
        help="Skip generations below this value when aggregating Y "
        "(cd1 generation_lower_bound). 0 = keep all.",
    ),
    base_config: str | None = typer.Option(
        None,
        help="Base vEcoli config JSON to merge with. Preserves parca_variants, "
        "analysis_options, and other multi-parca keys.",
    ),
    conditions: list[str] = typer.Option(
        [],
        help="RNA-seq dataset IDs for multi-condition UQ (one per --conditions). "
        "Populates parca_variants for cross-condition sensitivity analysis. "
        "Requires vEcoli multi-parca-aws branch.",
    ),
    api_url: str | None = typer.Option(
        None,
        help="SMS-API base URL for remote execution (e.g. https://sms.cam.uchc.edu). "
        "When set, simulations run on the SMS-API cluster instead of local vEcoli.",
    ),
    simulator_id: int | None = typer.Option(
        None,
        help="SMS-API simulator database_id (required with --api-url). "
        "Use `uq remote discover --api-url URL` to list available simulators.",
    ),
    config_filename: str = typer.Option(
        "api_simulation_default.json",
        help="vEcoli config filename on the server (used with --api-url). "
        "Use GET /simulations/discovery to list available files.",
    ),
    ecoli_sources_repo: str | None = typer.Option(
        None,
        help="GitHub URL for ecoli-sources data repo (used with --api-url). "
        "Server downloads and syncs to S3 automatically.",
    ),
    ecoli_sources_ref: str | None = typer.Option(
        None,
        help="Git ref for ecoli-sources repo (default: main).",
    ),
    backend: str = typer.Option(
        "vecoli",
        help="Simulation backend: v2ecoli (in-process via process-bigraph composite, "
        "preferred) or vecoli (subprocess via workflow.py, legacy).",
    ),
    max_workers: int | None = typer.Option(
        None,
        help="Max parallel workers for v2ecoli backend (default: 1). "
        "Ignored for vecoli (Nextflow handles parallelism).",
    ),
    variants_source: str = typer.Option(
        "params-file",
        help="Where vEcoli `variants` come from. "
        "`params-file` (default): PCRV-sample from --params-file bounds, "
        "encode as sim_data_setattr. "
        "`base-config`: trust the `variants` block in --base-config verbatim, "
        "reverse-map mutation values into X for `quantify` "
        "(local --backend vecoli only, requires sim_data_setattr module).",
    ),
) -> None:
    """UQPC Steps 1-3: sample via PCRV.sampleGerm(), run vEcoli.

    Three execution modes:
      LOCAL (default): subprocess vEcoli via runscripts/workflow.py
      V2ECOLI (--backend v2ecoli): in-process v2ecoli composite
      REMOTE (--api-url): submit to SMS-API, poll, download cd1 analysis TSVs

    \b
    Use --generations >= 2 to enable Strategy 2 (by-generation GSA).
    Use --n-test > 0 for held-out validation (UQPC --ntst).
    Use --observables to select cd1-style observable categories:
      mass             Raw mass/growth scalars (default, local only)
      higher_order     Derived metrics: doubling time, biomass composition
      exchange_fluxes  External metabolite fluxes (~87)
      transcriptome    mRNA cistron counts (~4300 genes)
      proteome         Protein monomer counts (~4300 monomers)
      fluxome          Base reaction fluxes (~2800, dry-mass normalized)
    Use --generation-lower-bound N to skip early generations.
    Use --api-url http://localhost:8080 --simulator-id 11 for remote execution.
    Use --backend v2ecoli for in-process v2ecoli composite execution
    (replaces subprocess + Nextflow).
    """
    import json as _json
    import shutil

    from libuq.pipeline.models import SimDataParameter
    from libuq.pipeline.param_loader import DEFAULT_SIM_DATA_PARAMETERS, ParameterDataset
    from libuq.sampling import PrecomputedCache
    from uq.vecoli_config import (
        _check_oob_variants,
        _design_matrix_conditioning,
        _germ_from_physical,
        _load_variants_from_base_config,
        _pce_sample_adequacy,
        _x_from_variants,
    )
    from uq.workflow import _setup_input_pc

    # ── Validate variants-source compatibility ──
    if variants_source not in ("params-file", "base-config"):
        console.print(
            f"[red]Unknown --variants-source '{variants_source}'. "
            "Use 'params-file' or 'base-config'.[/red]"
        )
        raise typer.Exit(1)

    is_remote = api_url is not None
    is_byo = variants_source == "base-config"

    if is_byo:
        # Local mode only for now — the BYO reverse-mapping path is unverified
        # against remote (X_all isn't pushed to SMS-API yet, todo §25-C) and
        # v2ecoli (different cache-bundle codepath).
        if is_remote:
            console.print(
                "[red]--variants-source base-config is not supported with "
                "--api-url. Remote mutation pushdown is todo §25-C.[/red]"
            )
            raise typer.Exit(1)
        if backend == "v2ecoli":
            console.print(
                "[red]--variants-source base-config is not supported with "
                "--backend v2ecoli yet. Use --backend vecoli.[/red]"
            )
            raise typer.Exit(1)
        if base_config is None:
            console.print(
                "[red]--variants-source base-config requires --base-config "
                "<PATH> to a vEcoli config JSON with a `variants` block.[/red]"
            )
            raise typer.Exit(1)
        if n_test > 0:
            console.print(
                "[yellow]--n-test is ignored under --variants-source "
                "base-config; held-out validation requires PCRV sampling.[/yellow]"
            )
            n_test = 0

    # ── Step 1: Setup inputs ──
    sim_data_path = str(Path(sim_data_path).resolve())
    cache_path = Path(cache_dir).resolve()

    mode_label = (
        f"[bold magenta]REMOTE → {api_url}[/bold magenta]" if is_remote
        else "[bold yellow]LOCAL · BYO variants[/bold yellow]" if is_byo
        else "[dim]LOCAL[/dim]"
    )
    console.print(f"[bold cyan]Sampling:[/bold cyan] {n_samples} variants, {n_init_sims} seeds, {generations} gens  {mode_label}")
    console.print(f"  [dim]simData: {sim_data_path}[/dim]")
    console.print(f"  [dim]cache:   {cache_path}[/dim]")

    console.print("[dim]Step 1: loading simData...[/dim]")
    ds = ParameterDataset(sim_data_path=sim_data_path)
    if params_file is not None:
        raw = _json.loads(Path(params_file).read_text())
        parameters = [SimDataParameter.from_dict(p) for p in raw]
    else:
        parameters = None
    param_space = ds.to_parameter_space(parameters=parameters)
    bounds = np.array(param_space.parameter_bounds)
    console.print(f"  [dim]{param_space.n_parameters} params: {param_space.parameter_names}[/dim]")

    germ_test: np.ndarray | None = None
    X_test: np.ndarray | None = None

    if is_byo:
        # ── Step 2 (BYO): reverse-map variants block → X ──
        console.print(f"[dim]Step 2: reading variants from {base_config}...[/dim]")
        variants_block = _load_variants_from_base_config(base_config)  # type: ignore[arg-type]
        X_train = _x_from_variants(variants_block, param_space._sim_data_parameters)
        n_samples = X_train.shape[0]
        germ_train = _germ_from_physical(X_train, bounds)
        console.print(f"  [dim]X shape: {X_train.shape}  (n_samples forced to {n_samples})[/dim]")

        # PCE basis-size diagnostic — n_samples is forced by the user's
        # variants list, so check it against the default quantify order (2).
        # Users who pass a different --polynomial-order to quantify can
        # ignore this if they're planning to drop to order 1.
        status, advisory = _pce_sample_adequacy(
            n_samples, param_space.n_parameters, order=2,
        )
        if status == "underdetermined":
            console.print(f"  [red]PCE adequacy: {advisory}[/red]")
        elif status == "marginal":
            console.print(f"  [yellow]PCE adequacy: {advisory}[/yellow]")
        else:
            console.print(f"  [dim]PCE adequacy: {advisory}[/dim]")

        # Design-matrix conditioning — PyTUQ-native joint design-quality
        # measure (subsumes marginal uniformity). Skip when count is
        # underdetermined (κ will be ∞ as a tautology from N < B).
        if status != "underdetermined":
            kappa, cond_status, cond_advisory = _design_matrix_conditioning(
                germ_train, param_space.n_parameters, order=2,
            )
            if cond_status in ("ill-conditioned", "singular"):
                console.print(f"  [red]Design κ(A): {cond_advisory}[/red]")
            elif cond_status == "marginal":
                console.print(f"  [yellow]Design κ(A): {cond_advisory}[/yellow]")
            else:
                console.print(f"  [dim]Design κ(A): {cond_advisory}[/dim]")
    else:
        # ── Step 2 (PCRV): sample germ → physical via PCRV ──
        console.print("[dim]Step 2: PCRV.sampleGerm()...[/dim]")
        input_pc, _, _ = _setup_input_pc(bounds)
        np.random.seed(seed)
        germ_train = input_pc.sampleGerm(n_samples)
        X_train = input_pc.evalPC(germ_train)
        console.print(f"  [dim]X shape: {X_train.shape}[/dim]")

        # UQPC ``--ntst``: draw additional held-out validation samples
        if n_test > 0:
            np.random.seed(seed + 1)
            germ_test = input_pc.sampleGerm(n_test)
            X_test = input_pc.evalPC(germ_test)
            console.print(f"  [dim]X_test shape: {X_test.shape} (held-out validation)[/dim]")

    # Concatenate train + test; vEcoli evaluates all variants in one workflow.
    X_all = np.vstack([X_train, X_test]) if X_test is not None else X_train

    # ── PyTUQ pdom invariant: every (variant, parameter) cell within bounds ──
    # PyTUQ treats pdom as a hard constraint; vEcoli's workflow.py fails
    # loudly on invalid configs. So we fail loudly here too — no soft
    # warning, no auto-bounds, no escape hatch. Runs in both modes; under
    # default PCRV sampling this is a no-op (evalPC stays within bounds).
    violations = _check_oob_variants(X_all, bounds, param_space.parameter_names)
    if violations:
        n = len(violations)
        plural = "s" if n > 1 else ""
        console.print(
            f"[red]Error: {n} variant/parameter cell{plural} violate "
            "--params-file bounds:[/red]"
        )
        for vi, name, v, lb, ub in violations[:20]:
            console.print(f"  [red]variant {vi}, {name} = {v:g} not in [{lb:g}, {ub:g}][/red]")
        if n > 20:
            console.print(f"  [red]... and {n - 20} more[/red]")
        console.print(
            "[red]Fix: widen the bounds in your --params-file (or omit "
            "--params-file to use DEFAULT_SIM_DATA_PARAMETERS).[/red]"
        )
        raise typer.Exit(1)

    # ── Step 3: Execute ──
    if is_remote:
        _sample_remote(
            api_url=api_url,  # type: ignore[arg-type]
            simulator_id=simulator_id,
            config_filename=config_filename,
            ecoli_sources_repo=ecoli_sources_repo,
            ecoli_sources_ref=ecoli_sources_ref,
            X_all=X_all,
            param_space=param_space,
            n_samples=n_samples,
            n_test=n_test,
            n_init_sims=n_init_sims,
            generations=generations,
            observables=observables,
            generation_lower_bound=generation_lower_bound,
            cache_path=cache_path,
            X_train=X_train,
            X_test=X_test,
            germ_train=germ_train,
            germ_test=germ_test,
            bounds=bounds,
            seed=seed,
            conditions=conditions,
        )
        return

    if backend == "v2ecoli":
        _sample_v2ecoli(
            sim_data_path=sim_data_path,
            cache_path=cache_path,
            X_all=X_all,
            param_space=param_space,
            n_samples=n_samples,
            n_test=n_test,
            generations=generations,
            max_duration=max_duration,
            observables=observables,
            generation_lower_bound=generation_lower_bound,
            X_train=X_train,
            X_test=X_test,
            germ_train=germ_train,
            germ_test=germ_test,
            bounds=bounds,
            seed=seed,
            max_workers=max_workers,
        )
        return

    _sample_local(
        sim_data_path=sim_data_path,
        cache_path=cache_path,
        X_all=X_all,
        param_space=param_space,
        n_samples=n_samples,
        n_test=n_test,
        n_init_sims=n_init_sims,
        generations=generations,
        max_duration=max_duration,
        observables=observables,
        generation_lower_bound=generation_lower_bound,
        base_config=base_config,
        conditions=conditions,
        X_train=X_train,
        X_test=X_test,
        germ_train=germ_train,
        germ_test=germ_test,
        bounds=bounds,
        seed=seed,
    )


def _sample_remote(
    *,
    api_url: str,
    simulator_id: int | None,
    config_filename: str,
    ecoli_sources_repo: str | None,
    ecoli_sources_ref: str | None,
    X_all: np.ndarray,
    param_space: Any,
    n_samples: int,
    n_test: int,
    n_init_sims: int,
    generations: int,
    observables: list[str],
    generation_lower_bound: int,
    cache_path: Path,
    X_train: np.ndarray,
    X_test: np.ndarray | None,
    germ_train: np.ndarray,
    germ_test: np.ndarray | None,
    bounds: np.ndarray,
    seed: int,
    conditions: list[str],
) -> None:
    """Step 3 (remote): submit to SMS-API, poll, download cd1 TSVs, cache."""
    import json as _json

    from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn

    from libuq.sampling import PrecomputedCache
    from uq.remote import (
        DEFAULT_CD1_ANALYSIS_OPTIONS,
        SmsApiClient,
        parse_cd1_tsvs,
        parse_cd1_tsvs_multi_variant,
    )

    if simulator_id is None:
        console.print("[red]--simulator-id is required with --api-url[/red]")
        raise typer.Exit(1)

    # Map UQ observable presets to cd1 module names for analysis_options
    cd1_presets = [p for p in observables if p in ("higher_order", "transcriptome", "proteome", "fluxome", "exchange_fluxes")]
    if not cd1_presets:
        # Default: use all cd1 modules
        cd1_presets = ["higher_order", "transcriptome", "proteome", "fluxome", "exchange_fluxes"]

    gen_lb = generation_lower_bound if generation_lower_bound > 0 else 5
    from uq.remote import CD1_MODULE_MAP

    analysis_options: dict[str, Any] = {
        "multiseed": {
            CD1_MODULE_MAP[preset]["module"]: {"generation_lower_bound": gen_lb}
            for preset in cd1_presets
            if preset in CD1_MODULE_MAP
        }
    }

    n_variants = X_all.shape[0]
    console.print(f"[dim]Step 3: submitting {n_variants} simulations to SMS-API at {api_url}...[/dim]")

    with SmsApiClient(base_url=api_url) as client:
        # Submit one simulation per variant sample
        sim_ids: list[int] = []
        for i in range(n_variants):
            exp_id = f"uq-sample-{i}"
            desc = f"UQ variant {i}/{n_variants}"
            sim = client.submit_simulation(
                simulator_id=simulator_id,
                experiment_id=exp_id,
                config_filename=config_filename,
                num_generations=generations,
                num_seeds=n_init_sims,
                description=desc,
                run_parca=True,
                ecoli_sources_repo_url=ecoli_sources_repo,
                ecoli_sources_ref=ecoli_sources_ref,
                analysis_options=analysis_options,
            )
            sim_ids.append(sim["database_id"])
            console.print(f"  [dim]Submitted variant {i} → sim {sim['database_id']}[/dim]")

        # Poll all simulations
        console.print(f"[dim]Polling {len(sim_ids)} simulations...[/dim]")
        with Progress(
            SpinnerColumn("dots", style="bold magenta"),
            TextColumn("[bold cyan]{task.description}"),
            TimeElapsedColumn(),
            console=console,
        ) as progress:
            task = progress.add_task(f"Waiting for {len(sim_ids)} simulations", total=len(sim_ids))

            def _on_status(sim_id: int, status: dict[str, Any]) -> None:
                current = status.get("status", "")
                if current.lower() in {"completed", "failed", "cancelled"}:
                    progress.advance(task)

            client.poll_batch(sim_ids, on_status=_on_status)

        # Download and parse cd1 TSVs
        console.print("[dim]Step 4: downloading cd1 analysis outputs...[/dim]")
        Y_rows: list[np.ndarray] = []
        obs_names: list[str] | None = None
        dl_dir = cache_path / "_remote"
        dl_dir.mkdir(parents=True, exist_ok=True)

        for i, sim_id in enumerate(sim_ids):
            dest = dl_dir / f"variant_{i}"
            output_dir = client.download_data(sim_id, dest=dest)
            y_row, names = parse_cd1_tsvs(output_dir, presets=cd1_presets)
            if obs_names is None:
                obs_names = names
            Y_rows.append(y_row)
            console.print(f"  [dim]variant {i}: {len(y_row)} observables from sim {sim_id}[/dim]")

    if not Y_rows:
        console.print("[red]No data collected from remote simulations.[/red]")
        raise typer.Exit(1)

    Y_all_arr = np.vstack(Y_rows)
    Y_agg = Y_all_arr[:n_samples]
    Y_test_arr = Y_all_arr[n_samples:] if n_test > 0 else None

    cache_path.mkdir(parents=True, exist_ok=True)
    cache = PrecomputedCache(
        cache_dir=cache_path,
        X=X_train,
        Y=Y_agg,
        parameter_names=param_space.parameter_names,
        metadata={
            "bounds": bounds.tolist(),
            "seed": seed,
            "observable_columns": obs_names or [],
            "source": "sms-api",
            "backend": "remote",
            "api_url": api_url,
            "simulator_id": simulator_id,
        },
        X_test=X_test,
        Y_test=Y_test_arr,
    )
    cache.save()
    np.save(cache_path / "germ_train.npy", germ_train)
    if germ_test is not None:
        np.save(cache_path / "germ_test.npy", germ_test)

    console.print(
        f"[bold green]Cached {Y_agg.shape[0]} training samples[/bold green] "
        f"({Y_agg.shape[1]} cd1 observables) to [cyan]{cache_path}[/cyan]"
    )
    if Y_test_arr is not None:
        console.print(f"  [dim]Held-out validation: {Y_test_arr.shape[0]} samples[/dim]")


def _sample_local(
    *,
    sim_data_path: str,
    cache_path: Path,
    X_all: np.ndarray,
    param_space: Any,
    n_samples: int,
    n_test: int,
    n_init_sims: int,
    generations: int,
    max_duration: float,
    observables: list[str],
    generation_lower_bound: int,
    base_config: str | None,
    conditions: list[str],
    X_train: np.ndarray,
    X_test: np.ndarray | None,
    germ_train: np.ndarray,
    germ_test: np.ndarray | None,
    bounds: np.ndarray,
    seed: int,
) -> None:
    """Step 3 (local): subprocess vEcoli via runscripts/workflow.py."""
    import json as _json
    import os
    import re
    import shutil
    import subprocess
    import sys
    import time as _time

    from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn

    from libuq.sampling import PrecomputedCache
    from uq.vecoli_config import (
        _build_config,
        _build_variants_from_samples,
        _count_completed_variants,
        _get_vecoli_root,
    )

    batch_dir = cache_path / "_batch"
    if batch_dir.exists():
        shutil.rmtree(batch_dir)
    batch_dir.mkdir(parents=True, exist_ok=True)
    output_dir = batch_dir / "output"
    output_dir.mkdir(exist_ok=True)
    experiment_id = "uqpc_batch"

    variants = _build_variants_from_samples(X_all, param_space._sim_data_parameters)
    config = _build_config(
        sim_data_path=sim_data_path,
        output_dir=str(output_dir),
        variants_section=variants,
        experiment_id=experiment_id,
        n_init_sims=n_init_sims,
        generations=generations,
        max_duration=max_duration,
        base_config_path=base_config,
        conditions=conditions if conditions else None,
    )
    config_path = batch_dir / "workflow_config.json"
    config_path.write_text(_json.dumps(config, indent=2))
    console.print(f"  [dim]Config JSON: {config_path}[/dim]")
    if conditions:
        console.print(f"  [dim]Multi-condition: {len(conditions)} parca_variants[/dim]")

    # Save conditions metadata for quantify to detect multi-condition cache
    if conditions:
        cond_meta = {"conditions": conditions, "n_samples": n_samples, "n_test": n_test}
        (cache_path / "conditions.json").write_text(_json.dumps(cond_meta, indent=2))

    vecoli_root = _get_vecoli_root()
    nf_temp = Path(vecoli_root) / "nextflow_temp" / experiment_id
    if nf_temp.exists():
        shutil.rmtree(nf_temp)

    n_variants = X_all.shape[0]
    total_sims = (n_variants + 1) * n_init_sims * generations
    history_base = output_dir / experiment_id / "history"
    ansi_re = re.compile(r"\x1b\[[\d;]*[A-Za-z]|\x1b\[\d*[A-GJK]|\x07")

    console.print(f"[dim]Step 3: running vEcoli workflow.py ({total_sims} sims)...[/dim]")

    env = os.environ.copy()
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = vecoli_root + (os.pathsep + existing if existing else "")
    env["PYTHONUNBUFFERED"] = "1"

    workflow_script = os.path.join(vecoli_root, "runscripts", "workflow.py")
    cmd = [sys.executable, workflow_script, "--config", str(config_path)]

    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        cwd=vecoli_root,
        env=env,
    )

    with Progress(
        SpinnerColumn("dots", style="bold magenta"),
        TextColumn("[bold cyan]{task.description:<50}"),
        BarColumn(bar_width=30, complete_style="green", finished_style="green"),
        TextColumn("[bold green]{task.percentage:>5.1f}%"),
        TextColumn("[dim]|[/dim]"),
        TimeElapsedColumn(),
        console=console,
        transient=False,
    ) as progress:
        task = progress.add_task("Launching Nextflow", total=total_sims)
        start = _time.monotonic()

        while proc.poll() is None:
            # Read available stdout
            if proc.stdout is not None:
                fd = proc.stdout.fileno()
                try:
                    chunk = os.read(fd, 8192)
                except OSError:
                    chunk = b""
                if chunk:
                    for raw_line in chunk.decode("utf-8", errors="replace").splitlines():
                        clean = ansi_re.sub("", raw_line).strip()
                        if not clean:
                            continue
                        low = clean.lower()
                        if any(kw in low for kw in ["error", "fail", "exception"]):
                            console.print(f"  [red]{clean}[/red]")
                        elif any(kw in low for kw in ["completed at", "duration", "succeeded"]):
                            console.print(f"  [green]{clean}[/green]")
                        elif any(kw in low for kw in ["warn", "note"]):
                            console.print(f"  [yellow]{clean}[/yellow]")
                        else:
                            console.print(f"  [dim]{clean}[/dim]")

            # Poll completed variants
            n_done = _count_completed_variants(history_base)
            elapsed = int(_time.monotonic() - start)
            progress.update(
                task,
                completed=n_done,
                description=f"Simulating  {n_done}/{total_sims}  [{elapsed}s]",
            )
            _time.sleep(0.5)

    # Drain remaining stdout
    if proc.stdout is not None:
        remaining = proc.stdout.read()
        if remaining:
            for raw_line in remaining.decode("utf-8", errors="replace").splitlines():
                clean = ansi_re.sub("", raw_line).strip()
                if clean:
                    console.print(f"  [dim]{clean}[/dim]")

    exit_code = proc.returncode
    if exit_code != 0:
        console.print(f"[yellow]workflow.py exited with code {exit_code}[/yellow]")

    # ── Step 4: Collect + cache ──
    console.print("[dim]Collecting Parquet outputs...[/dim]")
    if not history_base.exists():
        candidates = list(output_dir.glob("*/history"))
        if candidates:
            history_base = candidates[0]

    from uq.observables import collect_observables

    console.print(f"[dim]Observables: {observables}, gen_lower_bound={generation_lower_bound}[/dim]")
    n_variants = X_all.shape[0]
    Y_all, obs, Y_ts_all, Y_meta_all = collect_observables(
        history_base,
        n_variants,
        presets=observables,
        generation_lower_bound=generation_lower_bound,
    )

    # Split train and held-out test portions
    Y_agg = Y_all[:n_samples]
    Y_ts = Y_ts_all[:n_samples] if Y_ts_all else Y_ts_all
    Y_meta = Y_meta_all[:n_samples] if Y_meta_all else Y_meta_all

    Y_test_arr: np.ndarray | None = None
    Y_test_ts: list | None = None
    Y_test_meta: list | None = None
    if n_test > 0:
        Y_test_arr = Y_all[n_samples:]
        Y_test_ts = Y_ts_all[n_samples:] if Y_ts_all else None
        Y_test_meta = Y_meta_all[n_samples:] if Y_meta_all else None

    cache_path.mkdir(parents=True, exist_ok=True)
    cache = PrecomputedCache(
        cache_dir=cache_path,
        X=X_train,
        Y=Y_agg,
        parameter_names=param_space.parameter_names,
        metadata={"bounds": bounds.tolist(), "seed": seed, "observable_columns": obs, "backend": "vecoli"},
        Y_timeseries=Y_ts,
        Y_timeseries_meta=Y_meta,
        X_test=X_test,
        Y_test=Y_test_arr,
        Y_test_timeseries=Y_test_ts,
        Y_test_timeseries_meta=Y_test_meta,
    )
    cache.save()
    np.save(cache_path / "germ_train.npy", germ_train)
    if germ_test is not None:
        np.save(cache_path / "germ_test.npy", germ_test)

    console.print(
        f"[bold green]Cached {Y_agg.shape[0]} training samples[/bold green] "
        f"({Y_agg.shape[1]} outputs) to [cyan]{cache_path}[/cyan]"
    )
    if Y_test_arr is not None:
        console.print(f"  [dim]Held-out validation: {Y_test_arr.shape[0]} samples[/dim]")
    if Y_ts:
        console.print(f"  [dim]Timeseries: {len(Y_ts)} samples[/dim]")
    if Y_meta:
        console.print("  [dim]Metadata: generation/seed labels (strategies 2-3 enabled)[/dim]")


def _sample_v2ecoli(
    sim_data_path: str,
    cache_path: Path,
    X_all: np.ndarray,
    param_space: Any,
    n_samples: int,
    n_test: int,
    generations: int,
    max_duration: float,
    observables: list[str],
    generation_lower_bound: int,
    X_train: np.ndarray,
    X_test: np.ndarray | None,
    germ_train: np.ndarray,
    germ_test: np.ndarray | None,
    bounds: np.ndarray,
    seed: int,
    max_workers: int | None = None,
) -> None:
    """Step 3 (v2ecoli): in-process composite execution.

    Replaces the subprocess-based ``_sample_local`` with an in-process
    v2ecoli backend. Generates a cache bundle from simData, then runs
    one composite per sample with mutated parameters.
    """
    from libuq.sampling import PrecomputedCache
    from uq.generators.v2ecoli import (
        V2ecoliGenerator,
        generate_cache_bundle,
    )
    from uq.v2ecoli_bridge import observable_names_for_presets

    console.print("[dim]Step 3 (v2ecoli): generating cache bundle from simData...[/dim]")
    v2e_cache_dir = str(cache_path / "_v2ecoli_cache")
    generate_cache_bundle(sim_data_path, v2e_cache_dir, seed=seed)

    generator = V2ecoliGenerator(
        cache_dir=v2e_cache_dir,
        param_names=param_space.parameter_names,
        seed=seed,
        n_generations=generations,
        max_time=max_duration,
        max_workers=max_workers,
        presets=observables,
    )

    n_total = X_all.shape[0]
    console.print(f"[dim]Step 3: running {n_total} samples via v2ecoli (in-process)...[/dim]")

    Y_all, Y_ts_all, Y_meta_all = generator._run_batch(X_all)

    Y_agg = Y_all[:n_samples]
    Y_ts = Y_ts_all[:n_samples] if Y_ts_all else None
    Y_meta = Y_meta_all[:n_samples] if Y_meta_all else None

    Y_test_arr: np.ndarray | None = None
    Y_test_ts: list | None = None
    Y_test_meta: list | None = None
    if n_test > 0:
        Y_test_arr = Y_all[n_samples:]
        Y_test_ts = Y_ts_all[n_samples:] if Y_ts_all else None
        Y_test_meta = Y_meta_all[n_samples:] if Y_meta_all else None

    obs = observable_names_for_presets(observables)
    if generator._obs_names:
        obs = generator._obs_names

    cache_path.mkdir(parents=True, exist_ok=True)
    cache = PrecomputedCache(
        cache_dir=cache_path,
        X=X_train,
        Y=Y_agg,
        parameter_names=param_space.parameter_names,
        metadata={"bounds": bounds.tolist(), "seed": seed, "observable_columns": obs, "backend": "v2ecoli"},
        Y_timeseries=Y_ts,
        Y_timeseries_meta=Y_meta,
        X_test=X_test,
        Y_test=Y_test_arr,
        Y_test_timeseries=Y_test_ts,
        Y_test_timeseries_meta=Y_test_meta,
    )
    cache.save()
    np.save(cache_path / "germ_train.npy", germ_train)
    if germ_test is not None:
        np.save(cache_path / "germ_test.npy", germ_test)

    console.print(
        f"[bold green]Cached {Y_agg.shape[0]} training samples[/bold green] "
        f"(v2ecoli backend) to [cyan]{cache_path}[/cyan]"
    )
    if Y_test_arr is not None:
        console.print(f"  [dim]Held-out validation: {Y_test_arr.shape[0]} samples[/dim]")
    if Y_ts:
        console.print(f"  [dim]Timeseries: {len(Y_ts)} samples[/dim]")
    if Y_meta:
        console.print("  [dim]Metadata: generation/seed labels (strategies 2-3 enabled)[/dim]")


# ── quantify ─────────────────────────────────────────────────────────


@app.command()
def quantify(
    sim_data_path: str = typer.Argument(..., help="Path to simData.cPickle"),
    cache_dir: str = typer.Option("./uq_cache", help="Cache from sample step"),
    export_path: str = typer.Option("./uq_results", help="Export directory"),
    n_bins: int = 10,
    polynomial_order: int = 2,
    regression: str = "lsq",
    tol: float = typer.Option(1e-3, help="BCS sparsity tolerance (UQPC --tol, only used when --regression=bcs)"),
    bootstrap: int = typer.Option(
        0,
        "--bootstrap",
        help="Number of bootstrap resamples for empirical 95% CIs on Sobol "
        "indices (default 0 = off). Standard non-parametric bootstrap: "
        "resample (X, Y) rows with replacement, refit PCE, recompute Sobol, "
        "report 2.5–97.5 percentile interval per parameter. Typical: 200.",
    ),
) -> None:
    """UQPC Steps 4-5: fit PCE surrogates, compute Sobol (all 4 strategies).

    \b
    Strategy 1: Uniform (bulk) — population-averaged sensitivity
    Strategy 2: By generation — convergence control (needs generations >= 2)
    Strategy 3: By lineage seed — stochastic variance control
    Strategy 4: Growth-stratified — θ = normalized log(dry_mass)

    \b
    Regression methods (--regression):
      lsq  Least squares (default)
      bcs  Bayesian Compressed Sensing (sparse)
      anl  Analytical Bayesian
    """
    from uq.multi_condition import is_multi_condition_cache, quantify_multi_condition
    from uq.workflow import quantify as wf_quantify

    console.print(
        f"[bold cyan]Quantify:[/bold cyan] order={polynomial_order}, bins={n_bins}, regression={regression}, tol={tol}"
    )

    # Auto-detect multi-condition cache
    if is_multi_condition_cache(cache_dir):
        console.print("[bold magenta]Multi-condition cache detected — running cross-condition GSA (extension beyond RFC006)[/bold magenta]")
        mc_result = quantify_multi_condition(
            cache_dir=cache_dir,
            sim_data_path=sim_data_path,
            polynomial_order=polynomial_order,
            n_bins=n_bins,
            regression=regression,
            tolerance=tol,
            export_path=export_path,
        )
        for cond_id, cond_result in mc_result.per_condition.items():
            console.print(f"\n[bold cyan]── Condition: {cond_id} ──[/bold cyan]")
            _print_report(cond_result, polynomial_order=polynomial_order)
        _print_multi_condition_report(mc_result)
        console.print(f"\n  [bold green]Artifacts exported to:[/bold green] {export_path}")
        # Generate HTML report for each condition
        from uq.report import generate_html_report
        for cond_id in mc_result.per_condition:
            cond_dir = Path(export_path) / cond_id
            if (cond_dir / "uq_results.json").exists():
                rpt = generate_html_report(cond_dir)
                console.print(f"  [green]Report:[/green] {rpt}")
        return

    result = wf_quantify(
        cache_dir=cache_dir,
        sim_data_path=sim_data_path,
        polynomial_order=polynomial_order,
        n_bins=n_bins,
        regression=regression,
        tolerance=tol,
        export_path=export_path,
        n_bootstrap=bootstrap,
    )

    _print_report(result, polynomial_order=polynomial_order)
    _print_narrative(result)
    console.print(f"\n  [bold green]Artifacts exported to:[/bold green] {export_path}")

    # Generate HTML report
    from uq.report import generate_html_report
    rpt = generate_html_report(export_path)
    console.print(f"  [green]Report:[/green] {rpt}")


# ── Report rendering ─────────────────────────────────────────────────


def _pct(v: float) -> str:
    return f"{v * 100:.1f}%"


def _print_report(result: Any, polynomial_order: int | None = None) -> None:
    """Render QuantifyResult as a rich terminal report."""
    from rich.columns import Columns

    console.print()

    header = Text()
    header.append("  UQ SENSITIVITY REPORT  ", style="bold white on magenta")
    header.append("  vEcoli Whole-Cell Model", style="bold cyan")
    console.print(Panel(header, box=box.DOUBLE_EDGE, border_style="magenta", padding=(0, 1)))
    console.print("[dim]  Methods: PCE surrogate (Legendre, PyTUQ) + Sobol indices (Sudret 2008)[/dim]")
    console.print("[dim]  Strategies: 1=uniform, 2=by generation, 3=by seed, 4=growth-stratified (RFC006)[/dim]")
    console.print("[dim]  Refs: Macklin et al. Science 2020; Ahn-Horst et al. npj Syst Biol Appl 2022[/dim]")
    console.print()

    # Design quality at the *actual* polynomial order used in this quantify run
    # (the sample-time adequacy check assumes order=2; the user may have passed
    # a different --polynomial-order here, which changes basis_size and κ(A)).
    if polynomial_order is not None:
        _print_quantify_adequacy_panel(result, polynomial_order)

    # Surrogate quality diagnostics (UQPC step 5)
    _print_relerr_panel(result.strategy1)

    # Noise-floor decomposition (aleatoric vs epistemic variance) — uses
    # vEcoli's lineage_seed replicate structure already in the cache.
    noise = getattr(result, "noise_decomposition", None)
    if noise is not None:
        _print_noise_panel(noise)

    # Strategy 1: population
    s1_ci = getattr(result.strategy1, "sobol_total_order_ci", None)
    _print_sobol_table(
        "STRATEGY 1 // POPULATION-AVERAGED (all cells, all times)",
        result.strategy1.sobol,
        "cyan",
        total_ci=s1_ci,
    )

    # Strategy 2: by generation
    if result.strategy2:
        gen_tables = []
        for gen, r in sorted(result.strategy2.items()):
            r_ci = getattr(r, "sobol_total_order_ci", None)
            gen_tables.append(_sobol_table(f"Generation {gen}", r.sobol, "blue", n_top=3, total_ci=r_ci))
        console.print(
            Panel(
                Columns(gen_tables, equal=True, expand=True),
                title="[bold blue]STRATEGY 2 // BY GENERATION (convergence control)[/bold blue]",
                subtitle="[dim]Controls for transient dynamics in early generations[/dim]",
                border_style="blue",
                box=box.ROUNDED,
                padding=(0, 1),
            )
        )

    # Strategy 3: by lineage seed
    if result.strategy3:
        seed_tables = []
        for seed, r in sorted(result.strategy3.items()):
            r_ci = getattr(r, "sobol_total_order_ci", None)
            seed_tables.append(_sobol_table(f"Seed {seed}", r.sobol, "yellow", n_top=3, total_ci=r_ci))
        console.print(
            Panel(
                Columns(seed_tables, equal=True, expand=True),
                title="[bold yellow]STRATEGY 3 // BY LINEAGE SEED (stochastic variance control)[/bold yellow]",
                subtitle="[dim]Controls for gene expression noise and stochastic partitioning[/dim]",
                border_style="yellow",
                box=box.ROUNDED,
                padding=(0, 1),
            )
        )

    # Strategy 4: growth-stratified
    if result.strategy4_per_stage:
        n = len(result.strategy4_per_stage)
        tables = []
        for i, r in enumerate(result.strategy4_per_stage):
            lo, hi = i / n, (i + 1) / n
            r_ci = getattr(r, "sobol_total_order_ci", None)
            tables.append(_sobol_table(f"θ {lo:.0%}–{hi:.0%}", r.sobol, "green", n_top=3, total_ci=r_ci))
        console.print(
            Panel(
                Columns(tables, equal=True, expand=True),
                title=f"[bold green]STRATEGY 4 // GROWTH-STRATIFIED ({n} stages)[/bold green]",
                subtitle="[dim]θ = normalized log(dry_mass), 0=birth, 1=division[/dim]",
                border_style="green",
                box=box.ROUNDED,
                padding=(0, 1),
            )
        )


def _sobol_table(
    title: str, sobol: Any, border: str, n_top: int = 10,
    total_ci: Any = None,
) -> Table:
    table = Table(
        box=box.SIMPLE_HEAVY,
        show_header=True,
        header_style="bold magenta",
        border_style=border,
        title=title,
        title_style=f"bold {border}",
    )
    table.add_column("PARAMETER", style="bold yellow", no_wrap=True)
    if total_ci is not None:
        table.add_column("S_Ti  [95% CI]", style="bright_green", justify="right")
    else:
        table.add_column("S_Ti", style="bright_green", justify="right")

    total = sobol.total_order
    if total.ndim > 1:
        total = np.mean(total, axis=0)
    ranked = np.argsort(total)[::-1][:n_top]
    for i in ranked:
        name = sobol.parameter_names[i] if i < len(sobol.parameter_names) else f"x{i}"
        if total_ci is not None:
            cell = f"{_pct(total[i])}  [{_pct(total_ci[i, 0])}, {_pct(total_ci[i, 1])}]"
        else:
            cell = _pct(total[i])
        table.add_row(name, cell)
    return table


def _print_relerr_panel(s1: Any) -> None:
    """Show PCE surrogate relative errors (UQPC step 5 diagnostic).

    Columns: TRAIN (always); TEST (when an independent held-out set is
    in the cache); CV (k-fold cross-validation, when no test set —
    typically BYO mode or default with --n-test 0).
    """
    table = Table(
        box=box.SIMPLE_HEAVY,
        show_header=True,
        header_style="bold magenta",
        title="SURROGATE QUALITY // RELATIVE ERRORS",
        title_style="bold magenta",
    )
    table.add_column("OBSERVABLE", style="bold yellow", justify="left")
    table.add_column("TRAIN", style="bright_green", justify="right")

    test = s1.relerr_test
    cv = getattr(s1, "relerr_cv", None)
    cv_n_folds = getattr(s1, "cv_n_folds", 0)
    if test is not None:
        table.add_column("TEST", style="bright_cyan", justify="right")
    if cv is not None and len(cv) > 0:
        table.add_column(f"CV ({cv_n_folds}-fold)", style="bright_cyan", justify="right")

    train = s1.relerr_train
    n_out = len(train)
    for i in range(n_out):
        row = [f"output[{i}]", f"{train[i]:.3e}"]
        if test is not None:
            row.append(f"{test[i]:.3e}")
        if cv is not None and len(cv) > 0:
            row.append(f"{cv[i]:.3e}")
        table.add_row(*row)

    subtitle = "[dim]||Y − Ŷ||₂ / ||Y||₂ per output column"
    if test is None and (cv is None or len(cv) == 0):
        subtitle += " — no test/CV (n_samples too small for held-out fold)"
    subtitle += "[/dim]"

    console.print(
        Panel(
            table,
            border_style="magenta",
            box=box.ROUNDED,
            padding=(0, 1),
            subtitle=subtitle,
        )
    )


def _print_quantify_adequacy_panel(result: Any, polynomial_order: int) -> None:
    """PCE design quality at the actual quantify-time polynomial order.

    Re-runs ``_pce_sample_adequacy`` and ``_design_matrix_conditioning`` on
    the cached germ samples at the polynomial order the user actually passed
    to ``uq quantify`` (which may differ from the order=2 default the
    sample-time check assumed). Catches the case where ``--polynomial-order
    3`` silently turns an ok design into an underdetermined one.
    """
    from uq.vecoli_config import (
        _design_matrix_conditioning,
        _pce_sample_adequacy,
    )

    germ = getattr(result.strategy1, "germ_train", None)
    if germ is None or germ.ndim != 2:
        return
    n_samples, n_params = int(germ.shape[0]), int(germ.shape[1])

    status, advisory = _pce_sample_adequacy(n_samples, n_params, polynomial_order)
    if status == "underdetermined":
        adequacy_line = f"[red]Count: {advisory}[/red]"
    elif status == "marginal":
        adequacy_line = f"[yellow]Count: {advisory}[/yellow]"
    else:
        adequacy_line = f"[dim]Count: {advisory}[/dim]"

    if status == "underdetermined":
        cond_line = "[dim]κ(A): skipped — count check already flagged the basis as larger than N.[/dim]"
    else:
        _kappa, cond_status, cond_advisory = _design_matrix_conditioning(
            germ, n_params, polynomial_order,
        )
        if cond_status in ("ill-conditioned", "singular"):
            cond_line = f"[red]κ(A): {cond_advisory}[/red]"
        elif cond_status == "marginal":
            cond_line = f"[yellow]κ(A): {cond_advisory}[/yellow]"
        else:
            cond_line = f"[dim]κ(A): {cond_advisory}[/dim]"

    console.print(
        Panel(
            f"{adequacy_line}\n{cond_line}",
            title=f"[bold magenta]DESIGN QUALITY @ order={polynomial_order}[/bold magenta]",
            subtitle="[dim]Re-checked against the actual --polynomial-order used here[/dim]",
            border_style="magenta",
            box=box.ROUNDED,
            padding=(0, 1),
        )
    )


def _print_noise_panel(noise: Any) -> None:
    """Aleatoric vs epistemic variance decomposition.

    Uses vEcoli's lineage_seed replicate structure: each variant has
    ``n_init_sims`` independent stochastic realizations; within-variant
    variance across those is the noise floor. PCE Sobol indices measure
    the *signal* portion — interpret them against the reported signal
    fraction.
    """
    status = getattr(noise, "status", "estimated")

    if status == "not_estimable_single_seed":
        console.print(
            Panel(
                "[dim]Noise floor: not estimable — every variant has exactly "
                "one lineage seed (n_init_sims = 1). Aleatoric / epistemic "
                "separation requires ≥ 2 replicates per variant. "
                "Rerun `uq sample` with `--n-init-sims 4` (or higher) to "
                "estimate the noise floor honestly.[/dim]",
                title="[bold magenta]NOISE FLOOR // ALEATORIC vs EPISTEMIC[/bold magenta]",
                border_style="magenta",
                box=box.ROUNDED,
                padding=(0, 1),
            )
        )
        return

    signal = noise.per_output_signal_fraction
    noise_frac = noise.per_output_noise_fraction
    obs_names = getattr(noise, "observable_names", None) or [f"output[{j}]" for j in range(len(signal))]

    table = Table(
        box=box.SIMPLE_HEAVY,
        show_header=True,
        header_style="bold magenta",
        title="NOISE FLOOR // ALEATORIC vs EPISTEMIC",
        title_style="bold magenta",
    )
    table.add_column("OBSERVABLE", style="bold yellow", justify="left")
    table.add_column("SIGNAL  η²_between", style="bright_green", justify="right")
    table.add_column("NOISE  η²_within", style="bright_red", justify="right")

    for name, s, nz in zip(obs_names, signal, noise_frac):
        table.add_row(name, _pct(float(s)), _pct(float(nz)))

    mean_signal = float(signal.mean())
    if mean_signal >= 0.80:
        verdict = f"[dim]Signal dominates ({mean_signal * 100:.0f}% mean) — Sobol indices interpret straightforwardly.[/dim]"
    elif mean_signal >= 0.50:
        verdict = f"[yellow]Moderate aleatoric contribution ({1 - mean_signal:.0%} mean noise) — Sobol indices undershoot the per-parameter share of the *explainable* variance by ~1/{mean_signal:.2f}×.[/yellow]"
    else:
        verdict = f"[red]Aleatoric exceeds parameter signal ({1 - mean_signal:.0%} mean noise) — Sobol indices measure {mean_signal:.0%} of total variance only; interpret against the shrunken explainable variance.[/red]"

    extra = ""
    if status == "unbalanced":
        extra = "  [dim](unbalanced design: replicate counts vary across variants)[/dim]"

    console.print(
        Panel(
            table,
            border_style="magenta",
            box=box.ROUNDED,
            padding=(0, 1),
            subtitle=f"{verdict}{extra}",
        )
    )


def _print_sobol_table(title: str, sobol: Any, border: str, total_ci: Any = None) -> None:
    console.print(
        Panel(
            _sobol_table(title, sobol, border, total_ci=total_ci),
            border_style=border,
            box=box.ROUNDED,
            padding=(0, 1),
        )
    )


def _print_narrative(result: Any) -> None:
    """Print a plain-English 'Key Findings' summary after the Sobol tables."""
    names = result.parameter_names
    s1_total = result.strategy1.sobol.total_order
    if s1_total.ndim > 1:
        s1_total = np.mean(s1_total, axis=0)

    # Top parameter (bulk)
    top_idx = int(np.argmax(s1_total))
    top_name = names[top_idx]
    top_pct = s1_total[top_idx] * 100

    # Negligible parameter (lowest S_Ti across all strategies)
    min_idx = int(np.argmin(s1_total))
    min_name = names[min_idx]
    min_pct = s1_total[min_idx] * 100

    # Most sensitive growth stage (strategy 4)
    stage_line = ""
    if result.strategy4_per_stage:
        stage_sums = []
        for sr in result.strategy4_per_stage:
            st = sr.sobol.total_order
            if st.ndim > 1:
                st = np.mean(st, axis=0)
            stage_sums.append(float(np.sum(st)))
        peak_stage = int(np.argmax(stage_sums))
        n = len(result.strategy4_per_stage)
        lo_pct = int(100 * peak_stage / n)
        hi_pct = int(100 * (peak_stage + 1) / n)
        stage_line = f"  • Growth stage θ {lo_pct}–{hi_pct}% shows the strongest parameter sensitivity."

    # Recommendation
    if top_pct > 50:
        rec = f"[bold]{top_name}[/bold] dominates output variance — prioritize its measurement accuracy."
    elif top_pct > 20:
        rec = f"[bold]{top_name}[/bold] is the leading driver — consider tighter bounds or dedicated experiments."
    else:
        rec = "No single parameter dominates — output variance is distributed across multiple inputs."

    lines = [
        f"  • [bold]{top_name}[/bold] explains [bold]{top_pct:.1f}%[/bold] of population-level output variance (S_Ti).",
        f"  • [bold]{min_name}[/bold] is least influential at [bold]{min_pct:.1f}%[/bold] — candidate for fixing at nominal.",
    ]
    if stage_line:
        lines.append(stage_line)
    lines.append(f"  • Recommendation: {rec}")

    console.print()
    console.print(
        Panel(
            "\n".join(lines),
            title="[bold white on blue] KEY FINDINGS [/bold white on blue]",
            border_style="blue",
            box=box.ROUNDED,
            padding=(1, 2),
        )
    )


def _print_multi_condition_report(mc_result: Any) -> None:
    """Print cross-condition comparison table and narrative."""
    from uq.multi_condition import MultiConditionResult

    names = mc_result.parameter_names
    conditions = mc_result.conditions

    # ── Cross-condition S_Ti comparison table ──
    table = Table(
        box=box.SIMPLE_HEAVY,
        show_header=True,
        header_style="bold magenta",
        title="CROSS-CONDITION SENSITIVITY COMPARISON",
        title_style="bold magenta",
    )
    table.add_column("PARAMETER", style="bold yellow", no_wrap=True)
    for cond_id in conditions:
        table.add_column(cond_id, justify="right")
    table.add_column("STABILITY", justify="right")

    stability = mc_result.rank_stability
    for i, name in enumerate(names):
        row = [name]
        for cond_id in conditions:
            s_ti = mc_result.per_condition[cond_id].strategy1.sobol.total_order
            if s_ti.ndim > 1:
                s_ti = np.mean(s_ti, axis=0)
            row.append(_pct(s_ti[i]))
        # Stability indicator
        n_dots = max(1, int(stability[i] * 5))
        dots = "●" * n_dots + "○" * (5 - n_dots)
        row.append(f"{dots} ({stability[i]:.2f})")
        table.add_row(*row)

    console.print()
    console.print(
        Panel(
            table,
            border_style="magenta",
            box=box.ROUNDED,
            padding=(0, 1),
        )
    )

    # ── Cross-condition narrative ──
    lines = []
    if mc_result.universal_drivers:
        drivers = ", ".join(f"[bold]{d}[/bold]" for d in mc_result.universal_drivers)
        lines.append(f"  • Universal drivers (S_Ti > 10% in all conditions): {drivers}")
        lines.append("    These should be measured precisely regardless of growth condition.")

    for cond_id, params in mc_result.condition_specific.items():
        param_str = ", ".join(f"[bold]{p}[/bold]" for p in params)
        lines.append(f"  • Condition-specific to [cyan]{cond_id}[/cyan]: {param_str}")

    if not lines:
        lines.append("  • No clear universal or condition-specific patterns detected.")

    console.print(
        Panel(
            "\n".join(lines),
            title="[bold white on magenta] CROSS-CONDITION FINDINGS [/bold white on magenta]",
            border_style="magenta",
            box=box.ROUNDED,
            padding=(1, 2),
        )
    )


@app.command()
def dashboard(
    results_path: str | None = None,
    run_mode: str = "tk",
) -> None:
    """Launch the uq interactive dashboard.

    \b
    --run-mode tk   Tkinter DAW (default) — draggable parameter markers
    --run-mode mo   Marimo notebook — slider-reactive
    --run-mode web  Standalone web app (Dash) — shareable URL

    If --results-path is not given, looks for ./uq_results/uq_results.json.
    """
    import subprocess as _sp

    # Auto-detect results path
    if results_path is None:
        default = Path("./uq_results/uq_results.json")
        if default.exists():
            results_path = str(default)

    if run_mode == "mo":
        _sp.run(["uv", "run", "marimo", "edit", "--no-token", "app/dashboard_simple.py"], check=True)  # noqa: S607
    elif run_mode == "web":
        from app.web_dashboard import run_web_dashboard

        # Pass the directory, not the JSON file
        rp = results_path
        if rp and rp.endswith(".json"):
            rp = str(Path(rp).parent)
        run_web_dashboard(results_path=rp)
    else:
        from app.uq_daw_simple import run_tk_dashboard_simple

        run_tk_dashboard_simple(data_path=results_path)  # type: ignore[no-untyped-call]


@app.command()
def tui() -> None:
    """Launch the UQPC interactive terminal UI (Textual).

    \b
    Sidebar: config + sample + quantify + results.
    Keyboard: q=quit  d=theme
    """
    from uq.tui import UQPCApp

    UQPCApp().run()


@app.command()
def gui() -> None:
    """Launch the UQPC interactive GUI (marimo).

    Full workflow in the browser: configure, sample, quantify, explore.
    """
    import subprocess as _sp

    _sp.run(
        ["uv", "run", "marimo", "run", "--no-token", "app/gui.py"],  # noqa: S607
        check=True,
    )


@app.command(name="show-config")
def show_config(
    sim_data_path: str = typer.Argument(..., help="Path to simData.cPickle"),
    n_samples: int = 5,
    seed: int = 42,
    generations: int = 1,
    n_init_sims: int = 1,
    max_duration: float = 10800.0,
    params_file: str | None = None,
    output_file: str | None = typer.Option(None, help="Write JSON to this file instead of stdout"),
) -> None:
    """Show the full vEcoli workflow config JSON that `sample` would pass.

    \b
    Generates the config without running vEcoli — useful for review,
    debugging, or manual execution via workflow.py --config.
    """
    import json as _json

    from libuq.pipeline.models import SimDataParameter
    from libuq.pipeline.param_loader import ParameterDataset
    from uq.vecoli_config import _build_config, _build_variants_from_samples
    from uq.workflow import _setup_input_pc

    sim_data_path = str(Path(sim_data_path).resolve())
    ds = ParameterDataset(sim_data_path=sim_data_path)
    if params_file is not None:
        raw = _json.loads(Path(params_file).read_text())
        parameters = [SimDataParameter.from_dict(p) for p in raw]
    else:
        parameters = None
    param_space = ds.to_parameter_space(parameters=parameters)
    bounds = np.array(param_space.parameter_bounds)

    input_pc, _, _ = _setup_input_pc(bounds)
    np.random.seed(seed)
    germ = input_pc.sampleGerm(n_samples)
    X = input_pc.evalPC(germ)

    variants = _build_variants_from_samples(X, param_space._sim_data_parameters)
    config = _build_config(
        sim_data_path=sim_data_path,
        output_dir="<OUTPUT_DIR>",
        variants_section=variants,
        n_init_sims=n_init_sims,
        generations=generations,
        max_duration=max_duration,
    )

    text = _json.dumps(config, indent=2)
    if output_file:
        Path(output_file).write_text(text)
        console.print(f"[green]Config written to {output_file}[/green]")
    else:
        from rich.syntax import Syntax

        console.print(Syntax(text, "json", theme="monokai", line_numbers=True))

    console.print(f"\n[dim]{n_samples} variants × {n_init_sims} seeds × {generations} gens "
                  f"= {(n_samples + 1) * n_init_sims * generations} total sims[/dim]")


@app.command()
def compare(
    dirs: list[str] = typer.Argument(..., help="Two or more uq export directories to compare"),
) -> None:
    """Compare Sobol indices across multiple UQ experiments.

    \b
    Loads uq_results.json from each directory and prints side-by-side
    Sobol tables with differential highlighting.

    Example:
      uq compare ./results_baseline ./results_perturbed
    """
    import json as _json

    if len(dirs) < 2:
        console.print("[red]Need at least 2 directories to compare.[/red]")
        raise typer.Exit(1)

    results = []
    for d in dirs:
        p = Path(d) / "uq_results.json"
        if not p.exists():
            console.print(f"[red]Not found: {p}[/red]")
            raise typer.Exit(1)
        results.append((_json.loads(p.read_text()), Path(d).name))

    # Gather all param names (union)
    all_params = list(results[0][0]["parameters"].keys())

    # Side-by-side S_Ti table
    table = Table(
        box=box.SIMPLE_HEAVY,
        show_header=True,
        header_style="bold magenta",
        title="S_Ti COMPARISON (Population)",
        title_style="bold cyan",
    )
    table.add_column("PARAMETER", style="bold yellow", no_wrap=True)
    for _, name in results:
        table.add_column(name, justify="right")
    if len(results) == 2:
        table.add_column("Δ", justify="right", style="bold")

    for p in all_params:
        row = [p]
        vals = []
        for data, _ in results:
            v = data["phase1_population"]["sobol_total_order"].get(p, 0)
            vals.append(v)
            row.append(f"{v:.3f}")
        if len(results) == 2:
            delta = vals[1] - vals[0]
            sign = "+" if delta >= 0 else ""
            color = "green" if abs(delta) < 0.05 else ("red" if delta > 0 else "blue")
            row.append(f"[{color}]{sign}{delta:.3f}[/{color}]")
        table.add_row(*row)

    console.print(Panel(table, border_style="cyan", box=box.ROUNDED, padding=(0, 1)))

    # Per-strategy summary
    for i, (data, name) in enumerate(results):
        s2 = data.get("strategy2_by_generation", {})
        s3 = data.get("strategy3_by_seed", {})
        s4 = data.get("phase2_growth_stratified", {})
        console.print(
            f"  [dim]{name}: S2={s2.get('n_generations', 0)} gens, "
            f"S3={s3.get('n_seeds', 0)} seeds, "
            f"S4={s4.get('n_stages', 0)} stages[/dim]"
        )


@app.command(name="export-figures")
def export_figures(
    results_path: str = typer.Option("./uq_results", help="Path to uq export directory"),
    output_dir: str | None = typer.Option(None, help="Output directory (default: <results>/figures/)"),
) -> None:
    """Generate publication-ready PDFs and LaTeX from UQ results.

    \b
    Outputs:
      sobol_bar_chart.pdf    Grouped bars (S_Ti per strategy, all params)
      spectrogram.pdf        Print-quality sensitivity heatmap
      response_curves.pdf    PCE response curves at midpoint
      sobol_table.tex        LaTeX table of top-K params per strategy

    Requires kaleido: uv pip install kaleido
    """
    from uq.viz_export import export_all_figures

    console.print(f"[bold cyan]Exporting figures from:[/bold cyan] {results_path}")
    generated = export_all_figures(results_path, output_dir)
    for p in generated:
        console.print(f"  [green]✓[/green] {p}")
    console.print(f"\n[bold green]{len(generated)} figures exported.[/bold green]")


@app.command(name="report")
def report(
    results_path: str = typer.Option("./uq_results", help="Path to uq export directory"),
    output: str | None = typer.Option(None, "--output", "-o", help="Output HTML path (default: <results>/report.html)"),
) -> None:
    """Generate a self-contained HTML report from UQ results.

    \b
    Reads uq_results.json and manifest.json from the export directory
    and produces a single HTML file with inline SVG charts, parameter
    rankings, spectrogram, and provenance metadata.
    """
    from uq.report import generate_html_report

    path = generate_html_report(results_path, output)
    console.print(f"[bold green]Report written to:[/bold green] {path}")


@app.command(name="suggest-experiment")
def suggest_experiment(
    results_path: str = typer.Option("./uq_results", help="Path to uq export directory"),
    n_grid: int = typer.Option(1000, help="Grid points for variance scanning"),
) -> None:
    """Identify where to measure next to reduce uncertainty most.

    \b
    Loads the fitted PCE surrogate, evaluates prediction variance
    across the parameter space, and reports the region of maximum
    uncertainty in physical units.
    """
    import json as _json

    results_dir = Path(results_path)
    pop_dir = results_dir / "population_surrogate"
    if not (pop_dir / "coefficients.npy").exists():
        console.print("[red]Surrogate files not found. Run `uq quantify` first.[/red]")
        raise typer.Exit(1)

    coeffs = np.load(pop_dir / "coefficients.npy")
    mi = np.load(pop_dir / "multi_indices.npy")
    bounds = np.load(pop_dir / "input_bounds.npy")

    data = _json.loads((results_dir / "uq_results.json").read_text())
    params = list(data["parameters"].keys())
    n_params = len(params)

    # Grid-sample parameter space, estimate prediction variance
    # via PCE coefficient variance approximation
    rng = np.random.default_rng(42)
    X_grid = bounds[:, 0] + rng.random((n_grid, n_params)) * (bounds[:, 1] - bounds[:, 0])

    # Evaluate PCE at each grid point — variance proxy: |Ŷ - Ŷ_midpoint|
    mid = 0.5 * (bounds[:, 0] + bounds[:, 1])
    mid_norm = 2.0 * (mid - bounds[:, 0]) / (bounds[:, 1] - bounds[:, 0] + 1e-12) - 1.0

    def _legendre_eval_vec(x_phys):
        x_norm = 2.0 * (x_phys - bounds[:, 0]) / (bounds[:, 1] - bounds[:, 0] + 1e-12) - 1.0
        max_ord = int(mi.max()) if mi.size else 0
        P = np.zeros((max_ord + 1, n_params))
        P[0, :] = 1.0
        if max_ord >= 1:
            P[1, :] = x_norm
        for n in range(2, max_ord + 1):
            P[n, :] = ((2 * n - 1) * x_norm * P[n - 1, :] - (n - 1) * P[n - 2, :]) / n
        result = 0.0
        for t in range(len(coeffs)):
            term = float(coeffs[t])
            for p in range(n_params):
                term *= P[mi[t, p], p]
            result += term
        return result

    y_mid = _legendre_eval_vec(mid)
    variances = np.zeros(n_grid)
    for i in range(n_grid):
        y_i = _legendre_eval_vec(X_grid[i])
        variances[i] = (y_i - y_mid) ** 2

    # Find max-variance point
    best_idx = int(np.argmax(variances))
    best_x = X_grid[best_idx]
    best_var = variances[best_idx]

    console.print("\n[bold cyan]SUGGESTED NEXT EXPERIMENT[/bold cyan]")
    console.print(f"  [dim]Scanned {n_grid} parameter combinations[/dim]\n")

    table = Table(box=box.SIMPLE_HEAVY, show_header=True, header_style="bold magenta")
    table.add_column("PARAMETER", style="bold yellow")
    table.add_column("VALUE", justify="right")
    table.add_column("RANGE", justify="right", style="dim")

    for i, p in enumerate(params):
        table.add_row(
            p,
            f"{best_x[i]:.4f}",
            f"[{bounds[i, 0]:.4f}, {bounds[i, 1]:.4f}]",
        )

    console.print(Panel(
        table,
        title="[bold]Max-uncertainty parameter setting[/bold]",
        subtitle=f"[dim]Prediction variance: {best_var:.3e}[/dim]",
        border_style="cyan",
        box=box.ROUNDED,
    ))

    # Which param contributes most to variance at this point?
    sensitivities = []
    for pi in range(n_params):
        delta = (bounds[pi, 1] - bounds[pi, 0]) * 0.01
        x_plus = best_x.copy()
        x_plus[pi] = min(best_x[pi] + delta, bounds[pi, 1])
        x_minus = best_x.copy()
        x_minus[pi] = max(best_x[pi] - delta, bounds[pi, 0])
        sens = abs(_legendre_eval_vec(x_plus) - _legendre_eval_vec(x_minus)) / (2 * delta + 1e-12)
        sensitivities.append((params[pi], sens))
    sensitivities.sort(key=lambda x: -x[1])
    top_param = sensitivities[0][0]
    console.print(
        f"\n  [bold]Recommendation:[/bold] Measure [bold]{top_param}[/bold] more precisely "
        f"in the range [{bounds[params.index(top_param), 0]:.4f}, "
        f"{bounds[params.index(top_param), 1]:.4f}] to reduce uncertainty most."
    )


@app.command(name="init")
def init_project() -> None:
    """Guided setup wizard — configure a UQ project interactively.

    \b
    Detects vEcoli, validates simData, chooses observable presets,
    and writes uq_config.json for use with `uq sample`.
    """
    import json as _json

    console.print("[bold cyan]UQ Project Setup Wizard[/bold cyan]\n")

    # Step 1: Detect vEcoli simData
    sim_data_path = None
    search_paths = [
        Path("../vEcoli/reconstruction/sim_data/kb/simData.cPickle"),
        Path("sim_data/baseline/kb/simData.cPickle"),
        Path.home() / ".local/share/vEcoli/simData.cPickle",
    ]
    for sp in search_paths:
        if sp.resolve().exists():
            sim_data_path = str(sp.resolve())
            console.print(f"  [green]Auto-detected simData:[/green] {sim_data_path}")
            break

    if sim_data_path is None:
        sim_data_path = typer.prompt("Path to simData.cPickle")
    else:
        override = typer.prompt(f"simData path [{sim_data_path}]", default=sim_data_path)
        sim_data_path = override

    if not Path(sim_data_path).exists():
        console.print(f"[red]Not found: {sim_data_path}[/red]")
        raise typer.Exit(1)

    # Step 2: Validate vEcoli importable
    try:
        import ecoli  # type: ignore[import-not-found]  # noqa: F401
        console.print("  [green]vEcoli: importable[/green]")
    except ImportError:
        console.print("  [yellow]vEcoli not importable — sampling will fail.[/yellow]")

    # Step 3: Observable preset
    console.print("\n[bold]Observable presets:[/bold]")
    presets = {
        "mass": "Raw mass/growth scalars (5 features, fastest)",
        "higher_order": "Derived cd1 metrics: doubling time, growth rate, compositions (6 features)",
        "exchange_fluxes": "External metabolite fluxes (~87 features)",
        "transcriptome": "mRNA cistron counts (~4300 genes)",
        "proteome": "Protein monomer counts (~4300 monomers)",
        "fluxome": "Base reaction fluxes (~2800, dry-mass normalized)",
    }
    for k, v in presets.items():
        console.print(f"  [cyan]{k:<20s}[/cyan] {v}")
    obs_input = typer.prompt("\nObservable presets (comma-separated)", default="mass")
    obs_list = [o.strip() for o in obs_input.split(",") if o.strip()]

    # Step 4: Sample count
    console.print("\n[bold]Sample count:[/bold]")
    console.print("  20  — quick exploration (~2h)")
    console.print("  50  — production quality (~5h)")
    console.print("  200 — high-fidelity (~20h)")
    n_samples = int(typer.prompt("Number of samples", default="20"))

    # Step 5: Regression method
    reg = typer.prompt("Regression method (lsq/bcs/anl)", default="lsq")

    # Step 6: Write config
    config = {
        "sim_data_path": sim_data_path,
        "cache_dir": "./uq_cache",
        "export_path": "./uq_results",
        "n_samples": n_samples,
        "observables": obs_list,
        "polynomial_order": 2,
        "regression": reg,
        "n_bins": 10,
        "generations": 1,
        "n_init_sims": 1,
        "seed": 42,
    }

    config_path = Path("uq_config.json")
    config_path.write_text(_json.dumps(config, indent=2))
    console.print(f"\n[bold green]Config written to:[/bold green] {config_path}")
    console.print(
        f"\n[bold]Ready![/bold] Run:\n"
        f"  [cyan]uv run uq sample {sim_data_path} "
        f"--cache-dir ./uq_cache --n-samples {n_samples} "
        f"--observables {' --observables '.join(obs_list)}[/cyan]\n"
        f"  [cyan]uv run uq quantify {sim_data_path} "
        f"--cache-dir ./uq_cache --export-path ./uq_results "
        f"--regression {reg}[/cyan]"
    )


@app.command()
def fetch(
    simulation_id: int = typer.Argument(..., help="SMS-API simulation database_id to fetch"),
    cache_dir: str = typer.Option("./uq_cache", help="Cache output directory"),
    api_url: str = typer.Option("http://localhost:8080", help="SMS-API base URL"),
    observables: list[str] = typer.Option(
        ["higher_order", "transcriptome", "proteome", "fluxome", "exchange_fluxes"],
        help="cd1 observable presets to extract from the downloaded TSVs.",
    ),
) -> None:
    """Fetch cd1 analysis outputs from a completed SMS-API simulation.

    \b
    Downloads the tar.gz from /simulations/{id}/data, parses cd1_* TSV
    files (identifier, mean, std), and prints the observable summary.
    Useful for inspecting what the SMS-API produces before running a
    full remote UQ campaign.

    \b
    Example:
      uq fetch 48 --api-url http://localhost:8080
      uq fetch 48 --observables higher_order --observables transcriptome
    """
    from uq.remote import SmsApiClient, find_cd1_tsvs, parse_cd1_tsv, parse_cd1_tsvs

    cache_path = Path(cache_dir).resolve()
    dl_dir = cache_path / "_remote" / f"sim_{simulation_id}"
    dl_dir.mkdir(parents=True, exist_ok=True)

    console.print(f"[bold cyan]Fetching simulation {simulation_id}[/bold cyan] from {api_url}")

    with SmsApiClient(base_url=api_url) as client:
        # Get simulation info
        sim_info = client.get_simulation(simulation_id)
        exp_id = sim_info.get("experiment_id", "?")
        cfg = sim_info.get("config", {})
        gens = cfg.get("generations", "?")
        seeds = cfg.get("n_init_sims", "?")
        console.print(f"  [dim]experiment: {exp_id}[/dim]")
        console.print(f"  [dim]config: {gens} gens, {seeds} seeds[/dim]")

        # Download
        console.print("[dim]Downloading analysis outputs...[/dim]")
        output_dir = client.download_data(simulation_id, dest=dl_dir)
        console.print(f"  [dim]Extracted to: {output_dir}[/dim]")

    # Parse cd1 TSVs
    tsv_paths = find_cd1_tsvs(output_dir)
    if not tsv_paths:
        console.print("[red]No cd1 TSV files found in the downloaded archive.[/red]")
        raise typer.Exit(1)

    console.print(f"\n[bold cyan]cd1 Analysis Outputs[/bold cyan]")

    table = Table(
        box=box.SIMPLE_HEAVY,
        show_header=True,
        header_style="bold magenta",
    )
    table.add_column("PRESET", style="bold yellow")
    table.add_column("MODULE", style="dim")
    table.add_column("ROWS", justify="right")
    table.add_column("FILE", style="dim")

    total_obs = 0
    for preset in observables:
        if preset not in tsv_paths:
            table.add_row(preset, "—", "0", "[red]not found[/red]")
            continue
        path = tsv_paths[preset]
        ids, means, _stds = parse_cd1_tsv(path)
        from uq.remote import CD1_MODULE_MAP
        module = CD1_MODULE_MAP.get(preset, {}).get("module", "?")
        table.add_row(preset, module, str(len(ids)), str(path.name))
        total_obs += len(ids)

        # Show first few identifiers
        if ids:
            preview = ", ".join(ids[:5])
            if len(ids) > 5:
                preview += f", ... (+{len(ids) - 5} more)"
            console.print(f"    [dim]{preview}[/dim]")

    console.print(Panel(table, border_style="cyan", box=box.ROUNDED, padding=(0, 1)))
    console.print(f"\n  [bold green]Total observables: {total_obs}[/bold green]")
    console.print(f"  [dim]Archive at: {dl_dir}[/dim]")

    if total_obs == 0:
        console.print(
            "\n  [yellow]All TSVs are empty (headers only). This usually means "
            "generation_lower_bound filtered out all data. Check that the simulation "
            "ran enough generations.[/yellow]"
        )


@app.command(name="plan")
def plan(
    budget: int = typer.Option(
        ...,
        "--budget",
        help="Total vEcoli runs available (n_samples × n_init_sims × generations).",
    ),
    n_params: int = typer.Option(
        6,
        "--params",
        help="PCE input dimension. Default 6 (DEFAULT_SIM_DATA_PARAMETERS).",
    ),
    polynomial_order: int = typer.Option(
        2,
        "--polynomial-order",
        help="PCE order at quantify time. Default 2.",
    ),
    noise_replicates: int = typer.Option(
        4,
        "--noise-replicates",
        help="Minimum n_init_sims required to estimate the noise floor "
        "(Triage 5 / Gap 2). Default 4.",
    ),
    generations: int = typer.Option(
        4,
        "--generations",
        help="Generations per variant for cell-cycle coverage. Default 4.",
    ),
) -> None:
    """Recommend (n_samples, n_init_sims, generations) for a compute budget.

    \b
    Holds noise_replicates and generations fixed and derives n_samples
    from the budget; reports PCE adequacy at the requested polynomial
    order plus alternatives showing the trade-off (halve generations,
    drop replicates to 1). Use before `uq sample` to size a run.
    """
    from uq.vecoli_config import _recommend_compute_allocation

    candidates = _recommend_compute_allocation(
        budget=budget,
        n_params=n_params,
        polynomial_order=polynomial_order,
        min_replicates=noise_replicates,
        generations=generations,
    )

    console.print(
        f"[bold cyan]Budget:[/bold cyan] {budget} vEcoli runs   "
        f"[dim]·[/dim]   d={n_params}, p={polynomial_order}, "
        f"basis_size={candidates[0].basis_size}"
    )

    table = Table(
        box=box.SIMPLE_HEAVY,
        show_header=True,
        header_style="bold magenta",
        title="COMPUTE ALLOCATION OPTIONS",
        title_style="bold magenta",
    )
    table.add_column("OPTION", style="bold yellow")
    table.add_column("n_samples", justify="right")
    table.add_column("n_init_sims", justify="right")
    table.add_column("generations", justify="right")
    table.add_column("ratio (N/B)", justify="right")
    table.add_column("status", justify="left")
    table.add_column("noise floor?", justify="left")

    for c in candidates:
        if c.adequacy_status == "underdetermined":
            status_cell = f"[red]{c.adequacy_status}[/red]"
        elif c.adequacy_status == "marginal":
            status_cell = f"[yellow]{c.adequacy_status}[/yellow]"
        else:
            status_cell = f"[green]{c.adequacy_status}[/green]"
        noise_cell = "[green]yes[/green]" if c.noise_estimable else "[red]no (n=1)[/red]"
        table.add_row(
            c.label,
            str(c.n_samples),
            str(c.n_init_sims),
            str(c.generations),
            f"{c.adequacy_ratio:.2f}×",
            status_cell,
            noise_cell,
        )

    console.print(Panel(table, border_style="magenta", box=box.ROUNDED, padding=(0, 1)))

    rec = candidates[0]
    if rec.adequacy_status == "underdetermined":
        console.print(
            f"  [red]Recommended config is underdetermined.[/red] Either "
            "raise --budget, drop --polynomial-order, or use --regression bcs."
        )
    elif rec.adequacy_status == "marginal":
        console.print(
            f"  [yellow]Recommended config is marginal[/yellow] — PCE will fit "
            "but Sobol indices will be noisy. Consider raising budget."
        )
    if not rec.noise_estimable:
        console.print(
            "  [red]Recommended config has only 1 replicate — noise floor "
            "(Gap 2) will not be estimable.[/red]"
        )


_HELP_SUBCOMMAND_ALIASES = {"help", "--help", "-h"}


@app.command(name="help")
def display_help(
    command: str | None = typer.Argument(default=None, help="Command to show help for"),
) -> None:
    """Show help for a specific subcommand, or the main CLI.

    \b
    Examples:
      uq help              Show all available commands
      uq help sample       Show help for the sample command
      uq help quantify     Show help for the quantify command
      uq sample help       Same as uq help sample
    """
    import click

    cmd = typer.main.get_command(app)
    if command is None:
        # Show main CLI help
        with click.Context(cmd) as ctx:
            console.print(cmd.get_help(ctx))
        return

    # Show help for a specific subcommand
    with click.Context(cmd) as ctx:
        sub = cmd.get_command(ctx, command)  # type: ignore[attr-defined]
        if sub is None:
            console.print(f"[red]Unknown command: {command}[/red]")
            console.print("[dim]Run `uq help` to see available commands.[/dim]")
            raise typer.Exit(1)
        with click.Context(sub, parent=ctx) as sub_ctx:
            console.print(sub.get_help(sub_ctx))


def main() -> None:
    """Entry point.

    Rewrites ``uq <cmd> ... help`` → ``uq <cmd> ... --help`` so users can
    discover flags with a trailing ``help`` word at any nesting level
    (``uq help``, ``uq sample help``, ``uq quantify help``, …)
    in addition to the standard ``uq <cmd> --help`` form.
    """
    import sys

    argv = sys.argv
    # "uq help" → "uq --help" (bare help with no subcommand)
    if len(argv) == 2 and argv[-1] in _HELP_SUBCOMMAND_ALIASES:
        argv[-1] = "--help"
    # "uq sample help" → "uq sample --help" (trailing help on any command)
    elif len(argv) >= 3 and argv[-1] in _HELP_SUBCOMMAND_ALIASES:
        argv[-1] = "--help"
    app()


if __name__ == "__main__":
    main()
