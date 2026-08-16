"""SMS-API client for remote vEcoli simulation execution.

Wraps the REST API at ``http://localhost:8080`` (stanford-test via port-forward)
to submit simulation workflows, poll for completion, download cd1 analysis
outputs, and parse them into UQ-compatible Y vectors.

The SMS-API ``/simulations/{id}/data`` endpoint returns a tar.gz archive
containing cd1 analysis TSV files::

    {experiment_id}/analyses/variant={v}/plots/analysis={module}/{file}.tsv

Each TSV has 3 tab-separated columns: ``identifier``, ``mean``, ``std``.
The ``mean`` column is what we use as the Y vector for UQ.

cd1 module → UQ observable mapping:

    cd1_higher_order_properties  → higher_order   (Properties, mean, std)
    cd1_transcriptomics          → transcriptome   (EcoCyc Gene ID, mean, std)
    cd1_proteomics               → proteome        (EcoCyc Monomer ID, mean, std)
    cd1_fluxomics                → fluxome         (EcoCyc Reaction ID, mean, std)
    cd1_metabolomics             → exchange_fluxes (EcoCyc Compound ID, mean, std)

Usage::

    from uq.remote import SmsApiClient, parse_cd1_tsvs

    client = SmsApiClient(base_url="http://localhost:8080")
    sim = client.submit_simulation(simulator_id=11, experiment_id="uq-0", ...)
    client.poll_until_complete(sim["database_id"])
    output_dir = client.download_data(sim["database_id"], dest=Path("./output"))
    Y, obs_names = parse_cd1_tsvs(output_dir, presets=["higher_order", "transcriptome"])
"""

from __future__ import annotations

import csv
import logging
import tarfile
import time
from pathlib import Path
from typing import Any

import httpx
import numpy as np

logger = logging.getLogger(__name__)

DEFAULT_BASE_URL = "http://localhost:8080"
DEFAULT_TIMEOUT = 300
POLL_INTERVAL = 30
MAX_POLL_DURATION = 7200  # 2 hours

# ── cd1 module ↔ UQ observable preset mapping ────────────────────────

CD1_MODULE_MAP: dict[str, dict[str, str]] = {
    "higher_order": {
        "module": "cd1_higher_order_properties",
        "filename": "higher_order_properties.tsv",
    },
    "transcriptome": {
        "module": "cd1_transcriptomics",
        "filename": "transcriptomics.tsv",
    },
    "proteome": {
        "module": "cd1_proteomics",
        "filename": "proteomics.tsv",
    },
    "fluxome": {
        "module": "cd1_fluxomics",
        "filename": "cd1_fluxomics_detailed.tsv",
    },
    "exchange_fluxes": {
        "module": "cd1_metabolomics",
        "filename": "metabolomics.tsv",
    },
}

# Default analysis_options to request when submitting via the API at the sms-api-stanford-test namespace
DEFAULT_CD1_ANALYSIS_OPTIONS: dict[str, Any] = {
    "multiseed": {
        "cd1_fluxomics": {"generation_lower_bound": 5},
        "cd1_proteomics": {"generation_lower_bound": 5},
        "cd1_metabolomics": {"generation_lower_bound": 5},
        "cd1_transcriptomics": {"generation_lower_bound": 5},
        "cd1_higher_order_properties": {"generation_lower_bound": 5},
    }
}


# Default analysis_options to request when submitting via the API at the sms-api-rke namespace (ccam hpc)
DEFAULT_ACADEMIC_ANALYSIS_OPTIONS: dict[str, Any] = {
    "single": {
        "mass_fraction_summary": {}
    }
}


# ── TSV parsing ──────────────────────────────────────────────────────


def parse_cd1_tsv(tsv_path: Path) -> tuple[list[str], np.ndarray, np.ndarray]:
    """Parse a cd1 analysis TSV file.

    Returns:
        (identifiers, means, stds) where identifiers are the row labels
        (gene IDs, reaction IDs, etc.) and means/stds are float arrays.
    """
    identifiers: list[str] = []
    means: list[float] = []
    stds: list[float] = []

    with open(tsv_path, newline="") as f:
        reader = csv.reader(f, delimiter="\t")
        header = next(reader, None)
        if header is None:
            return [], np.array([]), np.array([])

        for row in reader:
            if len(row) < 3 or not row[0].strip():
                continue
            identifiers.append(row[0].strip())
            means.append(float(row[1]))
            stds.append(float(row[2]))

    return identifiers, np.array(means), np.array(stds)


def find_cd1_tsvs(
    output_dir: Path,
    variant: int = 0,
) -> dict[str, Path]:
    """Locate cd1 TSV files in an extracted simulation archive.

    Returns dict mapping UQ preset name → TSV path.
    """
    found: dict[str, Path] = {}

    for preset, info in CD1_MODULE_MAP.items():
        module_name = info["module"]
        filename = info["filename"]

        # Standard path: {exp_id}/analyses/variant={v}/plots/analysis={module}/{file}.tsv
        candidates = list(output_dir.rglob(f"analysis={module_name}/{filename}"))
        # Filter to the correct variant
        for c in candidates:
            if f"variant={variant}" in str(c):
                found[preset] = c
                break
        # Fallback: any match
        if preset not in found and candidates:
            found[preset] = candidates[0]

    return found


def parse_cd1_tsvs(
    output_dir: Path,
    presets: list[str] | None = None,
    variant: int = 0,
) -> tuple[np.ndarray, list[str]]:
    """Parse cd1 TSV files from an extracted archive into a Y vector.

    Args:
        output_dir: Root of extracted tar.gz.
        presets: UQ observable presets to include. If None, includes all found.
        variant: Variant index to extract (default 0 = baseline).

    Returns:
        (Y, observable_names) where Y is shape ``(n_obs,)`` of mean values
        and observable_names are the full identifiers (e.g. "cd1_transcriptomics:geneX").
    """
    tsv_paths = find_cd1_tsvs(output_dir, variant=variant)

    if presets is None:
        presets = list(tsv_paths.keys())

    all_means: list[float] = []
    all_names: list[str] = []

    for preset in presets:
        if preset not in tsv_paths:
            logger.warning("Preset %r not found in output (available: %s)", preset, list(tsv_paths.keys()))
            continue
        path = tsv_paths[preset]
        module = CD1_MODULE_MAP[preset]["module"]
        identifiers, means, _stds = parse_cd1_tsv(path)
        for ident, val in zip(identifiers, means):
            all_names.append(f"{module}:{ident}")
            all_means.append(val)

    return np.array(all_means), all_names


def parse_cd1_tsvs_multi_variant(
    output_dir: Path,
    n_variants: int,
    presets: list[str] | None = None,
) -> tuple[np.ndarray, list[str]]:
    """Parse cd1 TSVs for multiple variants into a Y matrix.

    Returns:
        (Y, observable_names) where Y is shape ``(n_variants, n_obs)``.
    """
    rows: list[np.ndarray] = []
    obs_names: list[str] | None = None

    for v in range(n_variants):
        y_row, names = parse_cd1_tsvs(output_dir, presets=presets, variant=v)
        if obs_names is None:
            obs_names = names
        rows.append(y_row)

    if not rows:
        return np.array([]), []

    return np.vstack(rows), obs_names or []


# ── Exceptions ───────────────────────────────────────────────────────


class SmsApiError(Exception):
    """Raised when the SMS-API returns a non-200 response."""

    def __init__(self, status_code: int, detail: str) -> None:
        self.status_code = status_code
        self.detail = detail
        super().__init__(f"SMS-API {status_code}: {detail}")


# ── Client ───────────────────────────────────────────────────────────


class SmsApiClient:
    """Thin httpx wrapper for the SMS-API simulation endpoints.

    Default base URL is ``http://localhost:8080`` (stanford-test namespace
    via ``kubectl port-forward`` or ``ptools-proxy.sh``).

    Includes automatic retry for transient 502/504 ALB errors (Pitfall 4
    in sms-api CLAUDE.md — ALB target group flakes after sustained traffic).
    """

    def __init__(
        self,
        base_url: str = DEFAULT_BASE_URL,
        timeout: int = DEFAULT_TIMEOUT,
        max_retries: int = 3,
        retry_delay: float = 5.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.client = httpx.Client(base_url=self.base_url, timeout=timeout)
        self._max_retries = max_retries
        self._retry_delay = retry_delay

    def _request_with_retry(
        self, method: str, path: str, **kwargs: Any
    ) -> httpx.Response:
        """Execute an HTTP request with automatic retry on ALB 502/504."""
        last_exc: Exception | None = None
        for attempt in range(self._max_retries + 1):
            try:
                resp = self.client.request(method, path, **kwargs)
                if resp.status_code in (502, 504) and attempt < self._max_retries:
                    logger.warning(
                        "ALB %d on %s %s (attempt %d/%d, retrying in %.0fs)",
                        resp.status_code, method, path, attempt + 1,
                        self._max_retries + 1, self._retry_delay,
                    )
                    time.sleep(self._retry_delay)
                    continue
                return resp
            except httpx.ReadTimeout as e:
                last_exc = e
                if attempt < self._max_retries:
                    logger.warning(
                        "Timeout on %s %s (attempt %d/%d, retrying)",
                        method, path, attempt + 1, self._max_retries + 1,
                    )
                    time.sleep(self._retry_delay)
                    continue
                raise
        raise last_exc or SmsApiError(504, "Max retries exceeded")  # type: ignore[arg-type]

    def close(self) -> None:
        self.client.close()

    def __enter__(self) -> SmsApiClient:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # ── Simulation lifecycle ──────────────────────────────────────────

    def submit_simulation(
        self,
        simulator_id: int,
        experiment_id: str,
        config_filename: str = "api_simulation_default.json",
        num_generations: int | None = None,
        num_seeds: int | None = None,
        description: str | None = None,
        run_parca: bool = True,
        observables: list[str] | None = None,
        ecoli_sources_repo_url: str | None = None,
        ecoli_sources_ref: str | None = None,
        analysis_options: dict[str, Any] | None = None,
        variants: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """POST /api/v1/simulations — submit a vEcoli workflow.

        Args:
            variants: vEcoli variants config, e.g.
                ``{"sim_data_setattr": {"mutations": {"value": [...]}}}``.
                Passed in the JSON body alongside analysis_options.
                Requires SMS-API v0.8+ (``variants`` body field support).

        Returns the Simulation dict (contains ``database_id``, ``experiment_id``, etc.).
        """
        items: list[tuple[str, str]] = [
            (k, str(v))
            for k, v in {
                "simulator_id": simulator_id,
                "simulation_config_filename": config_filename,
                "experiment_id": experiment_id,
                "num_generations": num_generations,
                "num_seeds": num_seeds,
                "description": description,
                "run_parca": run_parca,
                "ecoli_sources_repo_url": ecoli_sources_repo_url,
                "ecoli_sources_ref": ecoli_sources_ref,
            }.items()
            if v is not None
        ]
        if observables:
            items.extend(("observables", obs) for obs in observables)

        body: dict[str, Any] | None = None
        if analysis_options or variants:
            body = {}
            if analysis_options:
                body["analysis_options"] = analysis_options
            if variants:
                body["variants"] = variants

        resp = self.client.post(
            "/api/v1/simulations",
            params=httpx.QueryParams(items),
            json=body,
        )
        if resp.status_code != 200:
            raise SmsApiError(resp.status_code, resp.text)
        return resp.json()  # type: ignore[no-any-return]

    def get_simulation(self, simulation_id: int) -> dict[str, Any]:
        """GET /api/v1/simulations/{id}."""
        resp = self._request_with_retry("GET", f"/api/v1/simulations/{simulation_id}")
        if resp.status_code != 200:
            raise SmsApiError(resp.status_code, resp.text)
        return resp.json()  # type: ignore[no-any-return]

    def get_status(self, simulation_id: int) -> dict[str, Any]:
        """GET /api/v1/simulations/{id}/status — returns ``{id, status, error_message}``."""
        resp = self._request_with_retry("GET", f"/api/v1/simulations/{simulation_id}/status")
        if resp.status_code != 200:
            raise SmsApiError(resp.status_code, resp.text)
        return resp.json()  # type: ignore[no-any-return]

    def get_log(self, simulation_id: int, truncate: bool = True) -> str:
        """GET /api/v1/simulations/{id}/log."""
        resp = self._request_with_retry(
            "GET",
            f"/api/v1/simulations/{simulation_id}/log",
            params={"truncate": str(truncate).lower()},
        )
        if resp.status_code != 200:
            raise SmsApiError(resp.status_code, resp.text)
        return resp.text

    def cancel_simulation(self, simulation_id: int) -> dict[str, Any]:
        """DELETE /api/v1/simulations/{id}/cancel."""
        resp = self._request_with_retry("DELETE", f"/api/v1/simulations/{simulation_id}/cancel")
        if resp.status_code != 200:
            raise SmsApiError(resp.status_code, resp.text)
        return resp.json()  # type: ignore[no-any-return]

    def download_data(
        self,
        simulation_id: int,
        dest: Path,
        timeout: int = 600,
    ) -> Path:
        """POST /api/v1/simulations/{id}/data — download + extract output tar.gz.

        Returns the root of the extracted archive (the experiment_id directory).
        """
        dest.mkdir(parents=True, exist_ok=True)
        archive_path = dest / f"sim_{simulation_id}.tar.gz"

        # Use a dedicated client with longer timeout for large downloads
        # response_type=streaming avoids ALB 504s on large archives (Pitfall 4)
        with httpx.Client(base_url=self.base_url, timeout=timeout) as dl_client:
            with dl_client.stream(
                "POST",
                f"/api/v1/simulations/{simulation_id}/data",
                params={"response_type": "streaming"},
            ) as resp:
                if resp.status_code != 200:
                    raise SmsApiError(resp.status_code, f"Failed to download data for simulation {simulation_id}")
                with open(archive_path, "wb") as f:
                    for chunk in resp.iter_bytes():
                        f.write(chunk)

        with tarfile.open(archive_path, "r:gz") as tar:
            tar.extractall(dest)  # noqa: S202

        # Find the extracted experiment directory (contains analyses/)
        for child in dest.iterdir():
            if child.is_dir() and (child / "analyses").exists():
                return child

        return dest

    # ── Discovery ─────────────────────────────────────────────────────

    def discover_repo(self, simulator_id: int) -> dict[str, Any]:
        """GET /api/v1/simulations/discovery — list config files and analysis modules."""
        resp = self._request_with_retry(
            "GET", "/api/v1/simulations/discovery",
            params={"simulator_id": simulator_id},
        )
        if resp.status_code != 200:
            raise SmsApiError(resp.status_code, resp.text)
        return resp.json()  # type: ignore[no-any-return]

    def list_simulations(self) -> list[dict[str, Any]]:
        """GET /api/v1/simulations."""
        resp = self._request_with_retry("GET", "/api/v1/simulations")
        if resp.status_code != 200:
            raise SmsApiError(resp.status_code, resp.text)
        return resp.json()  # type: ignore[no-any-return]

    # ── Polling ───────────────────────────────────────────────────────

    def poll_until_complete(
        self,
        simulation_id: int,
        interval: float = POLL_INTERVAL,
        max_duration: float = MAX_POLL_DURATION,
        on_status: Any | None = None,
    ) -> dict[str, Any]:
        """Block until simulation reaches a terminal state.

        Args:
            simulation_id: Database ID of the simulation.
            interval: Seconds between polls.
            max_duration: Max total wait time before raising TimeoutError.
            on_status: Optional callback ``(status_dict) -> None`` called each poll.

        Returns:
            Final status dict.

        Raises:
            TimeoutError: If max_duration exceeded.
            SmsApiError: If simulation fails.
        """
        terminal = {"completed", "failed", "cancelled"}
        start = time.monotonic()

        while True:
            status = self.get_status(simulation_id)
            current = status.get("status", "").lower()

            if on_status is not None:
                on_status(status)

            if current in terminal:
                if current == "failed":
                    raise SmsApiError(
                        500,
                        f"Simulation {simulation_id} failed: {status.get('error_message', 'unknown')}",
                    )
                return status

            elapsed = time.monotonic() - start
            if elapsed > max_duration:
                raise TimeoutError(
                    f"Simulation {simulation_id} did not complete within {max_duration}s "
                    f"(last status: {current})"
                )

            logger.info(
                "Simulation %d status: %s (%.0fs elapsed)", simulation_id, current, elapsed
            )
            time.sleep(interval)

    # ── Batch helpers ─────────────────────────────────────────────────

    def submit_batch(
        self,
        simulator_id: int,
        experiment_ids: list[str],
        config_filename: str = "api_simulation_default.json",
        num_generations: int | None = None,
        num_seeds: int | None = None,
        run_parca: bool = True,
        observables: list[str] | None = None,
        ecoli_sources_repo_url: str | None = None,
        ecoli_sources_ref: str | None = None,
        analysis_options: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Submit multiple simulations (one per experiment_id).

        Returns list of Simulation dicts.
        """
        simulations = []
        for exp_id in experiment_ids:
            sim = self.submit_simulation(
                simulator_id=simulator_id,
                experiment_id=exp_id,
                config_filename=config_filename,
                num_generations=num_generations,
                num_seeds=num_seeds,
                run_parca=run_parca,
                observables=observables,
                ecoli_sources_repo_url=ecoli_sources_repo_url,
                ecoli_sources_ref=ecoli_sources_ref,
                analysis_options=analysis_options,
            )
            simulations.append(sim)
        return simulations

    def poll_batch(
        self,
        simulation_ids: list[int],
        interval: float = POLL_INTERVAL,
        max_duration: float = MAX_POLL_DURATION,
        on_status: Any | None = None,
    ) -> list[dict[str, Any]]:
        """Poll multiple simulations until all reach terminal state.

        Returns list of final status dicts.
        """
        terminal = {"completed", "failed", "cancelled"}
        pending = set(simulation_ids)
        results: dict[int, dict[str, Any]] = {}
        start = time.monotonic()

        while pending:
            for sim_id in list(pending):
                status = self.get_status(sim_id)
                current = status.get("status", "").lower()

                if on_status is not None:
                    on_status(sim_id, status)

                if current in terminal:
                    results[sim_id] = status
                    pending.discard(sim_id)

            if pending:
                elapsed = time.monotonic() - start
                if elapsed > max_duration:
                    raise TimeoutError(
                        f"{len(pending)} simulations did not complete within {max_duration}s: "
                        f"{sorted(pending)}"
                    )
                logger.info(
                    "%d/%d simulations pending (%.0fs elapsed)",
                    len(pending), len(simulation_ids), elapsed,
                )
                time.sleep(interval)

        return [results[sid] for sid in simulation_ids]
