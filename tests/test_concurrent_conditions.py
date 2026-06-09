"""Tests for §25-F concurrent design-condition execution.

Covers:
  - ``_safe_condition_concurrency`` constraint math
  - ``CondResult`` dataclass contract
  - ``_condition_worker._read_log_tail`` IO behavior

End-to-end orchestration (``ProcessPoolExecutor`` over actual workers)
is exercised by integration runs against real ``simData.cPickle``;
the helpers covered here are the pure pieces that determine concurrency
and failure-isolation behavior independent of vEcoli.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from uq._condition_worker import CondResult, _read_log_tail
from uq.vecoli_config import _safe_condition_concurrency


# ── _safe_condition_concurrency ───────────────────────────────────────


def test_concurrency_capped_at_n_conditions() -> None:
    """Never run more workers than there are conditions."""
    n = _safe_condition_concurrency(
        n_conditions=3, max_user=100, memory_per_workflow_gb=0.001,
        cpus_per_workflow=1, hard_ceiling=100,
    )
    assert n == 3


def test_concurrency_respects_max_user() -> None:
    n = _safe_condition_concurrency(
        n_conditions=100, max_user=2, memory_per_workflow_gb=0.001,
        cpus_per_workflow=1, hard_ceiling=100,
    )
    assert n == 2


def test_concurrency_respects_hard_ceiling() -> None:
    n = _safe_condition_concurrency(
        n_conditions=100, max_user=None, memory_per_workflow_gb=0.001,
        cpus_per_workflow=1, hard_ceiling=2,
    )
    assert n == 2


def test_concurrency_respects_cpu_constraint() -> None:
    """cpus_per_workflow=999999 forces a 1-wide cpu limit on any machine."""
    n = _safe_condition_concurrency(
        n_conditions=100, max_user=None, memory_per_workflow_gb=0.001,
        cpus_per_workflow=999_999, hard_ceiling=100,
    )
    assert n == 1


def test_concurrency_at_least_one() -> None:
    """Even with absurd memory_per_workflow_gb, we return ≥ 1."""
    n = _safe_condition_concurrency(
        n_conditions=10, max_user=None, memory_per_workflow_gb=1e9,
        cpus_per_workflow=999_999, hard_ceiling=100,
    )
    assert n >= 1


def test_concurrency_max_user_zero_treated_as_unset() -> None:
    """max_user=0 should not silently force n=0; defer to other constraints."""
    n = _safe_condition_concurrency(
        n_conditions=5, max_user=0, memory_per_workflow_gb=0.001,
        cpus_per_workflow=1, hard_ceiling=100,
    )
    assert n >= 1


def test_concurrency_unfocused_max_user_one_forces_serial() -> None:
    """max_user=1 means user wants strict serial."""
    n = _safe_condition_concurrency(
        n_conditions=10, max_user=1, memory_per_workflow_gb=0.001,
        cpus_per_workflow=1, hard_ceiling=100,
    )
    assert n == 1


# ── CondResult dataclass contract ─────────────────────────────────────


def test_condresult_required_fields() -> None:
    r = CondResult(
        j=3,
        condition_id="mod_0003_abc123",
        status="ok",
        n_samples_cached=30,
        wall_clock_seconds=42.5,
        log_tail="last log line\n",
        error=None,
    )
    assert r.j == 3
    assert r.condition_id == "mod_0003_abc123"
    assert r.status == "ok"
    assert r.n_samples_cached == 30
    assert r.wall_clock_seconds == pytest.approx(42.5)
    assert r.error is None


def test_condresult_failed_with_error() -> None:
    r = CondResult(
        j=0, condition_id="x", status="failed", n_samples_cached=0,
        wall_clock_seconds=1.0, log_tail="", error="Traceback...",
    )
    assert r.status == "failed"
    assert r.error and r.error.startswith("Traceback")


# ── _read_log_tail ─────────────────────────────────────────────────────


def test_read_log_tail_returns_empty_for_missing_file(tmp_path: Path) -> None:
    assert _read_log_tail(tmp_path / "nope.log") == ""


def test_read_log_tail_returns_last_n_lines(tmp_path: Path) -> None:
    log = tmp_path / "x.log"
    log.write_text("\n".join(f"line {i}" for i in range(100)) + "\n")
    tail = _read_log_tail(log, n_lines=10)
    lines = tail.strip().splitlines()
    assert len(lines) == 10
    assert lines[-1] == "line 99"
    assert lines[0] == "line 90"


def test_read_log_tail_handles_short_file(tmp_path: Path) -> None:
    log = tmp_path / "short.log"
    log.write_text("only line 1\nonly line 2\n")
    tail = _read_log_tail(log, n_lines=50)
    assert "only line 1" in tail
    assert "only line 2" in tail


# ── _estimate_resources ────────────────────────────────────────────────


def test_estimate_resources_basic_shape() -> None:
    from uq.vecoli_config import _estimate_resources
    est = _estimate_resources(
        n_conditions=12, n_samples=30, n_init_sims=4, generations=4,
        observables=["mass"], max_condition_parallel=4,
        memory_per_workflow_gb=3.0, wall_clock_per_sim_seconds=30.0,
    )
    # 30 × 4 × 4 = 480 sims/condition × 12 = 5,760 total
    assert est.sims_per_condition == 480
    assert est.total_sims == 5760


def test_estimate_resources_ram_peak() -> None:
    """RAM peak = max_concurrent × per_workflow + parent_overhead."""
    from uq.vecoli_config import _estimate_resources
    est = _estimate_resources(
        n_conditions=12, n_samples=30, n_init_sims=4, generations=4,
        observables=["mass"], max_condition_parallel=4,
        memory_per_workflow_gb=3.0, parent_ram_overhead_gb=1.0,
    )
    # 4 × 3 + 1 = 13 GB
    assert est.ram_peak_gb == pytest.approx(13.0)


def test_estimate_resources_disk_scales_with_observables() -> None:
    """Heavier observable presets produce larger disk estimates."""
    from uq.vecoli_config import _estimate_resources

    mass = _estimate_resources(
        n_conditions=12, n_samples=30, n_init_sims=4, generations=4,
        observables=["mass"], max_condition_parallel=4,
    )
    txp = _estimate_resources(
        n_conditions=12, n_samples=30, n_init_sims=4, generations=4,
        observables=["transcriptome"], max_condition_parallel=4,
    )
    # transcriptome is ~80 MB/sim vs mass's ~3 MB/sim → roughly 25× more
    assert txp.disk_total_gb > 10 * mass.disk_total_gb


def test_estimate_resources_multiple_observables_sum() -> None:
    from uq.vecoli_config import _estimate_resources, OBSERVABLE_DISK_MB_PER_SIM

    multi = _estimate_resources(
        n_conditions=1, n_samples=10, n_init_sims=1, generations=1,
        observables=["mass", "exchange_fluxes"], max_condition_parallel=1,
    )
    # Sum of per-sim MB = 3 + 15 = 18; sims = 10 → 180 MB parquet + 180 MB scratch + 200 MB pickle
    expected_mb = 200.0 + 10 * 18.0 * 2  # pickle + (parquet + scratch)
    assert multi.disk_per_condition_gb == pytest.approx(expected_mb / 1024.0, rel=0.01)


def test_estimate_resources_wall_clock_batches() -> None:
    """Wall-clock = ceil(M / max_concurrent) × per-condition wall."""
    from uq.vecoli_config import _estimate_resources

    # 12 conditions, 4 concurrent, 100 sims/cond, 30s/sim, intra_workflow=2
    # Per-condition wall = 100 × 30 / 2 = 1500s
    # n_batches = ceil(12 / 4) = 3
    # Total = 3 × 1500 = 4500s = 1.25h
    est = _estimate_resources(
        n_conditions=12, n_samples=5, n_init_sims=4, generations=5,
        observables=["mass"], max_condition_parallel=4,
        wall_clock_per_sim_seconds=30.0, intra_workflow_concurrency=2,
    )
    assert est.sims_per_condition == 100
    assert est.wall_clock_hours == pytest.approx(1.25, abs=0.01)


def test_estimate_resources_serial_falls_back_to_total_time() -> None:
    """With max_condition_parallel=1, wall clock = M × per-condition wall."""
    from uq.vecoli_config import _estimate_resources
    est = _estimate_resources(
        n_conditions=3, n_samples=5, n_init_sims=1, generations=1,
        observables=["mass"], max_condition_parallel=1,
        wall_clock_per_sim_seconds=60.0, intra_workflow_concurrency=1,
    )
    # 5 sims/cond × 60s = 300s per condition × 3 conditions = 900s = 0.25h
    assert est.wall_clock_hours == pytest.approx(0.25, abs=0.01)


def test_estimate_resources_unknown_observable_uses_default() -> None:
    from uq.vecoli_config import _estimate_resources
    est = _estimate_resources(
        n_conditions=1, n_samples=10, n_init_sims=1, generations=1,
        observables=["not_a_real_preset"], max_condition_parallel=1,
    )
    # Default fallback is 10 MB/sim
    expected = (200.0 + 10 * 10.0 * 2) / 1024.0
    assert est.disk_per_condition_gb == pytest.approx(expected, rel=0.01)


def test_estimate_resources_verdicts(tmp_path) -> None:
    """RAM verdict reflects ram_available vs peak; disk reflects free vs total."""
    from uq.vecoli_config import _estimate_resources
    est = _estimate_resources(
        n_conditions=1, n_samples=1, n_init_sims=1, generations=1,
        observables=["mass"], max_condition_parallel=1,
        memory_per_workflow_gb=0.001,  # trivially small to ensure ram_verdict=fits
        cache_dir=str(tmp_path),
    )
    # On any normal dev host, this should fit (RAM peak ~ 0.001 + 1 GB ≈ 1 GB)
    assert est.ram_verdict in {"fits", "tight"}
    assert est.disk_verdict in {"fits", "tight"}
    assert est.cache_dir_checked is not None


def test_estimate_resources_handles_missing_cache_dir(tmp_path) -> None:
    """Cache dir that doesn't yet exist still gets disk-available from parent."""
    from uq.vecoli_config import _estimate_resources
    nonexistent = tmp_path / "deeply" / "nested" / "future_cache"
    est = _estimate_resources(
        n_conditions=1, n_samples=1, n_init_sims=1, generations=1,
        observables=["mass"], max_condition_parallel=1,
        cache_dir=str(nonexistent),
    )
    assert est.disk_available_gb is not None  # walked up to an existing parent


def test_estimate_resources_no_cache_dir_skips_disk_avail() -> None:
    from uq.vecoli_config import _estimate_resources
    est = _estimate_resources(
        n_conditions=1, n_samples=1, n_init_sims=1, generations=1,
        observables=["mass"], max_condition_parallel=1,
        cache_dir=None,
    )
    assert est.disk_available_gb is None
    assert est.disk_verdict == "unknown"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
