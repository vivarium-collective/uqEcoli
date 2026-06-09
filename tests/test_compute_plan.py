"""Tests for `uq.vecoli_config._recommend_compute_allocation`.

Triage 6 / Gap 6 — recommend (n_samples, n_init_sims, generations) for a
given compute budget so the user can size a run before committing it to
vEcoli.
"""

from __future__ import annotations

import pytest

from uq.vecoli_config import _pce_basis_size, _recommend_compute_allocation


def test_recommended_is_first_in_list() -> None:
    """The first candidate is always labeled `recommended`."""
    cands = _recommend_compute_allocation(
        budget=1000, n_params=6, polynomial_order=2,
        min_replicates=4, generations=4,
    )
    assert cands[0].label == "recommended"


def test_recommended_uses_requested_replicates_and_generations() -> None:
    cands = _recommend_compute_allocation(
        budget=1000, n_params=6, polynomial_order=2,
        min_replicates=4, generations=4,
    )
    rec = cands[0]
    assert rec.n_init_sims == 4
    assert rec.generations == 4
    # n_samples derived from budget: 1000 // (4 × 4) = 62
    assert rec.n_samples == 62
    # And total runs are within budget
    assert rec.total_runs <= 1000


def test_adequacy_ratio_matches_basis() -> None:
    cands = _recommend_compute_allocation(
        budget=2000, n_params=6, polynomial_order=2,
        min_replicates=4, generations=4,
    )
    rec = cands[0]
    basis = _pce_basis_size(2, 6)  # = 28
    assert rec.basis_size == basis
    assert rec.adequacy_ratio == pytest.approx(rec.n_samples / basis)


def test_underdetermined_flag_at_low_budget() -> None:
    """Budget so small that n_samples < basis_size."""
    cands = _recommend_compute_allocation(
        budget=100, n_params=6, polynomial_order=2,
        min_replicates=4, generations=4,
    )
    # 100 // 16 = 6 samples, basis=28 → underdetermined
    assert cands[0].adequacy_status == "underdetermined"


def test_ok_status_at_ample_budget() -> None:
    cands = _recommend_compute_allocation(
        budget=5000, n_params=6, polynomial_order=2,
        min_replicates=4, generations=4,
    )
    # 5000 // 16 = 312 samples, basis=28 → ratio 11× → ok
    assert cands[0].adequacy_status == "ok"


def test_alternative_halve_generations_doubles_samples() -> None:
    cands = _recommend_compute_allocation(
        budget=1000, n_params=6, polynomial_order=2,
        min_replicates=4, generations=4,
    )
    alt = next(c for c in cands if c.label == "halve generations")
    assert alt.generations == 2
    # 1000 // (4 × 2) = 125 samples ≈ 2× the recommended 62
    assert alt.n_samples == 125
    assert alt.n_samples > cands[0].n_samples


def test_alternative_single_replicate_kills_noise_estimability() -> None:
    cands = _recommend_compute_allocation(
        budget=1000, n_params=6, polynomial_order=2,
        min_replicates=4, generations=4,
    )
    alt = next(c for c in cands if c.label == "single replicate")
    assert alt.n_init_sims == 1
    assert not alt.noise_estimable
    # Should give the largest n_samples of the three options
    assert alt.n_samples >= max(c.n_samples for c in cands if c.label != "single replicate")


def test_no_single_replicate_alternative_when_already_at_one() -> None:
    """If user requests min_replicates=1, don't emit the redundant alternative."""
    cands = _recommend_compute_allocation(
        budget=1000, n_params=6, polynomial_order=2,
        min_replicates=1, generations=4,
    )
    labels = [c.label for c in cands]
    assert "single replicate" not in labels


def test_no_halve_alternative_when_generations_is_one() -> None:
    cands = _recommend_compute_allocation(
        budget=1000, n_params=6, polynomial_order=2,
        min_replicates=4, generations=1,
    )
    labels = [c.label for c in cands]
    assert "halve generations" not in labels


def test_higher_order_increases_basis_and_lowers_ratio() -> None:
    same_budget = 2000
    p2 = _recommend_compute_allocation(
        budget=same_budget, n_params=6, polynomial_order=2,
        min_replicates=4, generations=4,
    )[0]
    p3 = _recommend_compute_allocation(
        budget=same_budget, n_params=6, polynomial_order=3,
        min_replicates=4, generations=4,
    )[0]
    # Same n_samples (budget split unchanged), bigger basis at p=3 → ratio drops
    assert p3.basis_size > p2.basis_size
    assert p3.adequacy_ratio < p2.adequacy_ratio


# ── Multi-condition planning (--design-config support) ───────────────


def test_recommend_divides_budget_across_conditions() -> None:
    """Under n_conditions=M, per-condition n_samples = budget/(M × replicates × gens)."""
    cands = _recommend_compute_allocation(
        budget=12000, n_params=6, polynomial_order=2,
        min_replicates=4, generations=4, n_conditions=12,
    )
    rec = cands[0]
    # per-condition budget = 12000 / 12 = 1000; n_samples = 1000 // 16 = 62
    assert rec.n_samples == 62
    assert rec.n_conditions == 12
    assert rec.total_runs == 12 * 62 * 4 * 4  # M × N × S × G


def test_recommend_per_condition_status_at_marginal() -> None:
    """Budget = 12 × 30 × 16 = 5,760 → per-condition n_samples=30 → ratio 1.07× → marginal."""
    cands = _recommend_compute_allocation(
        budget=5760, n_params=6, polynomial_order=2,
        min_replicates=4, generations=4, n_conditions=12,
    )
    assert cands[0].n_samples == 30
    assert cands[0].adequacy_status == "marginal"
    assert 1.0 <= cands[0].adequacy_ratio < 2.0


def test_recommend_single_condition_unchanged() -> None:
    """n_conditions=1 should behave exactly like the original single-condition path."""
    multi = _recommend_compute_allocation(
        budget=1000, n_params=6, polynomial_order=2,
        min_replicates=4, generations=4, n_conditions=1,
    )
    single = _recommend_compute_allocation(
        budget=1000, n_params=6, polynomial_order=2,
        min_replicates=4, generations=4,  # n_conditions default = 1
    )
    assert multi[0].n_samples == single[0].n_samples
    assert multi[0].total_runs == single[0].total_runs


def test_recommend_total_runs_includes_conditions_factor() -> None:
    """total_runs accounts for n_conditions, n_samples, n_init_sims, generations."""
    cands = _recommend_compute_allocation(
        budget=12000, n_params=6, polynomial_order=2,
        min_replicates=4, generations=4, n_conditions=12,
    )
    rec = cands[0]
    assert rec.total_runs == rec.n_conditions * rec.n_samples * rec.n_init_sims * rec.generations


def test_recommend_alternatives_also_apply_per_condition() -> None:
    """halve generations + single replicate alternatives also respect n_conditions."""
    cands = _recommend_compute_allocation(
        budget=12000, n_params=6, polynomial_order=2,
        min_replicates=4, generations=4, n_conditions=12,
    )
    rec = next(c for c in cands if c.label == "recommended")
    halve = next(c for c in cands if c.label == "halve generations")
    single = next(c for c in cands if c.label == "single replicate")

    # halve generations: per-condition budget 1000 → 1000//(4×2) = 125 samples
    assert halve.n_samples == 125
    # single replicate: per-condition budget 1000 → 1000//4 = 250 samples
    assert single.n_samples == 250
    # Total runs all include the M factor
    assert halve.total_runs == 12 * 125 * 4 * 2
    assert single.total_runs == 12 * 250 * 1 * 4
    # PCE quality moves the same direction as without n_conditions
    assert halve.n_samples > rec.n_samples
    assert single.n_samples > halve.n_samples


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
