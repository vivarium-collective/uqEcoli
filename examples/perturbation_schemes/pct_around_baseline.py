"""Example perturbation scheme: ±30% around each parameter's baseline value.

Reads the current baseline ``sim_data`` value for each attr_path and emits a
``SimDataParameter`` with bounds = ``baseline × (1 ± pct)``. Useful as a
quick-start when you don't have literature ranges handy — gives PCRV a
reasonable box to explore around the parameter's nominal point.

Pair with ``uq create mutations`` via::

    uq create mutations sim_data/baseline/kb/simData.cPickle \\
        --output ./my_params.json \\
        --perturbation-scheme examples/perturbation_schemes/pct_around_baseline.py
"""

from libuq.pipeline.models import SimDataParameter

# Tunable knob — change here or fork the file for a different range.
PCT = 0.30

# Attribute paths to perturb. Picked from the canonical six physiological
# parameters; extend / replace for your study.
ATTR_PATHS: list[str] = [
    "process.transcription.fraction_active_rnap_free",
    "process.transcription.fraction_active_rnap_bound",
    "process.translation.basal_elongation_rate",
    "process.metabolism.kinetic_objective_weight",
    "process.metabolism.secretion_penalty_coeff",
    "mass.cell_dry_mass_fraction",
]


def build_parameters(sim_data, n_samples: int = 0, seed: int = 42) -> list[SimDataParameter]:
    """Read baseline values from ``sim_data`` and emit ±PCT bounds.

    The ``n_samples`` and ``seed`` kwargs are accepted but unused — this
    scheme is deterministic, only the parameter geometry changes.
    """
    params: list[SimDataParameter] = []
    for attr_path in ATTR_PATHS:
        baseline = _read_dotted(sim_data, attr_path)
        if not isinstance(baseline, (int, float)) or baseline == 0:
            # Skip non-numeric and zero-baseline paths (would give bounds=[0,0])
            continue
        lo = baseline * (1.0 - PCT)
        hi = baseline * (1.0 + PCT)
        # Preserve direction when baseline is negative.
        if lo > hi:
            lo, hi = hi, lo
        params.append(
            SimDataParameter(
                name=attr_path.split(".")[-1],
                attr_path=attr_path,
                bounds=(float(lo), float(hi)),
                description=(
                    f"±{int(PCT * 100)}% around baseline {baseline:g} "
                    "(perturbation scheme: pct_around_baseline)"
                ),
            )
        )
    return params


def _read_dotted(obj, attr_path: str):
    """Walk ``obj`` along a dot-path and return the leaf value."""
    for part in attr_path.split("."):
        obj = getattr(obj, part)
    return obj
