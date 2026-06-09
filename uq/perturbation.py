"""Perturbation-scheme loader for ``uq create mutations``.

A perturbation scheme defines *which* ``sim_data`` attributes the UQ
pipeline will perturb, and *what bounds* PCRV will sample within. By
keeping this decision in user-supplied Python (not just a JSON file), we
give scientists a place to encode literature ranges, ±X%-around-baseline
heuristics, knockout filters, or any other domain-specific selection
logic — separate from the pipeline orchestration.

Scheme contract — a scheme module must expose one of:

  - ``PARAMETERS: list[SimDataParameter]``  (static list)
  - ``build_parameters(sim_data, n_samples=0, seed=42) -> list[SimDataParameter]``

``build_parameters`` is checked first when present; falling back to
``PARAMETERS``. Schemes that ignore ``n_samples`` / ``seed`` may omit them
from their signature — we introspect via ``inspect.signature`` and pass
only the kwargs the scheme accepts.

When no scheme is provided, ``DEFAULT_SIM_DATA_PARAMETERS`` (the six
physiological knobs the rest of the framework defaults to) is used.

Example minimal scheme::

    # my_scheme.py
    from libuq.pipeline.models import SimDataParameter

    PARAMETERS = [
        SimDataParameter(
            name="elongation_rate",
            attr_path="process.translation.basal_elongation_rate",
            bounds=(15.0, 28.0),
            description="±30% around baseline 22 aa/s",
        ),
    ]
"""

from __future__ import annotations

import importlib
import importlib.util
import inspect
from pathlib import Path
from typing import Any, Callable

from libuq.pipeline.models import SimDataParameter
from libuq.pipeline.param_loader import DEFAULT_SIM_DATA_PARAMETERS


def load_perturbation_scheme(spec: str | None):
    """Load a perturbation-scheme module from a file path or dotted module name.

    Args:
        spec: ``None`` → returns ``None`` (caller should fall back to
            defaults). A path ending in ``.py`` (or any string pointing
            at an existing file) is loaded via ``importlib.util.spec_from_file_location``.
            Otherwise the string is treated as a dotted module path
            (``my_pkg.schemes.foo``) and loaded via ``importlib.import_module``.

    Returns:
        The loaded module (or ``None`` when ``spec`` is ``None``). Caller
        should then call :func:`build_parameters_from_scheme` to extract
        the parameter list.

    Raises:
        ImportError: When the spec is neither a readable file nor an
            importable dotted module.
    """
    if spec is None:
        return None

    p = Path(spec)
    if p.exists() and p.is_file():
        module_name = f"_uq_user_scheme_{p.stem}"
        spec_loader = importlib.util.spec_from_file_location(module_name, str(p))
        if spec_loader is None or spec_loader.loader is None:
            raise ImportError(f"Could not load perturbation scheme from {p!r}.")
        module = importlib.util.module_from_spec(spec_loader)
        spec_loader.loader.exec_module(module)
        return module

    try:
        return importlib.import_module(spec)
    except ImportError as exc:
        raise ImportError(
            f"Could not load --perturbation-scheme {spec!r}: not a file path "
            "and not an importable dotted module. Pass a path to a .py file "
            "or a module path like 'my_pkg.schemes.literature_ranges'."
        ) from exc


def build_parameters_from_scheme(
    scheme,
    sim_data: Any,
    n_samples: int = 0,
    seed: int = 42,
) -> list[SimDataParameter]:
    """Extract ``SimDataParameter`` specs from a loaded scheme module.

    Tries ``scheme.build_parameters(...)`` first (passing only the kwargs
    its signature accepts), then falls back to ``scheme.PARAMETERS``.
    Returns ``DEFAULT_SIM_DATA_PARAMETERS`` when ``scheme`` is ``None``.

    Args:
        scheme: Loaded module from :func:`load_perturbation_scheme`, or
            ``None`` to use the framework defaults.
        sim_data: Loaded baseline ``SimulationDataEcoli`` — passed to
            dynamic schemes so they can read baseline values for
            ±X%-around-baseline computations.
        n_samples: Forwarded to dynamic schemes that accept the kwarg.
        seed: Forwarded to dynamic schemes that accept the kwarg.

    Returns:
        List of validated-shape ``SimDataParameter`` specs.

    Raises:
        ValueError: When ``scheme`` exists but exposes neither
            ``build_parameters`` nor ``PARAMETERS``.
        TypeError: When ``scheme.build_parameters`` returns the wrong type.
    """
    if scheme is None:
        return list(DEFAULT_SIM_DATA_PARAMETERS)

    builder: Callable[..., list[SimDataParameter]] | None = getattr(
        scheme, "build_parameters", None,
    )
    if builder is not None:
        accepted_kwargs = _filter_kwargs(
            builder, n_samples=n_samples, seed=seed,
        )
        result = builder(sim_data, **accepted_kwargs)
        if not isinstance(result, list):
            raise TypeError(
                f"Perturbation scheme {scheme.__name__}.build_parameters "
                f"must return list[SimDataParameter]; got {type(result).__name__}."
            )
        return list(result)

    static = getattr(scheme, "PARAMETERS", None)
    if static is not None:
        if not isinstance(static, list):
            raise TypeError(
                f"Perturbation scheme {scheme.__name__}.PARAMETERS must be a "
                f"list[SimDataParameter]; got {type(static).__name__}."
            )
        return list(static)

    raise ValueError(
        f"Perturbation scheme {scheme.__name__!r} exposes neither "
        "build_parameters(sim_data, ...) nor PARAMETERS = [...]. Define "
        "one of these to make it loadable. See uq/perturbation.py's "
        "module docstring for the contract."
    )


def _filter_kwargs(fn: Callable[..., Any], **candidates: Any) -> dict[str, Any]:
    """Return the subset of candidates that the function's signature accepts.

    Lets perturbation schemes adopt the kwargs they need without having to
    declare unused ones. Tolerates ``**kwargs`` (passes everything in that
    case).
    """
    try:
        sig = inspect.signature(fn)
    except (TypeError, ValueError):
        return dict(candidates)
    params = sig.parameters
    if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values()):
        return dict(candidates)
    accepted = {k: v for k, v in candidates.items() if k in params}
    return accepted
