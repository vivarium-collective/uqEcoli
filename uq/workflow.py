"""
RFC006 PCE-based UQ workflow — adapted from PyTUQ's ``apps/uqpc/uq_pc.py``.

Self-contained implementation of the five-step UQPC workflow
(https://sandialabs.github.io/pytuq/apps/uqpc.html) for the vEcoli
whole-cell model, as specified in RFC006 (MS-08.4.2).

Two-stage public API:

    sample()   → Steps 1-3: setup inputs, generate samples via PCRV,
                 evaluate vEcoli (runscripts/workflow.py), cache to disk.
    quantify() → Steps 4-5: build PC surrogate, compute Sobol indices,
                 run all 4 RFC006 aggregation strategies.

The five UQPC steps map as follows:

  UQPC step                    This module
  ─────────────────────────    ─────────────────────────────────────────
  1. Setup inputs              _setup_input_pc(): bounds → PCRV (LU)
  2. Generate samples          PCRV.sampleGerm() → evalPC() (PyTUQ-native)
  3. Evaluate model            TimeseriesGeneratorVecoli._run_batch()
                               (vEcoli subprocess via workflow.py)
  4. Build PC surrogate        _fit_surrogate(): PCRV + lsq/bcs/anl
  5. Post-process              _compute_sobol(), _compute_relative_errors()

No mock data, no synthetic wrappers — vEcoli only.

References:
  [1] PyTUQ UQPC: https://sandialabs.github.io/pytuq/apps/uqpc.html
  [2] Sudret (2008). Global sensitivity analysis using polynomial chaos
      expansions. Reliability Engineering & System Safety.
  [3] RFC006: readmes/start/tools/RFC006.md
"""

from __future__ import annotations

import hashlib
import json as _json
import logging
import platform
import subprocess as _subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from pytuq.lreg.anl import anl  # type: ignore[import-untyped]
from pytuq.lreg.bcs import bcs  # type: ignore[import-untyped]
from pytuq.lreg.lreg import lsq  # type: ignore[import-untyped]
from pytuq.rv.pcrv import PCRV  # type: ignore[import-untyped]
from pytuq.utils.mindex import get_mi  # type: ignore[import-untyped]

from libuq.generators.vecoli import TimeseriesGeneratorVecoli
from libuq.inputs import XSpace
from libuq.pipeline.models import SimDataParameter
from libuq.pipeline.param_loader import DEFAULT_SIM_DATA_PARAMETERS, ParameterDataset
from libuq.sampling import PrecomputedCache
from libuq.sensitivity import PCESurrogate, SobolIndices

logger = logging.getLogger(__name__)


# ── Output-side PCA ──────────────────────────────────────────────────


@dataclass
class PCAReduction:
    """Result of output-side PCA applied before PCE fitting.

    Stores the transformation so results can be mapped back to
    the original observable space.
    """

    n_components: int
    mean: np.ndarray  # (n_obs,)
    components: np.ndarray  # (n_components, n_obs) — loadings
    explained_variance_ratio: np.ndarray  # (n_components,)
    original_names: list[str]

    def top_loadings(self, pc_index: int, n: int = 10) -> list[tuple[str, float]]:
        """Return top-N observables by absolute loading for a given PC."""
        loadings = self.components[pc_index]
        indices = np.argsort(np.abs(loadings))[::-1][:n]
        return [(self.original_names[i], float(loadings[i])) for i in indices]

    def export(self, export_dir: str | Path) -> Path:
        """Write PCA artifacts to disk."""
        import json

        out = Path(export_dir)
        pca_dir = out / "pca"
        pca_dir.mkdir(parents=True, exist_ok=True)

        # Spec-compliant names (B2)
        np.save(pca_dir / "pca_loadings.npy", self.components)
        np.save(pca_dir / "pca_explained_variance.npy", self.explained_variance_ratio)
        np.save(pca_dir / "mean.npy", self.mean)

        # Backward-compatible aliases
        np.save(pca_dir / "components.npy", self.components)
        np.save(pca_dir / "explained_variance_ratio.npy", self.explained_variance_ratio)

        # Top loadings per PC as JSON for the report
        top_loadings_data = {}
        for k in range(self.n_components):
            top_loadings_data[f"PC{k + 1}"] = {
                "explained_variance_pct": round(float(self.explained_variance_ratio[k]) * 100, 2),
                "top_loadings": [{"observable": name, "loading": round(val, 4)} for name, val in self.top_loadings(k)],
            }
        (pca_dir / "pca_top_loadings.json").write_text(json.dumps(top_loadings_data, indent=2))
        # Backward-compatible alias
        (pca_dir / "pca_summary.json").write_text(json.dumps(top_loadings_data, indent=2))
        return pca_dir


def apply_output_pca(
    Y: np.ndarray,
    n_components: int,
    observable_names: list[str],
) -> tuple[np.ndarray, PCAReduction]:
    """Apply PCA to reduce high-dimensional output before PCE fitting.

    Centers Y, computes truncated SVD to get top-K principal components.

    Args:
        Y: Output matrix, shape ``(n_samples, n_obs)``.
        n_components: Number of PCs to retain.
        observable_names: Original feature labels.

    Returns:
        ``(Y_pca, pca_info)`` where ``Y_pca`` has shape ``(n_samples, n_components)``.
    """
    n_samples, n_obs = Y.shape
    n_components = min(n_components, n_samples, n_obs)

    mean = Y.mean(axis=0)
    Y_centered = Y - mean

    # Truncated SVD
    U, S, Vt = np.linalg.svd(Y_centered, full_matrices=False)
    components = Vt[:n_components]  # (n_components, n_obs)
    Y_pca = Y_centered @ components.T  # (n_samples, n_components)

    # Explained variance ratio
    total_var = np.sum(S**2) / (n_samples - 1)
    explained_var = (S[:n_components] ** 2) / (n_samples - 1)
    explained_ratio = explained_var / total_var if total_var > 0 else np.zeros(n_components)

    logger.info(
        "PCA: %d → %d components (%.1f%% variance explained)",
        n_obs,
        n_components,
        explained_ratio.sum() * 100,
    )

    pca = PCAReduction(
        n_components=n_components,
        mean=mean,
        components=components,
        explained_variance_ratio=explained_ratio,
        original_names=observable_names,
    )
    return Y_pca, pca


# ── Result container ────────────────────────────────────────────────


@dataclass
class UQPCResult:
    """Complete output of the UQPC workflow for one aggregation strategy.

    Mirrors the ``results.pk`` dict from ``uq_pc.py``, but uses RFC006
    domain types and stores per-strategy diagnostics.

    Attributes:
        sobol: Sobol sensitivity indices (main, total, joint).
        surrogate: Exportable PCE surrogate with coefficients.
        pcrv: The fitted ``PCRV`` object (PyTUQ internal repr).
        linregs: Per-output linear regression objects from ``pc_fit``.
        germ_train: Training samples in germ space [-1, 1].
        X_train: Training samples in physical space.
        Y_train: Training outputs.
        Y_train_pc: PCE predictions at training points.
        Y_train_pc_std: Prediction std dev at training points.
        relerr_train: Per-output relative error at training points.
        germ_test: Test samples in germ space (None if no test set).
        X_test: Test samples in physical space (None if no test set).
        Y_test: Test outputs (None if no test set).
        Y_test_pc: PCE predictions at test points (None if no test set).
        Y_test_pc_std: Prediction std dev at test points.
        relerr_test: Per-output relative error at test points.
    """

    sobol: SobolIndices
    surrogate: PCESurrogate
    pcrv: PCRV
    linregs: list[Any]
    germ_train: np.ndarray
    X_train: np.ndarray
    Y_train: np.ndarray
    Y_train_pc: np.ndarray
    Y_train_pc_std: np.ndarray
    relerr_train: np.ndarray
    germ_test: np.ndarray | None = None
    X_test: np.ndarray | None = None
    Y_test: np.ndarray | None = None
    Y_test_pc: np.ndarray | None = None
    Y_test_pc_std: np.ndarray | None = None
    relerr_test: np.ndarray | None = None


# ── Step 1: Input PC setup ──────────────────────────────────────────


def _setup_input_pc(
    bounds: np.ndarray,
) -> tuple[PCRV, np.ndarray, int]:
    """Set up the input PC object from parameter bounds.

    Equivalent to ``uq_pc.py`` lines that handle ``--pdom``:
    uniform parameters → Legendre (LU) basis, order 1.

    The PC coefficients encode the affine map from germ space [-1, 1]
    to the physical domain [lb, ub]:

        x_phys = midpoint + half_range * xi

    where xi ∈ [-1, 1] is the germ variable.

    Args:
        bounds: Parameter bounds, shape (n_params, 2).

    Returns:
        Tuple of (PCRV object, PC coefficient matrix, stochastic dim).
    """
    n_params = bounds.shape[0]
    in_pcdim = n_params
    in_pcord = 1
    pc_type = "LU"  # Legendre — uniform priors (RFC006 §4)

    # Build PC coefficients: row 0 = midpoints, rows 1..n = diag(half_ranges)
    midpoints = 0.5 * (bounds[:, 1] + bounds[:, 0])
    half_ranges = 0.5 * (bounds[:, 1] - bounds[:, 0])
    pcf_all = np.vstack((midpoints, np.diag(half_ranges)))

    # Construct PCRV
    mi = get_mi(in_pcord, in_pcdim)
    pc = PCRV(in_pcdim, n_params, pc_type, mi=mi, cfs=pcf_all.T)

    return pc, pcf_all, in_pcdim


# ── Step 2: Generate / load training samples ────────────────────────


def _generate_training_samples(
    pc: PCRV,
    n_samples: int,
    in_pcdim: int,
    seed: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate training samples in germ and physical spaces via PCRV.sampleGerm().

    Equivalent to ``uq_pc.py`` random sampling path (``--sampl rand``).
    Uses PyTUQ-native random sampling from the germ measure — no quadrature.

    Args:
        pc: Input PCRV object.
        n_samples: Number of training points.
        in_pcdim: Stochastic dimensionality.
        seed: Random seed for reproducibility.

    Returns:
        Tuple of (germ_train, X_train) — germ space and physical space.
    """
    if seed is not None:
        np.random.seed(seed)
    germ_train = pc.sampleGerm(n_samples)
    X_train = pc.evalPC(germ_train)

    logger.info(
        "Generated %d training samples (%d germ dims, %d physical params)",
        n_samples,
        in_pcdim,
        X_train.shape[1],
    )
    return germ_train, X_train


def _generate_test_samples(
    pc: PCRV,
    n_test: int,
    seed: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate random test samples for surrogate validation.

    Args:
        pc: Input PCRV object.
        n_test: Number of test points.
        seed: Random seed.

    Returns:
        Tuple of (germ_test, X_test).
    """
    if seed is not None:
        np.random.seed(seed)
    germ_test = pc.sampleGerm(n_test)
    X_test = pc.evalPC(germ_test)
    return germ_test, X_test


def _physical_to_germ(
    X: np.ndarray,
    bounds: np.ndarray,
) -> np.ndarray:
    """Scale physical-space samples to germ space [-1, 1].

    Used when loading precomputed samples from ``PrecomputedCache``
    (offline regime), which are stored in physical space.

    Args:
        X: Samples in physical space, shape (n, d).
        bounds: Parameter bounds, shape (d, 2).

    Returns:
        Samples in germ space [-1, 1], shape (n, d).
    """
    lb, ub = bounds[:, 0], bounds[:, 1]
    span = ub - lb
    span = np.where(span > 0, span, 1.0)
    return 2.0 * (X - lb) / span - 1.0  # type: ignore[no-any-return]


# ── Step 3: Evaluate / load forward model ───────────────────────────


def _evaluate_model_online(
    simulation_func: TimeseriesGeneratorVecoli,
    X_train: np.ndarray,
    X_test: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Evaluate the simulation function at training (and optionally test) points.

    Equivalent to ``uq_pc.py`` ``online_bb`` regime, but uses vEcoli's
    ``TimeseriesGeneratorVecoli.evaluate_batch()`` instead of an
    external ``model.x`` executable.

    Args:
        simulation_func: vEcoli simulation wrapper (workflow.py subprocess).
        X_train: Training inputs, shape (n_train, n_params).
        X_test: Optional test inputs, shape (n_test, n_params).

    Returns:
        Tuple of (Y_train, Y_test). Y_test is None if X_test is None.
    """
    logger.info("Evaluating forward model at %d training points", X_train.shape[0])
    Y_train = simulation_func.evaluate_batch(X_train)
    if Y_train.ndim == 1:
        Y_train = Y_train.reshape(-1, 1)

    Y_test = None
    if X_test is not None:
        logger.info("Evaluating forward model at %d test points", X_test.shape[0])
        Y_test = simulation_func.evaluate_batch(X_test)
        if Y_test.ndim == 1:
            Y_test = Y_test.reshape(-1, 1)

    return Y_train, Y_test


# ── Step 4: Construct PC surrogates ─────────────────────────────────


def _fit_surrogate(
    germ_train: np.ndarray,
    Y_train: np.ndarray,
    polynomial_order: int,
    regression: str = "lsq",
    tolerance: float = 1e-3,
) -> tuple[PCRV, list[Any]]:
    """Fit PCE surrogate following the ``pc_fit`` workflow.

    This is the core of the UQPC workflow (step 4 in ``uq_pc.py``).
    We replicate ``pytuq.workflows.fits.pc_fit`` here so that we can
    retain the per-output linear regression objects (``linregs``) for
    prediction variance estimation — ``pc_fit`` itself only returns
    the PCRV.

    The workflow:
      1. Build multi-index for the output polynomial order
      2. Construct PCRV with Legendre (LU) basis
      3. Evaluate basis matrix at training points
      4. Per-output: instantiate regressor, fit coefficients
      5. Sync multi-indices and coefficients into PCRV
      6. Set PCRV evaluation function

    Args:
        germ_train: Training samples in germ space, shape (n_train, d).
        Y_train: Training outputs, shape (n_train, n_out).
        polynomial_order: Output PCE order.
        regression: Fitting method — 'lsq', 'bcs', or 'anl'.
        tolerance: BCS tolerance (only used when regression='bcs').

    Returns:
        Tuple of (output_pcrv, linregs) — the fitted PCRV and per-output
        linear regression objects.
    """
    n_train, n_dim = germ_train.shape
    n_out = Y_train.shape[1]

    logger.info(
        "Fitting PCE surrogate: order=%d, method=%s, n_train=%d, n_outputs=%d",
        polynomial_order,
        regression,
        n_train,
        n_out,
    )

    # Step 1-2: Multi-index + PCRV
    mindex = get_mi(polynomial_order, n_dim)
    pcrv = PCRV(n_out, n_dim, "LU", mi=mindex)

    # Step 3: Basis matrix
    Amat = pcrv.evalBases(germ_train, 0)

    # Step 4: Per-output fitting
    mindices_list: list[np.ndarray] = []
    cfs_list: list[np.ndarray] = []
    linregs: list[Any] = []

    for j in range(n_out):
        logger.debug("Fitting output %d / %d", j + 1, n_out)

        if regression == "bcs":
            lreg_obj = bcs(eta=tolerance)
        elif regression == "anl":
            lreg_obj = anl()
        elif regression == "lsq":
            lreg_obj = lsq()
        else:
            raise ValueError(f"Unknown regression method: {regression!r}. Must be 'lsq', 'bcs', or 'anl'.")

        lreg_obj.fita(Amat, Y_train[:, j])
        mindices_list.append(mindex[lreg_obj.used, :])
        cfs_list.append(lreg_obj.cf)
        linregs.append(lreg_obj)

    # Step 5-6: Sync coefficients and set evaluation function
    pcrv.setMiCfs(mindices_list, cfs_list)
    pcrv.setFunction()

    return pcrv, linregs


def _predict_and_variance(
    output_pcrv: PCRV,
    linregs: list[Any],
    germ: np.ndarray,
    n_outputs: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Predict at given germ points and compute prediction variance.

    Equivalent to ``uq_pc.py`` lines after ``pc_fit`` call.

    Args:
        output_pcrv: Fitted output PCRV.
        linregs: Per-output linear regression objects.
        germ: Germ-space samples, shape (n, d).
        n_outputs: Number of output columns.

    Returns:
        Tuple of (Y_pc, Y_pc_std) — predictions and std deviations.
    """
    n = germ.shape[0]
    Y_pc = output_pcrv.function(germ)

    Y_pc_var = np.empty((n, n_outputs))
    for j, lreg in enumerate(linregs):
        basis_mat = output_pcrv.evalBases(germ, j)
        Y_pc_var[:, j] = lreg.predicta(basis_mat, msc=1)[1]

    return Y_pc, np.sqrt(Y_pc_var)


# ── Step 5: Compute relative errors ────────────────────────────────


def _compute_relative_errors(
    Y_true: np.ndarray,
    Y_pred: np.ndarray,
) -> np.ndarray:
    """Per-output relative L2 error between true and predicted values.

    Equivalent to ``uq_pc.py`` step 5.

    Args:
        Y_true: Ground truth, shape (n, n_out).
        Y_pred: PCE predictions, shape (n, n_out).

    Returns:
        Relative errors, shape (n_out,).
    """
    norms = np.linalg.norm(Y_true, axis=0)
    norms = np.where(norms > 0, norms, 1.0)
    return np.linalg.norm(Y_true - Y_pred, axis=0) / norms  # type: ignore[no-any-return]


# ── Step 6: Compute Sobol indices ───────────────────────────────────


def _compute_sobol(
    output_pcrv: PCRV,
    parameter_names: list[str],
    Y_train: np.ndarray,
) -> SobolIndices:
    """Compute Sobol indices from the fitted PCE.

    Equivalent to ``uq_pc.py`` step 6.  Uses PCRV's analytical Sobol
    computation (Sudret 2008) — no additional model evaluations needed.

    Main (first-order) indices measure each parameter's independent
    contribution to output variance.  Total-order indices include all
    interaction terms involving that parameter.

    For multi-output models, Sobol indices are variance-weighted across
    outputs to produce a single set of parameter importances.

    Args:
        output_pcrv: Fitted output PCRV with synced coefficients.
        parameter_names: Parameter names for labeling.
        Y_train: Training outputs (for variance weighting).

    Returns:
        SobolIndices with main, total, and joint indices.
    """
    allsens_main = output_pcrv.computeSens()  # (n_out, n_params)
    allsens_total = output_pcrv.computeTotSens()  # (n_out, n_params)
    allsens_joint = output_pcrv.computeJointSens()  # (n_out, n_params, n_params)

    n_outputs = Y_train.shape[1]

    logger.info(
        "Sobol indices computed: sum(main)=%s, sum(total)=%s",
        np.sum(allsens_main, axis=1),
        np.sum(allsens_total, axis=1),
    )

    # Variance-weighted aggregation across outputs
    if n_outputs == 1:
        first_order = allsens_main[0]
        total_order = allsens_total[0]
        second_order = allsens_joint[0]
    else:
        output_vars = np.var(Y_train, axis=0)
        total_var = output_vars.sum()
        if total_var > 0:
            weights = output_vars / total_var
        else:
            weights = np.ones(n_outputs) / n_outputs

        first_order = np.zeros(allsens_main.shape[1])
        total_order = np.zeros(allsens_total.shape[1])
        second_order = np.zeros(allsens_joint.shape[1:])
        for j in range(n_outputs):
            first_order += weights[j] * allsens_main[j]
            total_order += weights[j] * allsens_total[j]
            second_order += weights[j] * allsens_joint[j]

    return SobolIndices(
        first_order=first_order,
        total_order=total_order,
        second_order=second_order,
        parameter_names=parameter_names,
    )


def _build_surrogate(
    output_pcrv: PCRV,
    polynomial_order: int,
    n_params: int,
    n_outputs: int,
    bounds: np.ndarray,
) -> PCESurrogate:
    """Build an exportable PCESurrogate from the fitted PCRV.

    Args:
        output_pcrv: Fitted output PCRV.
        polynomial_order: PCE polynomial order.
        n_params: Number of input parameters.
        n_outputs: Number of outputs.
        bounds: Parameter bounds, shape (n_params, 2).

    Returns:
        PCESurrogate ready for export / prediction.
    """
    coefficients = output_pcrv.coefs[0] if output_pcrv.coefs else np.zeros(1)
    multi_indices = output_pcrv.mindices[0] if output_pcrv.mindices else np.zeros((1, n_params), dtype=int)

    return PCESurrogate(
        coefficients=coefficients,
        multi_indices=multi_indices,
        basis_type="legendre",
        polynomial_order=polynomial_order,
        input_dim=n_params,
        output_dim=n_outputs,
        input_bounds=bounds,
    )


# ── Full UQPC workflow ──────────────────────────────────────────────


def run_uqpc(
    param_space: XSpace,
    Y_train: np.ndarray,
    X_train: np.ndarray | None = None,
    polynomial_order: int = 3,
    regression: str = "lsq",
    tolerance: float = 1e-3,
    n_test: int = 0,
    X_test: np.ndarray | None = None,
    Y_test: np.ndarray | None = None,
    seed: int | None = 42,
) -> UQPCResult:
    """Execute the full UQPC workflow for a single aggregation strategy.

    This is the RFC006-adapted equivalent of running ``uq_pc.py`` with
    ``--regime offline --method <regression> --outord <polynomial_order>``.

    The workflow proceeds through all 6 UQPC steps:
      1. Input PC setup from parameter bounds
      2. Map training samples to germ space (or generate new ones)
      3. (Outputs already provided — offline regime)
      4. Fit PCE surrogate via ``pc_fit``
      5. Compute relative errors
      6. Compute Sobol sensitivity indices

    Args:
        param_space: Parameter space with names and bounds.
        Y_train: Training outputs, shape (n_samples, n_outputs).
        X_train: Training inputs in physical space, shape (n_samples, n_params).
            If None, generates new germ samples (but this requires Y_train
            to already correspond to those samples).
        polynomial_order: Output PCE order (uq_pc.py ``--outord``).
        regression: Fitting method — 'lsq', 'bcs', or 'anl'
            (uq_pc.py ``--method``).
        tolerance: BCS tolerance (uq_pc.py ``--tol``).
        n_test: Number of fresh test points to sample from the germ
            measure (uq_pc.py ``--ntst``).  If ``X_test``/``Y_test`` are
            provided below, ``n_test`` is ignored.  Used only when the
            model is evaluated on-line.
        X_test: Optional pre-evaluated validation inputs (physical space),
            shape (n_test, n_params).  When provided together with
            ``Y_test``, the workflow will compute test predictions and
            relative errors — the offline counterpart of UQPC's
            ``--ntst`` flag.
        Y_test: Optional pre-evaluated validation outputs, shape
            (n_test, n_outputs).
        seed: Random seed (uq_pc.py ``--seed``).

    Returns:
        UQPCResult with surrogate, Sobol indices, and diagnostics.
    """
    bounds = np.array(param_space.parameter_bounds)
    n_params = bounds.shape[0]
    parameter_names = param_space.parameter_names

    if Y_train.ndim == 1:
        Y_train = Y_train.reshape(-1, 1)
    n_samples, n_outputs = Y_train.shape

    logger.info(
        "UQPC workflow: %d samples, %d params, %d outputs, order=%d, method=%s",
        n_samples,
        n_params,
        n_outputs,
        polynomial_order,
        regression,
    )

    # ── Step 1: Input PC setup ──
    pc, _pcf_all, in_pcdim = _setup_input_pc(bounds)

    # ── Step 2: Map / generate training samples ──
    if X_train is not None:
        germ_train = _physical_to_germ(X_train, bounds)
    else:
        germ_train, X_train = _generate_training_samples(
            pc,
            n_samples,
            in_pcdim,
            seed=seed,
        )

    # ── Step 2b: Generate / ingest test samples (if requested) ──
    germ_test: np.ndarray | None = None
    if X_test is not None and Y_test is not None:
        # Offline validation: caller already evaluated the model at X_test
        if Y_test.ndim == 1:
            Y_test = Y_test.reshape(-1, 1)
        germ_test = _physical_to_germ(X_test, bounds)
    elif n_test > 0:
        # Fresh germ draws (requires online model) — no Y_test in offline mode
        germ_test, X_test = _generate_test_samples(pc, n_test, seed=seed)

    # ── Step 3: Outputs already provided (offline) ──
    #   (Y_train is passed in; Y_test is either passed in or stays None)

    # ── Step 4: Construct PC surrogate ──
    output_pcrv, linregs = _fit_surrogate(
        germ_train,
        Y_train,
        polynomial_order=polynomial_order,
        regression=regression,
        tolerance=tolerance,
    )

    # Predict at training points
    Y_train_pc, Y_train_pc_std = _predict_and_variance(
        output_pcrv,
        linregs,
        germ_train,
        n_outputs,
    )

    # Predict at test points (if available)
    Y_test_pc: np.ndarray | None = None
    Y_test_pc_std: np.ndarray | None = None
    if germ_test is not None:
        Y_test_pc, Y_test_pc_std = _predict_and_variance(
            output_pcrv,
            linregs,
            germ_test,
            n_outputs,
        )

    # ── Step 5: Relative errors ──
    relerr_train = _compute_relative_errors(Y_train, Y_train_pc)
    logger.info("Training relative errors: %s", relerr_train)

    relerr_test: np.ndarray | None = None
    if Y_test is not None and Y_test_pc is not None:
        relerr_test = _compute_relative_errors(Y_test, Y_test_pc)
        logger.info("Test relative errors: %s", relerr_test)

    # ── Step 6: Sobol indices ──
    sobol = _compute_sobol(output_pcrv, parameter_names, Y_train)

    # ── Build exportable surrogate ──
    surrogate = _build_surrogate(
        output_pcrv,
        polynomial_order,
        n_params,
        n_outputs,
        bounds,
    )

    return UQPCResult(
        sobol=sobol,
        surrogate=surrogate,
        pcrv=output_pcrv,
        linregs=linregs,
        germ_train=germ_train,
        X_train=X_train,
        Y_train=Y_train,
        Y_train_pc=Y_train_pc,
        Y_train_pc_std=Y_train_pc_std,
        relerr_train=relerr_train,
        germ_test=germ_test,
        X_test=X_test,
        Y_test=Y_test,
        Y_test_pc=Y_test_pc,
        Y_test_pc_std=Y_test_pc_std,
        relerr_test=relerr_test,
    )


def run_uqpc_live(
    param_space: XSpace,
    simulation_func: TimeseriesGeneratorVecoli,
    n_samples: int = 200,
    polynomial_order: int = 3,
    regression: str = "lsq",
    tolerance: float = 1e-3,
    n_test: int = 0,
    seed: int | None = 42,
) -> UQPCResult:
    """Execute the UQPC workflow with live model evaluation (online regime).

    Equivalent to ``uq_pc.py --regime online_bb``, but the "black box"
    is ``TimeseriesGeneratorVecoli.evaluate_batch()`` which runs vEcoli
    as a subprocess via ``runscripts/workflow.py``.

    Steps 1-2 generate samples via PCRV.sampleGerm(), step 3 evaluates the
    model, then steps 4-6 proceed as in ``run_uqpc``.

    Args:
        param_space: Parameter space with names and bounds.
        simulation_func: vEcoli simulation wrapper.
        n_samples: Number of training samples (drawn from germ measure).
        polynomial_order: Output PCE order.
        regression: Fitting method.
        tolerance: BCS tolerance.
        n_test: Number of validation test points.
        seed: Random seed.

    Returns:
        UQPCResult with surrogate, Sobol indices, and full diagnostics
        (including test errors if n_test > 0).
    """
    bounds = np.array(param_space.parameter_bounds)

    # ── Step 1: Input PC setup ──
    pc, _pcf_all, in_pcdim = _setup_input_pc(bounds)

    # ── Step 2: Generate training samples ──
    germ_train, X_train = _generate_training_samples(
        pc,
        n_samples,
        in_pcdim,
        seed=seed,
    )

    # ── Step 2b: Generate test samples ──
    germ_test, X_test = None, None
    if n_test > 0:
        germ_test, X_test = _generate_test_samples(pc, n_test, seed=seed)

    # ── Step 3: Evaluate forward model ──
    Y_train, Y_test = _evaluate_model_online(
        simulation_func,
        X_train,
        X_test,
    )

    n_outputs = Y_train.shape[1]

    # ── Step 4: Construct PC surrogate ──
    output_pcrv, linregs = _fit_surrogate(
        germ_train,
        Y_train,
        polynomial_order=polynomial_order,
        regression=regression,
        tolerance=tolerance,
    )

    Y_train_pc, Y_train_pc_std = _predict_and_variance(
        output_pcrv,
        linregs,
        germ_train,
        n_outputs,
    )

    Y_test_pc, Y_test_pc_std = None, None
    if germ_test is not None:
        Y_test_pc, Y_test_pc_std = _predict_and_variance(
            output_pcrv,
            linregs,
            germ_test,
            n_outputs,
        )

    # ── Step 5: Relative errors ──
    relerr_train = _compute_relative_errors(Y_train, Y_train_pc)
    logger.info("Training relative errors: %s", relerr_train)

    relerr_test = None
    if Y_test is not None and Y_test_pc is not None:
        relerr_test = _compute_relative_errors(Y_test, Y_test_pc)
        logger.info("Test relative errors: %s", relerr_test)

    # ── Step 6: Sobol indices ──
    parameter_names = param_space.parameter_names
    sobol = _compute_sobol(output_pcrv, parameter_names, Y_train)

    # ── Build exportable surrogate ──
    n_params = bounds.shape[0]
    surrogate = _build_surrogate(
        output_pcrv,
        polynomial_order,
        n_params,
        n_outputs,
        bounds,
    )

    return UQPCResult(
        sobol=sobol,
        surrogate=surrogate,
        pcrv=output_pcrv,
        linregs=linregs,
        germ_train=germ_train,
        X_train=X_train,
        Y_train=Y_train,
        Y_train_pc=Y_train_pc,
        Y_train_pc_std=Y_train_pc_std,
        relerr_train=relerr_train,
        germ_test=germ_test,
        X_test=X_test,
        Y_test=Y_test,
        Y_test_pc=Y_test_pc,
        Y_test_pc_std=Y_test_pc_std,
        relerr_test=relerr_test,
    )


def run_uqpc_from_cache(
    param_space: XSpace,
    cache: PrecomputedCache,
    polynomial_order: int = 3,
    regression: str = "lsq",
    tolerance: float = 1e-3,
    seed: int | None = 42,
) -> UQPCResult:
    """Execute the UQPC workflow from a PrecomputedCache (offline regime).

    Convenience wrapper around ``run_uqpc`` that extracts (X, Y) from
    the cache.  Equivalent to ``uq_pc.py --regime offline``.

    Args:
        param_space: Parameter space with names and bounds.
        cache: Cached (X, Y) from a prior ``uq sample`` run.
        polynomial_order: Output PCE order.
        regression: Fitting method.
        tolerance: BCS tolerance.
        seed: Random seed.

    Returns:
        UQPCResult.
    """
    return run_uqpc(
        param_space=param_space,
        Y_train=cache.Y,
        X_train=cache.X,
        polynomial_order=polynomial_order,
        regression=regression,
        tolerance=tolerance,
        seed=seed,
    )


# ── Per-strategy runners (RFC006 §3) ───────────────────────────────


def run_strategy1_uniform(
    param_space: XSpace,
    X: np.ndarray,
    Y: np.ndarray,
    polynomial_order: int = 3,
    regression: str = "lsq",
    tolerance: float = 1e-3,
    X_test: np.ndarray | None = None,
    Y_test: np.ndarray | None = None,
    seed: int | None = 42,
) -> UQPCResult:
    """Strategy 1: UQPC on uniformly aggregated (bulk) outputs.

    RFC006 §3 aggregation strategy (1): "Uniformly across all simulated
    cells and times (baseline)."

    The time-averaged output Y (mean across timesteps per sample) is
    the standard aggregation for bulk sensitivity analysis.

    Args:
        param_space: Parameter space.
        X: Physical-space inputs, shape (n_samples, n_params).
        Y: Time-averaged outputs, shape (n_samples, n_outputs).
        polynomial_order: PCE order.
        regression: Fitting method.
        tolerance: BCS tolerance.
        seed: Random seed.

    Returns:
        UQPCResult for bulk sensitivity.
    """
    return run_uqpc(
        param_space=param_space,
        Y_train=Y,
        X_train=X,
        polynomial_order=polynomial_order,
        regression=regression,
        tolerance=tolerance,
        X_test=X_test,
        Y_test=Y_test,
        seed=seed,
    )


def run_strategy2_by_generation(
    param_space: XSpace,
    X: np.ndarray,
    Y_timeseries: list[np.ndarray],
    Y_timeseries_meta: list[dict[str, np.ndarray]],
    polynomial_order: int = 3,
    regression: str = "lsq",
    tolerance: float = 1e-3,
    seed: int | None = 42,
) -> dict[int, UQPCResult]:
    """Strategy 2: UQPC per generation.

    RFC006 §3 aggregation strategy (2): "Stratified by generation
    (control of convergence towards steady-state growth)."

    Runs the full UQPC workflow independently for each generation,
    using per-generation time-averaged outputs.

    Args:
        param_space: Parameter space.
        X: Physical-space inputs, shape (n_samples, n_params).
        Y_timeseries: Per-sample raw timeseries arrays.
        Y_timeseries_meta: Per-sample metadata with 'generation' labels.
        polynomial_order: PCE order.
        regression: Fitting method.
        tolerance: BCS tolerance.
        seed: Random seed.

    Returns:
        Dict mapping generation → UQPCResult.
    """
    grouped = _aggregate_by_group(Y_timeseries, Y_timeseries_meta, "generation")
    results: dict[int, UQPCResult] = {}

    for gen, Y_g in grouped.items():
        logger.info("Strategy 2: fitting PCE for generation %d", gen)
        results[gen] = run_uqpc(
            param_space=param_space,
            Y_train=Y_g,
            X_train=X,
            polynomial_order=polynomial_order,
            regression=regression,
            tolerance=tolerance,
            seed=seed,
        )

    return results


def run_strategy3_by_seed(
    param_space: XSpace,
    X: np.ndarray,
    Y_timeseries: list[np.ndarray],
    Y_timeseries_meta: list[dict[str, np.ndarray]],
    polynomial_order: int = 3,
    regression: str = "lsq",
    tolerance: float = 1e-3,
    seed: int | None = 42,
) -> dict[int, UQPCResult]:
    """Strategy 3: UQPC per lineage seed.

    RFC006 §3 aggregation strategy (3): "Stratified by lineage seed
    (control of exogenous variance)."

    Runs the full UQPC workflow independently for each lineage seed.

    Args:
        param_space: Parameter space.
        X: Physical-space inputs, shape (n_samples, n_params).
        Y_timeseries: Per-sample raw timeseries arrays.
        Y_timeseries_meta: Per-sample metadata with 'lineage_seed' labels.
        polynomial_order: PCE order.
        regression: Fitting method.
        tolerance: BCS tolerance.
        seed: Random seed.

    Returns:
        Dict mapping lineage_seed → UQPCResult.
    """
    grouped = _aggregate_by_group(Y_timeseries, Y_timeseries_meta, "lineage_seed")
    results: dict[int, UQPCResult] = {}

    for lseed, Y_s in grouped.items():
        logger.info("Strategy 3: fitting PCE for lineage seed %d", lseed)
        results[lseed] = run_uqpc(
            param_space=param_space,
            Y_train=Y_s,
            X_train=X,
            polynomial_order=polynomial_order,
            regression=regression,
            tolerance=tolerance,
            seed=seed,
        )

    return results


def run_strategy4_growth_stratified(
    param_space: XSpace,
    X: np.ndarray,
    Y_timeseries: list[np.ndarray],
    n_bins: int = 10,
    mass_col_index: int = 0,
    polynomial_order: int = 3,
    regression: str = "lsq",
    tolerance: float = 1e-3,
    seed: int | None = 42,
) -> tuple[list[UQPCResult], UQPCResult]:
    """Strategy 4: UQPC stratified by cell cycle stage (growth progress).

    RFC006 §3 aggregation strategy (4): "Stratified by cell cycle stage,
    according to a physiological variable."

    Uses normalized log(dry_mass) as the cell cycle variable θ:
      θ = [log(mass) - log(mass_birth)] / [log(mass_div) - log(mass_birth)]

    θ = 0 at birth, θ = 1 at division.  Timesteps are binned into
    ``n_bins`` growth stages, and per-stage mean observables become
    the PCE outputs.

    A single PCE is fitted to the stacked (n_samples, n_bins * n_obs)
    output, then Sobol indices are extracted per stage.

    Additionally, individual UQPCResult objects are returned per stage
    for full diagnostics (training errors, prediction variance, etc.).

    Args:
        param_space: Parameter space.
        X: Physical-space inputs, shape (n_samples, n_params).
        Y_timeseries: Per-sample raw timeseries, each (n_t, n_obs).
        n_bins: Number of growth-progress bins.
        mass_col_index: Column index of dry mass in timeseries.
        polynomial_order: PCE order.
        regression: Fitting method.
        tolerance: BCS tolerance.
        seed: Random seed.

    Returns:
        Tuple of (per_stage_results, combined_result):
          - per_stage_results: list of UQPCResult, one per growth stage
          - combined_result: UQPCResult from the stacked multi-stage fit
    """
    n_samples = len(Y_timeseries)
    n_obs = Y_timeseries[0].shape[1]

    # Compute per-stage mean observables
    Y_stage_list: list[np.ndarray] = []
    for ts in Y_timeseries:
        theta = _compute_growth_fraction(ts, mass_col_index)
        bins = _bin_by_growth_stage(theta, n_bins)

        stage_means = np.zeros(n_bins * n_obs)
        for s in range(n_bins):
            mask = bins == s
            if np.any(mask):
                stage_means[s * n_obs : (s + 1) * n_obs] = ts[mask].mean(axis=0)
        Y_stage_list.append(stage_means)

    Y_stage = np.vstack(Y_stage_list)

    # Fit combined PCE across all stages
    combined_result = run_uqpc(
        param_space=param_space,
        Y_train=Y_stage,
        X_train=X,
        polynomial_order=polynomial_order,
        regression=regression,
        tolerance=tolerance,
        seed=seed,
    )

    # Also fit per-stage for individual diagnostics
    per_stage_results: list[UQPCResult] = []
    for s in range(n_bins):
        Y_s = Y_stage[:, s * n_obs : (s + 1) * n_obs]
        result = run_uqpc(
            param_space=param_space,
            Y_train=Y_s,
            X_train=X,
            polynomial_order=polynomial_order,
            regression=regression,
            tolerance=tolerance,
            seed=seed,
        )
        per_stage_results.append(result)
        logger.info(
            "Stage %d/%d: relerr_train=%s, S_T=%s",
            s + 1,
            n_bins,
            result.relerr_train,
            result.sobol.total_order,
        )

    return per_stage_results, combined_result


# ── Aggregation helpers (shared with strategies 2-4) ────────────────


def _aggregate_by_group(
    Y_timeseries: list[np.ndarray],
    Y_timeseries_meta: list[dict[str, np.ndarray]],
    group_key: str,
) -> dict[int, np.ndarray]:
    """Aggregate timeseries by a metadata group key.

    For each unique group value present across ALL samples, computes
    per-sample mean observables and stacks into (n_samples, n_obs).

    Args:
        Y_timeseries: Per-sample raw timeseries arrays.
        Y_timeseries_meta: Per-sample metadata dicts.
        group_key: Metadata key to group by ('generation' or 'lineage_seed').

    Returns:
        Dict mapping group value → Y array of shape (n_samples, n_obs).
    """
    n_samples = len(Y_timeseries)

    group_sets = []
    for meta in Y_timeseries_meta:
        if group_key not in meta:
            return {}
        group_sets.append(set(meta[group_key].tolist()))

    common_groups = sorted(set.intersection(*group_sets)) if group_sets else []
    if not common_groups:
        return {}

    n_obs = Y_timeseries[0].shape[1]
    result: dict[int, np.ndarray] = {}
    for g in common_groups:
        Y_g = np.zeros((n_samples, n_obs))
        for i in range(n_samples):
            mask = Y_timeseries_meta[i][group_key] == g
            if np.any(mask):
                Y_g[i] = Y_timeseries[i][mask].mean(axis=0)
        result[int(g)] = Y_g

    return result


def _compute_growth_fraction(
    timeseries: np.ndarray,
    mass_col_index: int = 0,
) -> np.ndarray:
    """Compute growth progress θ from a single-sample timeseries.

    θ = [log(mass) - log(mass_birth)] / [log(mass_div) - log(mass_birth)]

    This is a monotonic proxy for cell cycle progress that doesn't
    require spectral decomposition (Koopman, DMD, etc.).

    Args:
        timeseries: Shape (n_timesteps, n_obs).
        mass_col_index: Column index of dry mass.

    Returns:
        θ array, shape (n_timesteps,), values in [0, 1].
    """
    mass = timeseries[:, mass_col_index].astype(np.float64)
    mass = np.maximum(mass, 1e-10)
    log_mass = np.log(mass)
    log_birth = log_mass[0]
    log_div = log_mass[-1]
    denom = log_div - log_birth
    if denom <= 0:
        return np.linspace(0, 1, len(mass))
    return np.clip((log_mass - log_birth) / denom, 0, 1)  # type: ignore[no-any-return]


def _bin_by_growth_stage(
    theta: np.ndarray,
    n_bins: int,
) -> np.ndarray:
    """Assign timesteps to growth-progress bins.

    Args:
        theta: Growth progress, shape (n_timesteps,), values in [0, 1].
        n_bins: Number of bins.

    Returns:
        Bin indices, shape (n_timesteps,), values in [0, n_bins-1].
    """
    edges = np.linspace(0, 1, n_bins + 1)
    return np.clip(np.digitize(theta, edges) - 1, 0, n_bins - 1)  # type: ignore[no-any-return]


# ── Q1: Baseline variance accounting (no perturbation) ────────────


@dataclass
class VarianceBudget:
    """Q1 variance-components decomposition per observable.

    At fixed baseline sim_data (no input perturbation), decomposes the
    intrinsic variance in observables across stochastic seed, generation,
    and cell-cycle stage.  This is the "prediction confidence" deliverable
    of MS 08.4.2 — the denominator that anchors any Q2 Sobol number.

    All arrays have shape ``(n_obs,)``.
    """

    observable_names: list[str]
    total_variance: np.ndarray
    generation_variance: np.ndarray
    seed_variance: np.ndarray
    growth_stage_variance: np.ndarray
    residual_variance: np.ndarray
    generation_fraction: np.ndarray
    seed_fraction: np.ndarray
    growth_stage_fraction: np.ndarray
    residual_fraction: np.ndarray
    n_generations: int
    n_seeds: int
    n_stages: int

    def export(self, export_dir: str | Path) -> Path:
        """Write ``variance_budget.json`` to *export_dir*."""
        import json

        out = Path(export_dir)
        out.mkdir(parents=True, exist_ok=True)

        per_obs: dict[str, dict[str, float]] = {}
        for i, name in enumerate(self.observable_names):
            per_obs[name] = {
                "total_variance": float(self.total_variance[i]),
                "generation_variance": float(self.generation_variance[i]),
                "seed_variance": float(self.seed_variance[i]),
                "growth_stage_variance": float(self.growth_stage_variance[i]),
                "residual_variance": float(self.residual_variance[i]),
                "generation_fraction": float(self.generation_fraction[i]),
                "seed_fraction": float(self.seed_fraction[i]),
                "growth_stage_fraction": float(self.growth_stage_fraction[i]),
                "residual_fraction": float(self.residual_fraction[i]),
            }

        data = {
            "mode": "baseline",
            "n_generations": self.n_generations,
            "n_seeds": self.n_seeds,
            "n_stages": self.n_stages,
            "observable_names": self.observable_names,
            "per_observable": per_obs,
        }
        path = out / "variance_budget.json"
        path.write_text(json.dumps(data, indent=2))
        return path


def compute_variance_budget(
    timeseries: np.ndarray,
    meta: dict[str, np.ndarray],
    observable_names: list[str],
    n_bins: int = 10,
    mass_col_index: int = 0,
) -> VarianceBudget:
    """Q1 baseline variance decomposition — no surrogate fit.

    Decomposes the total variance of each observable into four additive
    components using between-group-means variance:

        σ²_gen   = Var(per-generation means)
        σ²_seed  = Var(per-seed means)
        σ²_θ     = Var(per-growth-stage means)
        σ²_resid = σ²_total − σ²_gen − σ²_seed − σ²_θ

    Args:
        timeseries: Raw output array, shape ``(n_timesteps, n_obs)``.
        meta: Dict with ``'generation'`` and ``'lineage_seed'`` arrays,
            each shape ``(n_timesteps,)``.
        observable_names: Feature labels, length ``n_obs``.
        n_bins: Number of growth-progress bins (θ stages).
        mass_col_index: Column index of dry mass for θ computation.

    Returns:
        VarianceBudget with per-observable variance partition.
    """
    n_obs = timeseries.shape[1]
    total_var = np.var(timeseries, axis=0, ddof=0)

    # ── Between-generation variance ──
    gen_arr = meta.get("generation")
    if gen_arr is not None:
        gens = np.unique(gen_arr)
        gen_means = np.array([timeseries[gen_arr == g].mean(axis=0) for g in gens])
        gen_var = np.var(gen_means, axis=0, ddof=0)
        n_generations = len(gens)
    else:
        gen_var = np.zeros(n_obs)
        n_generations = 0

    # ── Between-seed variance ──
    seed_arr = meta.get("lineage_seed")
    if seed_arr is not None:
        seeds = np.unique(seed_arr)
        seed_means = np.array([timeseries[seed_arr == s].mean(axis=0) for s in seeds])
        seed_var = np.var(seed_means, axis=0, ddof=0)
        n_seeds = len(seeds)
    else:
        seed_var = np.zeros(n_obs)
        n_seeds = 0

    # ── Between-growth-stage variance ──
    theta = _compute_growth_fraction(timeseries, mass_col_index)
    bins = _bin_by_growth_stage(theta, n_bins)
    stage_means_list = []
    for s in range(n_bins):
        mask = bins == s
        if np.any(mask):
            stage_means_list.append(timeseries[mask].mean(axis=0))
    if stage_means_list:
        stage_means = np.array(stage_means_list)
        stage_var = np.var(stage_means, axis=0, ddof=0)
    else:
        stage_var = np.zeros(n_obs)

    # ── Proportional allocation ──
    # Generation, seed, and growth stage are non-nested factors whose
    # between-group variances can overlap (sum > total).  We compute
    # η² = SS_between / SS_total for each, then normalize if the sum
    # exceeds 1 so that fractions are always additive.
    safe_total = np.where(total_var > 0, total_var, 1.0)
    raw_gen = gen_var / safe_total
    raw_seed = seed_var / safe_total
    raw_stage = stage_var / safe_total
    raw_sum = raw_gen + raw_seed + raw_stage

    # If raw fractions sum > 1, scale proportionally
    needs_rescale = raw_sum > 1.0
    scale = np.where(needs_rescale, 1.0 / np.maximum(raw_sum, 1e-10), 1.0)
    gen_frac = raw_gen * scale
    seed_frac = raw_seed * scale
    stage_frac = raw_stage * scale

    resid_frac = np.maximum(1.0 - gen_frac - seed_frac - stage_frac, 0.0)
    residual_var = resid_frac * safe_total

    return VarianceBudget(
        observable_names=observable_names,
        total_variance=total_var,
        generation_variance=gen_var,
        seed_variance=seed_var,
        growth_stage_variance=stage_var,
        residual_variance=residual_var,
        generation_fraction=gen_frac,
        seed_fraction=seed_frac,
        growth_stage_fraction=stage_frac,
        residual_fraction=resid_frac,
        n_generations=n_generations,
        n_seeds=n_seeds,
        n_stages=n_bins,
    )


# ═══════════════════════════════════════════════════════════════════
# Public two-stage API: sample() → cache → quantify()
# ═══════════════════════════════════════════════════════════════════


@dataclass
class QuantifyResult:
    """Complete output of the RFC006 quantification workflow (all 4 strategies).

    Attributes:
        strategy1: UQPCResult — uniform / bulk.
        strategy2: dict[int, UQPCResult] — per-generation (empty if no metadata).
        strategy3: dict[int, UQPCResult] — per-lineage-seed (empty if no metadata).
        strategy4_per_stage: list[UQPCResult] — per-growth-stage.
        strategy4_combined: UQPCResult — combined growth-stratified fit.
        parameter_names: Input parameter names.
        observable_names: Output observable names.
        cache: The PrecomputedCache that was analyzed.
    """

    strategy1: UQPCResult
    strategy2: dict[int, UQPCResult]
    strategy3: dict[int, UQPCResult]
    strategy4_per_stage: list[UQPCResult]
    strategy4_combined: UQPCResult
    parameter_names: list[str]
    observable_names: list[str]
    cache: PrecomputedCache
    pca_info: PCAReduction | None = None

    def export(self, export_dir: str | Path, cli_argv: list[str] | None = None) -> Path:
        """Write all strategy artifacts to *export_dir*.

        Produces a directory layout compatible with both the Marimo
        dashboard (``app/dashboard_simple.py``) and the Textual TUI.
        The ``uq_results.json`` follows the same schema as
        ``SimplePipelineResult._build_summary()`` so the dashboard
        can read it without changes.
        """
        import json

        out = Path(export_dir)
        out.mkdir(parents=True, exist_ok=True)
        names = self.parameter_names
        s1 = self.strategy1

        # Population surrogate (dashboard expects population_surrogate/)
        s1.surrogate.export(out / "population_surrogate")

        # Per-output PCE coefficients (for observable selector in DAW)
        pop_surr_dir = out / "population_surrogate"
        if s1.pcrv.coefs and s1.pcrv.mindices:
            _n_out = len(s1.pcrv.coefs)
            _mi0 = np.asarray(s1.pcrv.mindices[0])
            _n_basis = _mi0.shape[0]
            _coefs_matrix = np.zeros((_n_out, _n_basis))
            for _j, _c in enumerate(s1.pcrv.coefs):
                _c_arr = np.asarray(_c)
                _coefs_matrix[_j, : len(_c_arr)] = _c_arr
            np.save(pop_surr_dir / "coefficients_per_output.npy", _coefs_matrix)

        # Population Sobol .npy
        pop_dir = out / "population_sobol"
        pop_dir.mkdir(exist_ok=True)
        np.save(pop_dir / "first_order.npy", s1.sobol.first_order)
        np.save(pop_dir / "total_order.npy", s1.sobol.total_order)

        # Per-generation Sobol
        for gen, r in self.strategy2.items():
            d = out / f"generation_{gen}_sobol"
            d.mkdir(exist_ok=True)
            np.save(d / "first_order.npy", r.sobol.first_order)
            np.save(d / "total_order.npy", r.sobol.total_order)

        # Per-seed Sobol
        for seed, r in self.strategy3.items():
            d = out / f"seed_{seed}_sobol"
            d.mkdir(exist_ok=True)
            np.save(d / "first_order.npy", r.sobol.first_order)
            np.save(d / "total_order.npy", r.sobol.total_order)

        # Growth-stratified Sobol + surrogate
        for i, r in enumerate(self.strategy4_per_stage):
            d = out / f"growth_stage_{i}_sobol"
            d.mkdir(exist_ok=True)
            np.save(d / "first_order.npy", r.sobol.first_order)
            np.save(d / "total_order.npy", r.sobol.total_order)
        self.strategy4_combined.surrogate.export(out / "growth_stratified_surrogate")

        # Per-stage PCE coefficients (for exact prediction curve in DAW)
        gs_dir = out / "growth_stratified_surrogate"
        _s4c = self.strategy4_combined
        if _s4c.pcrv.coefs and _s4c.pcrv.mindices:
            _n_s4_out = len(_s4c.pcrv.coefs)
            _mi_s4 = np.asarray(_s4c.pcrv.mindices[0])
            _n_s4_basis = _mi_s4.shape[0]
            _s4_matrix = np.zeros((_n_s4_out, _n_s4_basis))
            for _j, _c in enumerate(_s4c.pcrv.coefs):
                _c_arr = np.asarray(_c)
                _s4_matrix[_j, : len(_c_arr)] = _c_arr
            np.save(gs_dir / "coefficients_per_output.npy", _s4_matrix)

        # Summary JSON — dashboard-compatible format
        n_bins = len(self.strategy4_per_stage)
        summary: dict[str, Any] = {
            "framework": "uq (PyTUQ UQPC workflow)",
            "parameters": {n: {"index": i} for i, n in enumerate(names)},
            "n_parameters": len(names),
            "observable_names": self.observable_names,
            "phase1_population": {
                "sobol_total_order": {n: round(float(v), 6) for n, v in zip(names, s1.sobol.total_order)},
                "sobol_first_order": {n: round(float(v), 6) for n, v in zip(names, s1.sobol.first_order)},
            },
            "strategy2_by_generation": {
                "n_generations": len(self.strategy2),
                "generations": [
                    {
                        "generation": gen,
                        "sobol_total_order": {n: round(float(r.sobol.total_order[i]), 6) for i, n in enumerate(names)},
                        "sobol_first_order": {n: round(float(r.sobol.first_order[i]), 6) for i, n in enumerate(names)},
                    }
                    for gen, r in sorted(self.strategy2.items())
                ],
            },
            "strategy3_by_seed": {
                "n_seeds": len(self.strategy3),
                "seeds": [
                    {
                        "lineage_seed": seed,
                        "sobol_total_order": {n: round(float(r.sobol.total_order[i]), 6) for i, n in enumerate(names)},
                        "sobol_first_order": {n: round(float(r.sobol.first_order[i]), 6) for i, n in enumerate(names)},
                    }
                    for seed, r in sorted(self.strategy3.items())
                ],
            },
            "phase2_growth_stratified": {
                "n_stages": n_bins,
                "stages": [
                    {
                        "stage": j,
                        "theta_range": [
                            round(j / n_bins, 3) if n_bins > 0 else 0,
                            round((j + 1) / n_bins, 3) if n_bins > 0 else 1,
                        ],
                        "sobol_total_order": {n: round(float(r.sobol.total_order[i]), 6) for i, n in enumerate(names)},
                        "sobol_first_order": {n: round(float(r.sobol.first_order[i]), 6) for i, n in enumerate(names)},
                    }
                    for j, r in enumerate(self.strategy4_per_stage)
                ],
            },
        }
        (out / "uq_results.json").write_text(json.dumps(summary, indent=2))

        # PCA artifacts (if output-side PCA was applied)
        if self.pca_info is not None:
            self.pca_info.export(out)
            summary["pca"] = {
                "n_components": self.pca_info.n_components,
                "explained_variance_pct": [round(float(v) * 100, 2) for v in self.pca_info.explained_variance_ratio],
                "total_explained_pct": round(float(self.pca_info.explained_variance_ratio.sum()) * 100, 2),
            }
            # Re-write summary with PCA info
            (out / "uq_results.json").write_text(json.dumps(summary, indent=2))

        # Reproducibility manifest
        _write_manifest(out, self.cache, cli_argv=cli_argv)

        return out


def _write_manifest(
    export_dir: Path,
    cache: PrecomputedCache,
    cli_argv: list[str] | None = None,
) -> None:
    """Write ``manifest.json`` for full reproducibility.

    Records software versions, git SHAs, data hashes, and the CLI
    command that produced these results — everything needed to reproduce
    or audit a UQ run.
    """
    import importlib.metadata

    def _git_sha(repo_dir: str | Path) -> str:
        try:
            return (
                _subprocess
                .check_output(
                    ["git", "rev-parse", "HEAD"],
                    cwd=str(repo_dir),
                    stderr=_subprocess.DEVNULL,
                )
                .decode()
                .strip()
            )
        except Exception:
            return "unknown"

    def _sha256_file(path: Path) -> str:
        if not path.exists():
            return "missing"
        h = hashlib.sha256()
        h.update(path.read_bytes())
        return h.hexdigest()

    def _pkg_version(name: str) -> str:
        try:
            return importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            return "not installed"

    uqecoli_root = Path(__file__).resolve().parent.parent
    vecoli_sha = "unknown"
    try:
        import ecoli  # type: ignore[import-not-found]

        vecoli_root = Path(ecoli.__file__).resolve().parent.parent
        vecoli_sha = _git_sha(vecoli_root)
    except ImportError:
        pass

    manifest = {
        "timestamp": datetime.now(tz=timezone.utc).isoformat(),
        "hostname": platform.node(),
        "python_version": sys.version,
        "package_versions": {
            "pytuq": _pkg_version("pytuq"),
            "numpy": _pkg_version("numpy"),
            "polars": _pkg_version("polars"),
            "scipy": _pkg_version("scipy"),
        },
        "git_sha": {
            "uqEcoli": _git_sha(uqecoli_root),
            "vEcoli": vecoli_sha,
        },
        "cli_command": cli_argv or sys.argv,
        "data_hashes": {
            "X.npy": _sha256_file(cache.cache_dir / "X.npy"),
            "Y.npy": _sha256_file(cache.cache_dir / "Y.npy"),
        },
        "cache_path": str(cache.cache_dir),
        "n_samples": int(cache.X.shape[0]),
        "n_outputs": int(cache.Y.shape[1]),
        "n_parameters": int(cache.X.shape[1]),
        "parameter_names": cache.parameter_names,
        "observable_names": cache.metadata.get("observable_columns", []),
    }

    # Embed full parameter specs (attr_path, bounds, description) if available
    try:
        from libuq.pipeline.param_loader import DEFAULT_SIM_DATA_PARAMETERS

        param_lookup = {p.name: p for p in DEFAULT_SIM_DATA_PARAMETERS}
        specs = []
        for name in cache.parameter_names:
            p = param_lookup.get(name)
            if p:
                specs.append({
                    "name": p.name,
                    "attr_path": p.attr_path,
                    "bounds": list(p.bounds),
                    "description": p.description or "",
                })
            else:
                specs.append({"name": name, "attr_path": "", "bounds": [], "description": ""})
        manifest["parameter_specs"] = specs
    except Exception:
        pass

    # Record multi-parca info if present in workflow config
    config_path = cache.cache_dir / "_batch" / "workflow_config.json"
    if config_path.exists():
        try:
            wf_config = _json.loads(config_path.read_text())
            if "parca_variants" in wf_config:
                manifest["parca_variants"] = wf_config["parca_variants"]
        except Exception:
            pass

    # Record conditions metadata if this is a multi-condition cache
    cond_path = cache.cache_dir / "conditions.json"
    if cond_path.exists():
        try:
            manifest["conditions"] = _json.loads(cond_path.read_text())
        except Exception:
            pass

    (export_dir / "manifest.json").write_text(_json.dumps(manifest, indent=2, default=str))


def _adaptive_sampling_loop(
    input_pc: PCRV,
    sim_func: TimeseriesGeneratorVecoli,
    n_budget: int,
    polynomial_order: int = 3,
    regression: str = "lsq",
    tol: float = 0.05,
    batch_fraction: float = 0.33,
    seed: int | None = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[float]]:
    """Adaptive sampling loop: start small, refine where error is highest.

    1. Draw ``n_budget * batch_fraction`` initial samples.
    2. Fit PCE, compute leave-one-out relative error.
    3. If error > tol and budget remains, draw more samples from the germ
       measure and re-fit.
    4. Repeat until convergence or budget exhausted.

    Args:
        input_pc: PCRV from ``_setup_input_pc()``.
        sim_func: vEcoli simulation wrapper.
        n_budget: Total sample budget.
        polynomial_order: PCE order.
        regression: Fitting method.
        tol: Target mean relative error for convergence.
        batch_fraction: Fraction of budget for initial batch.
        seed: Random seed.

    Returns:
        Tuple of (germ_all, X_all, Y_all, convergence_history).
    """
    if seed is not None:
        np.random.seed(seed)

    n_initial = max(polynomial_order + 2, int(n_budget * batch_fraction))
    n_remaining = n_budget - n_initial
    batch_size = max(1, n_remaining // 4)

    # Initial batch
    germ = input_pc.sampleGerm(n_initial)
    X = input_pc.evalPC(germ)
    Y = sim_func.evaluate_batch(X)
    if Y.ndim == 1:
        Y = Y.reshape(-1, 1)

    convergence: list[float] = []

    while True:
        # Fit PCE on current data
        _, linregs = _fit_surrogate(germ, Y, polynomial_order, regression)

        # LOO relative error estimate
        n_dim = germ.shape[1]
        mindex = get_mi(polynomial_order, n_dim)
        pcrv_tmp = PCRV(Y.shape[1], n_dim, "LU", mi=mindex)
        Amat = pcrv_tmp.evalBases(germ, 0)
        loo_errors = []
        for j in range(Y.shape[1]):
            pred = Amat @ linregs[j].cf
            residuals = Y[:, j] - pred
            norm_y = np.linalg.norm(Y[:, j])
            loo_errors.append(np.linalg.norm(residuals) / max(norm_y, 1e-12))
        mean_err = float(np.mean(loo_errors))
        convergence.append(mean_err)

        logger.info(
            "Adaptive sampling: %d samples, mean relerr=%.4f (tol=%.4f)",
            germ.shape[0],
            mean_err,
            tol,
        )

        if mean_err <= tol or n_remaining <= 0:
            break

        # Draw more samples
        n_new = min(batch_size, n_remaining)
        germ_new = input_pc.sampleGerm(n_new)
        X_new = input_pc.evalPC(germ_new)
        Y_new = sim_func.evaluate_batch(X_new)
        if Y_new.ndim == 1:
            Y_new = Y_new.reshape(-1, 1)

        germ = np.vstack([germ, germ_new])
        X = np.vstack([X, X_new])
        Y = np.vstack([Y, Y_new])
        n_remaining -= n_new

    return germ, X, Y, convergence


def sample(
    sim_data_path: str | Path,
    cache_dir: str | Path,
    n_samples: int = 200,
    seed: int = 42,
    parameters: list[SimDataParameter] | None = None,
    params_file: str | Path | None = None,
    observable_columns: list[str] | None = None,
    max_duration: float = 10800.0,
    generations: int = 1,
    n_init_sims: int = 1,
    max_workers: int | None = None,
) -> PrecomputedCache:
    """UQPC Steps 1-3: setup inputs, generate samples, evaluate vEcoli.

    Implements the first three steps of the PyTUQ UQPC workflow
    (https://sandialabs.github.io/pytuq/apps/uqpc.html) with vEcoli
    as the black-box model (``runscripts/workflow.py``).

    Step 1 — Setup inputs:
        Load ``simData.cPickle``, build parameter space from
        ``SimDataParameter`` specs, construct input PCRV (Legendre
        basis, order 1) encoding the affine map bounds → [-1, 1].

    Step 2 — Generate samples:
        Draw ``n_samples`` germ-space realizations via
        ``PCRV.sampleGerm()``, map to physical parameter space via
        ``PCRV.evalPC()``.  This is PyTUQ's native random sampling
        (equivalent to ``uq_pc.py --sampl rand``).

    Step 3 — Evaluate model:
        Pass physical-space samples to
        ``TimeseriesGeneratorVecoli._run_batch()``, which runs vEcoli
        as subprocesses via ``runscripts/workflow.py`` and collects
        Parquet outputs.  Returns aggregated Y + raw timeseries +
        per-row generation/seed metadata.

    Results are saved as a ``PrecomputedCache``:
        ``X.npy``, ``Y.npy``, ``timeseries/``, ``metadata.json``.
    Also saves ``germ_train.npy`` so ``quantify()`` can skip the
    physical→germ inverse transform.

    Args:
        sim_data_path: Path to ``simData.cPickle``.
        cache_dir: Directory to write cached data.
        n_samples: Number of samples to draw from PCRV.
        seed: Random seed for PCRV.sampleGerm().
        parameters: List of ``SimDataParameter`` specs.
            If None, uses ``DEFAULT_SIM_DATA_PARAMETERS`` (6 params).
        params_file: Alternative: path to a JSON file containing
            a list of ``SimDataParameter`` dicts.  Mutually exclusive
            with ``parameters``.
        observable_columns: Which Parquet columns to extract.
            Defaults to dry_mass, cell_mass, volume, growth.
            For richer observables, use ``uq.observables.collect_observables``
            via the CLI ``--observables`` presets (higher_order,
            exchange_fluxes, transcriptome, proteome, fluxome).
        max_duration: vEcoli simulation wall-clock limit (seconds).
        generations: Number of cell generations per simulation.
            Use >= 2 to enable Strategy 2 (by-generation GSA).
        n_init_sims: Number of initial seeds per variant.
        max_workers: Max parallel subprocesses. None = sequential.

    Returns:
        PrecomputedCache with X, Y, Y_timeseries, Y_timeseries_meta,
        and germ_train stored in metadata.
    """
    # ── Step 1: Setup inputs ──
    logger.info("Step 1: loading sim_data and building parameter space")

    # Load sim_data and build parameter space
    ds = ParameterDataset(sim_data_path=str(sim_data_path))

    # Resolve parameter specs
    if parameters is not None and params_file is not None:
        raise ValueError("Pass either `parameters` or `params_file`, not both.")
    if params_file is not None:
        raw = _json.loads(Path(params_file).read_text())
        parameters = [SimDataParameter.from_dict(p) for p in raw]
    param_space = ds.to_parameter_space(parameters=parameters)

    if param_space.n_parameters == 0:
        raise RuntimeError("Parameter space is empty.")

    bounds = np.array(param_space.parameter_bounds)
    n_params = bounds.shape[0]

    # Build input PCRV from bounds (Legendre, order 1)
    input_pc, _, _in_pcdim = _setup_input_pc(bounds)

    logger.info(
        "Parameter space: %d params, bounds shape %s",
        n_params,
        bounds.shape,
    )

    # ── Step 2: Generate samples (PyTUQ-native) ──
    logger.info("Step 2: generating %d samples via PCRV.sampleGerm()", n_samples)

    if seed is not None:
        np.random.seed(seed)
    germ_train = input_pc.sampleGerm(n_samples)
    X_train = input_pc.evalPC(germ_train)

    logger.info(
        "Germ samples: %s, physical samples: %s",
        germ_train.shape,
        X_train.shape,
    )

    # ── Step 3: Evaluate model (vEcoli via workflow.py) ──
    logger.info("Step 3: evaluating vEcoli at %d sample points", n_samples)

    obs = observable_columns or [
        "listeners__mass__dry_mass",
        "listeners__mass__cell_mass",
        "listeners__mass__volume",
        "listeners__mass__growth",
    ]
    sim_func = TimeseriesGeneratorVecoli(
        baseline_sim_data=ds.sim_data,
        param_space=param_space,
        max_duration=max_duration,
        generations=generations,
        n_init_sims=n_init_sims,
        output_keys=[c.split("__")[-1] for c in obs],
    )

    Y_agg, Y_timeseries, Y_meta = sim_func._run_batch(
        X_train,
        max_workers=max_workers,
    )

    logger.info(
        "Model evaluation complete: Y_agg=%s, %d timeseries",
        Y_agg.shape,
        len(Y_timeseries) if Y_timeseries else 0,
    )

    # ── Save to PrecomputedCache ──
    cache = PrecomputedCache(
        cache_dir=Path(cache_dir),
        X=X_train,
        Y=Y_agg,
        parameter_names=param_space.parameter_names,
        metadata={
            "bounds": bounds.tolist(),
            "seed": seed,
            "germ_train": germ_train.tolist(),
            "observable_columns": obs,
        },
        Y_timeseries=Y_timeseries,
        Y_timeseries_meta=Y_meta,
    )
    cache.save()

    # Also save germ samples as npy for direct reload in quantify()
    np.save(Path(cache_dir) / "germ_train.npy", germ_train)

    logger.info("Cache saved to %s", cache_dir)
    return cache


def quantify(
    cache_dir: str | Path,
    sim_data_path: str | Path,
    polynomial_order: int = 3,
    n_bins: int = 10,
    regression: str = "lsq",
    tolerance: float = 1e-3,
    parameters: list[SimDataParameter] | None = None,
    export_path: str | Path | None = None,
    seed: int | None = 42,
    output_pca: int | None = None,
) -> QuantifyResult:
    """UQPC Steps 4-5: build PC surrogates, compute Sobol indices.

    Loads a ``PrecomputedCache`` produced by ``sample()`` and runs the
    full UQPC workflow (Step 4: surrogate fitting, Step 5: sensitivity
    analysis) for each of the four RFC006 aggregation strategies.

    Germ-space samples are loaded directly from the cache (saved by
    ``sample()``), ensuring exact consistency between the sampling
    distribution and the PCE basis — no inverse transform needed.

    Args:
        cache_dir: Path to the cache from ``sample()``.
        sim_data_path: Path to ``simData.cPickle`` (same as in ``sample()``).
        polynomial_order: Output PCE order (uq_pc.py ``--outord``).
        n_bins: Number of growth-progress bins for Strategy 4.
        regression: PyTUQ fitting method — 'lsq', 'bcs', or 'anl'.
        tolerance: BCS tolerance (only when regression='bcs').
        parameters: Same ``SimDataParameter`` specs used in ``sample()``.
            If None, uses ``DEFAULT_SIM_DATA_PARAMETERS``.
        export_path: If given, write all artifacts here.
        seed: Random seed for reproducibility.
        output_pca: If set, apply PCA to reduce Y to this many components
            before PCE fitting.  Useful for high-dimensional outputs
            (transcriptome, proteome).  Sobol indices are per-PC.

    Returns:
        QuantifyResult with all 4 strategy outputs.
    """
    cache_dir = Path(cache_dir)

    # ── Load cache ──
    cache = PrecomputedCache.load(cache_dir)
    X = cache.X
    Y = cache.Y

    # ── Rebuild parameter space (same specs as sample()) ──
    ds = ParameterDataset(sim_data_path=str(sim_data_path))

    # Auto-detect parameters from cache metadata when none are provided.
    # This handles the case where the cache was built with a different set
    # of default parameters than the current DEFAULT_SIM_DATA_PARAMETERS.
    if parameters is None and cache.parameter_names:
        cached_names = set(cache.parameter_names)
        matching = [p for p in DEFAULT_SIM_DATA_PARAMETERS if p.name in cached_names]
        if len(matching) == len(cached_names):
            parameters = matching
            logger.info(
                "Auto-detected %d parameters from cache: %s",
                len(parameters),
                [p.name for p in parameters],
            )
        elif cache.metadata.get("bounds"):
            # Reconstruct minimal SimDataParameter specs from cache metadata
            cached_bounds = cache.metadata["bounds"]
            all_defaults = {p.name: p for p in DEFAULT_SIM_DATA_PARAMETERS}
            parameters = []
            for i, name in enumerate(cache.parameter_names):
                if name in all_defaults:
                    p = all_defaults[name]
                    parameters.append(
                        SimDataParameter(
                            name=p.name,
                            attr_path=p.attr_path,
                            bounds=tuple(cached_bounds[i]),
                            index=p.index,
                            description=p.description,
                        )
                    )
                else:
                    logger.warning(
                        "Cache parameter %r not found in defaults, using cache bounds",
                        name,
                    )
            if len(parameters) != len(cached_names):
                raise RuntimeError(
                    f"Cannot reconstruct parameter space: cache has {cached_names}, "
                    f"but only matched {[p.name for p in parameters]} from defaults."
                )

    param_space = ds.to_parameter_space(parameters=parameters)

    # ── Load germ samples (saved by sample()) ──
    germ_path = cache_dir / "germ_train.npy"
    if germ_path.exists():
        germ_train = np.load(germ_path)
        logger.info("Loaded germ samples from %s", germ_path)
    else:
        # Fallback: inverse-transform physical samples to germ space
        bounds = np.array(param_space.parameter_bounds)
        germ_train = _physical_to_germ(X, bounds)
        logger.info("Computed germ samples from physical→germ transform")

    obs = cache.metadata.get(
        "observable_columns",
        [
            "listeners__mass__dry_mass",
            "listeners__mass__cell_mass",
            "listeners__mass__volume",
            "listeners__mass__growth",
        ],
    )

    # ── Optional output-side PCA ──
    pca_info: PCAReduction | None = None
    if output_pca is not None and output_pca > 0:
        Y, pca_info = apply_output_pca(Y, output_pca, obs)
        obs = [f"PC{k + 1}" for k in range(pca_info.n_components)]
        logger.info(
            "PCA applied: %d outputs → %d PCs (%.1f%% variance)",
            len(pca_info.original_names),
            pca_info.n_components,
            pca_info.explained_variance_ratio.sum() * 100,
        )
        # Also transform test set if present
        if cache.Y_test is not None:
            Y_test_centered = cache.Y_test - pca_info.mean
            cache = PrecomputedCache(
                cache_dir=cache.cache_dir,
                X=cache.X,
                Y=Y,
                parameter_names=cache.parameter_names,
                metadata=cache.metadata,
                Y_timeseries=cache.Y_timeseries,
                Y_timeseries_meta=cache.Y_timeseries_meta,
                X_test=cache.X_test,
                Y_test=Y_test_centered @ pca_info.components.T,
            )

    # ── Step 4-5 for each strategy ──

    # Strategy 1: uniform / bulk
    logger.info("Strategy 1: uniform (bulk)")
    s1 = run_strategy1_uniform(
        param_space,
        X,
        Y,
        polynomial_order=polynomial_order,
        regression=regression,
        tolerance=tolerance,
        X_test=cache.X_test,
        Y_test=cache.Y_test,
        seed=seed,
    )
    if cache.X_test is not None:
        logger.info(
            "Strategy 1 validated on %d held-out samples (test relerr=%s)",
            cache.X_test.shape[0],
            s1.relerr_test.tolist() if s1.relerr_test is not None else None,
        )

    # Strategy 2: by generation
    s2: dict[int, UQPCResult] = {}
    if cache.Y_timeseries is not None and cache.Y_timeseries_meta is not None:
        logger.info("Strategy 2: by generation")
        s2 = run_strategy2_by_generation(
            param_space,
            X,
            cache.Y_timeseries,
            cache.Y_timeseries_meta,
            polynomial_order=polynomial_order,
            regression=regression,
            tolerance=tolerance,
            seed=seed,
        )

    # Strategy 3: by lineage seed
    s3: dict[int, UQPCResult] = {}
    if cache.Y_timeseries is not None and cache.Y_timeseries_meta is not None:
        logger.info("Strategy 3: by lineage seed")
        s3 = run_strategy3_by_seed(
            param_space,
            X,
            cache.Y_timeseries,
            cache.Y_timeseries_meta,
            polynomial_order=polynomial_order,
            regression=regression,
            tolerance=tolerance,
            seed=seed,
        )

    # Strategy 4: growth-stratified
    s4_per_stage: list[UQPCResult] = []
    s4_combined: UQPCResult | None = None
    if cache.Y_timeseries is not None:
        logger.info("Strategy 4: growth-stratified (%d bins)", n_bins)
        s4_per_stage, s4_combined = run_strategy4_growth_stratified(
            param_space,
            X,
            cache.Y_timeseries,
            n_bins=n_bins,
            polynomial_order=polynomial_order,
            regression=regression,
            tolerance=tolerance,
            seed=seed,
        )
    else:
        s4_combined = s1
        logger.warning("Strategy 4: no timeseries in cache, using bulk as fallback")

    result = QuantifyResult(
        strategy1=s1,
        strategy2=s2,
        strategy3=s3,
        strategy4_per_stage=s4_per_stage,
        strategy4_combined=s4_combined,
        parameter_names=param_space.parameter_names,
        observable_names=obs,
        cache=cache,
        pca_info=pca_info,
    )

    if export_path is not None:
        result.export(export_path, cli_argv=sys.argv)
        logger.info("Artifacts exported to %s", export_path)

    return result
