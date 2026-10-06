"""Strict ASV-to-MAG abundance estimation used in the associated study.

The estimator validates its inputs but does not silently intersect, reorder,
subset, impute, or drop data. Prepare the matrices explicitly before calling
``estimate_mag_abundance``.
"""

from __future__ import annotations

from itertools import combinations
from numbers import Integral, Real

import networkx as nx
import numpy as np
import pandas as pd
import sympy as spm
from scipy.optimize import nnls

__all__ = ["estimate_mag_abundance"]


def _positive_real(name: str, value: float, *, allow_zero: bool = False) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite real number.")
    value = float(value)
    if not np.isfinite(value):
        raise ValueError(f"{name} must be finite.")
    if value < 0 or (value == 0 and not allow_zero):
        relation = "nonnegative" if allow_zero else "positive"
        raise ValueError(f"{name} must be {relation}.")
    return value


def _positive_int(name: str, value: int) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral) or value < 1:
        raise ValueError(f"{name} must be a positive integer.")
    return int(value)


def _check_labels(frame: pd.DataFrame, name: str) -> None:
    for axis_name, labels in (("row", frame.index), ("column", frame.columns)):
        if not labels.is_unique:
            raise ValueError(f"{name} contains duplicate {axis_name} labels.")
        if labels.to_frame(index=False).isna().to_numpy().any():
            raise ValueError(f"{name} contains missing {axis_name} labels.")


def _validate_inputs(
    copy_number_matrix: pd.DataFrame,
    asv_abundance: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray]:
    if not isinstance(copy_number_matrix, pd.DataFrame):
        raise TypeError("copy_number_matrix must be a pandas DataFrame.")
    if not isinstance(asv_abundance, pd.DataFrame):
        raise TypeError("asv_abundance must be a pandas DataFrame.")
    if copy_number_matrix.empty:
        raise ValueError("copy_number_matrix must contain at least one ASV and one MAG.")
    if asv_abundance.empty:
        raise ValueError("asv_abundance must contain at least one ASV and one sample.")

    _check_labels(copy_number_matrix, "copy_number_matrix")
    _check_labels(asv_abundance, "asv_abundance")

    # ASV matching is deliberately not done inside the estimator.
    if not copy_number_matrix.index.equals(asv_abundance.index):
        raise ValueError(
            "ASV IDs and order must match exactly between copy_number_matrix and "
            "asv_abundance. Intersect and reorder them before estimation."
        )

    if any(pd.api.types.is_complex_dtype(dtype) for dtype in copy_number_matrix.dtypes):
        raise ValueError("copy_number_matrix must contain real values, not complex values.")
    if any(pd.api.types.is_complex_dtype(dtype) for dtype in asv_abundance.dtypes):
        raise ValueError("asv_abundance must contain real values, not complex values.")

    try:
        A = copy_number_matrix.to_numpy(dtype=float, copy=True)
        Y = asv_abundance.to_numpy(dtype=float, copy=True)
    except (TypeError, ValueError) as exc:
        raise ValueError("Both inputs must contain real numeric values.") from exc

    if not np.isfinite(A).all():
        raise ValueError("copy_number_matrix contains NaN or infinity.")
    if not np.isfinite(Y).all():
        raise ValueError(
            "asv_abundance contains NaN or infinity. Remove missing samples/time "
            "points before estimation."
        )
    if (A < 0).any():
        raise ValueError("copy_number_matrix contains negative values.")
    if (Y < 0).any():
        raise ValueError("asv_abundance contains negative values.")

    # Copy numbers are genomic counts. Reject fractions instead of silently
    # truncating them before the exact rank calculation.
    if not np.equal(A, np.floor(A)).all():
        raise ValueError("copy_number_matrix must contain nonnegative integer copy numbers.")

    zero_asv = ~(A > 0).any(axis=1)
    if zero_asv.any():
        ids = copy_number_matrix.index[zero_asv].tolist()
        raise ValueError(
            "copy_number_matrix contains ASVs that map to no MAG. "
            f"Remove or resolve these rows before estimation: {ids!r}"
        )

    zero_mag = ~(A > 0).any(axis=0)
    if zero_mag.any():
        ids = copy_number_matrix.columns[zero_mag].tolist()
        raise ValueError(
            "copy_number_matrix contains MAGs with no corresponding ASVs and "
            f"therefore cannot be estimated: {ids!r}"
        )

    return A, Y


def estimate_mag_abundance(
    copy_number_matrix: pd.DataFrame,
    asv_abundance: pd.DataFrame,
    lambda_penalty: float = 1e-8,
    epsilon: float = 1e-4,
    delta: float = 1e-4,
    max_iter: int = 50,
    tol: float = 1e-6,
) -> pd.DataFrame:
    """Estimate nonnegative MAG abundances independently for each sample.

    Parameters
    ----------
    copy_number_matrix : pandas.DataFrame
        ASV × MAG matrix of finite, nonnegative integer copy numbers. Every ASV
        must map to at least one MAG, and every MAG must contain at least one ASV.
    asv_abundance : pandas.DataFrame
        ASV × sample/time-point matrix of finite, nonnegative abundances. Its ASV
        IDs and row order must exactly match ``copy_number_matrix``. Remove
        missing samples/time points before calling this function.
    lambda_penalty : float, default=1e-8
        Strength of the network regularization. It must be positive if a
        rank-deficient connected component is present.
    epsilon : float, default=1e-4
        Small positive diagonal term added to the full MAG-space Laplacian when
        regularization is used.
    delta : float, default=1e-4
        Positive lower bound used in the Poisson-based IRLS variance term.
    max_iter : int, default=50
        Maximum number of outer IRLS iterations.
    tol : float, default=1e-6
        Relative-change stopping threshold for the MAG abundance vector.

    Returns
    -------
    pandas.DataFrame
        MAG × sample/time-point abundance estimates.

    Notes
    -----
    Samples/time points are fitted independently; no temporal information is used.

    The ASV–MAG bipartite graph is decomposed into connected components. Exact
    matrix rank is evaluated within each component. For rank-deficient components,
    MAG pairs are weighted by the number of ASV types they share, producing a
    graph-Laplacian smoothing term.

    If any rank-deficient component exists, the implementation factors
    ``L + epsilon * I`` in the full MAG space. ``L`` contains graph-smoothing
    edges only for rank-deficient components; the diagonal epsilon term does not
    connect otherwise independent components.
    """
    lambda_penalty = _positive_real(
        "lambda_penalty", lambda_penalty, allow_zero=True
    )
    epsilon = _positive_real("epsilon", epsilon)
    delta = _positive_real("delta", delta)
    tol = _positive_real("tol", tol)
    max_iter = _positive_int("max_iter", max_iter)

    A, Y = _validate_inputs(copy_number_matrix, asv_abundance)

    mag_ids = copy_number_matrix.columns
    n_asvs, n_mags = A.shape
    use_regularization = False

    # ---- ASV–MAG bipartite graph -------------------------------------------
    # Separate internal namespaces prevent collisions when an ASV and a MAG have
    # the same textual label.
    bip = nx.Graph()
    bip.add_nodes_from(("ASV", i) for i in range(n_asvs))
    bip.add_nodes_from(("MAG", j) for j in range(n_mags))
    rows, cols = np.nonzero(A)
    bip.add_edges_from(
        (("ASV", int(i)), ("MAG", int(j))) for i, j in zip(rows, cols)
    )

    # W[j, k] is the number of ASV types shared by MAG j and MAG k. W is
    # populated only for rank-deficient connected components.
    W = np.zeros((n_mags, n_mags), dtype=float)

    for component in nx.connected_components(bip):
        component_asvs = sorted(i for kind, i in component if kind == "ASV")
        component_mags = sorted(j for kind, j in component if kind == "MAG")
        block = A[np.ix_(component_asvs, component_mags)].astype(int)

        rank = spm.Matrix(block.tolist()).rank()
        if rank < len(component_mags):
            use_regularization = True
            for local_asv in range(block.shape[0]):
                linked_local = np.flatnonzero(block[local_asv])
                linked_global = [component_mags[k] for k in linked_local]
                for m_i, m_j in combinations(linked_global, 2):
                    W[m_i, m_j] += 1.0
                    W[m_j, m_i] += 1.0

    if use_regularization and lambda_penalty == 0:
        raise ValueError(
            "lambda_penalty must be positive when rank-deficient components are present."
        )

    # ---- Graph-Laplacian penalty -------------------------------------------
    if use_regularization:
        degree = W.sum(axis=0)
        L = np.diag(degree) - W

        # L contains graph-smoothing edges only for rank-deficient components.
        # epsilon * I is added in the full MAG space to obtain a positive-definite
        # matrix for Cholesky factorization; it does not connect components.
        L_eps = L + epsilon * np.eye(n_mags, dtype=float)
        R = np.linalg.cholesky(L_eps).T  # R.T @ R = L_eps
        zeros_ext = np.zeros(n_mags, dtype=float)

    result = pd.DataFrame(
        0.0,
        index=mag_ids.copy(),
        columns=asv_abundance.columns.copy(),
        dtype=float,
    )

    # ---- IRLS + NNLS, independently for each sample/time point -------------
    for t in range(Y.shape[1]):
        y = Y[:, t]
        mu = y.copy()
        x = np.zeros(n_mags, dtype=float)

        for iteration in range(max_iter):
            # Squared residuals are weighted by 1 / max(mu, delta).
            weight = np.sqrt(np.maximum(mu, delta))
            A_scaled = A / weight[:, None]
            y_scaled = y / weight

            if use_regularization:
                A_ext = np.vstack([A_scaled, np.sqrt(lambda_penalty) * R])
                b_ext = np.concatenate([y_scaled, zeros_ext])
            else:
                A_ext, b_ext = A_scaled, y_scaled

            x_new, _ = nnls(A_ext, b_ext)
            mu_new = A @ x_new

            if iteration == 0:
                x, mu = x_new, mu_new
                continue

            rel_x = np.linalg.norm(x_new - x) / max(np.linalg.norm(x), 1e-12)
            x, mu = x_new, mu_new
            if rel_x < tol:
                break

        # If max_iter is reached before tol is met, return the final iterate,
        # matching the analysis implementation.
        result.iloc[:, t] = x

    return result.astype(float)
