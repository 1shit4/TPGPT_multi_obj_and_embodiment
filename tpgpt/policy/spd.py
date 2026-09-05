"""Regression-safe parameterisation of symmetric positive-definite matrices.

Stiffness and damping labels live on ``S_3^+`` (paper Sec. III-A). Regressing
their six independent entries directly does not keep them there: a fitted mean
can easily be indefinite, which a Cartesian impedance controller would turn into
an unstable negative-stiffness direction.

The log-Cholesky chart avoids this. Writing ``M = L L^T`` with ``L`` lower
triangular and positive diagonal, and storing the log of that diagonal, gives an
unconstrained six-vector whose inverse map is positive definite for *any* input.
Interpolating there is therefore always safe.
"""

from __future__ import annotations

import numpy as np

_TRIL = np.tril_indices(3)
_DIAG_IN_TRIL = np.array([i == j for i, j in zip(*_TRIL)])


def spd_to_log_cholesky(M: np.ndarray) -> np.ndarray:
    """Map ``(n, 3, 3)`` SPD matrices to their ``(n, 6)`` log-Cholesky vectors.

    Args:
        M: Symmetric positive-definite matrices. Mild asymmetry is tolerated
            (the input is symmetrised first); indefinite input is not.
    """
    M = np.asarray(M, dtype=float).reshape(-1, 3, 3)
    M = 0.5 * (M + M.transpose(0, 2, 1))
    # Eigenvalue floor guards against labels that are PSD but singular.
    eigvals, eigvecs = np.linalg.eigh(M)
    floor = np.maximum(1e-12, 1e-12 * np.abs(eigvals).max(axis=1, keepdims=True))
    M = np.einsum(
        "nij,nj,nkj->nik", eigvecs, np.maximum(eigvals, floor), eigvecs
    )
    L = np.linalg.cholesky(M)
    out = L[:, _TRIL[0], _TRIL[1]].copy()
    out[:, _DIAG_IN_TRIL] = np.log(out[:, _DIAG_IN_TRIL])
    return out


def log_cholesky_to_spd(v: np.ndarray) -> np.ndarray:
    """Inverse of :func:`spd_to_log_cholesky`; always returns SPD matrices."""
    v = np.atleast_2d(np.asarray(v, dtype=float))
    entries = v.copy()
    entries[:, _DIAG_IN_TRIL] = np.exp(np.clip(entries[:, _DIAG_IN_TRIL], -30.0, 30.0))
    L = np.zeros((v.shape[0], 3, 3))
    L[:, _TRIL[0], _TRIL[1]] = entries
    return np.einsum("nij,nkj->nik", L, L)
