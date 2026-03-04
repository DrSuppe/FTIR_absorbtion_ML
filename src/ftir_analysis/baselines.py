"""Deterministic baseline models for FTIR concentration estimation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class BaselineMetrics:
    mae: np.ndarray
    median_ae: np.ndarray


class NNLSReferenceBaseline:
    """Reference-basis NNLS baseline with per-species concentration calibration."""

    def __init__(self, *, max_iter: int = 300, tol: float = 1e-6):
        self.max_iter = max_iter
        self.tol = tol
        self.basis: np.ndarray | None = None  # shape (Npts, C)
        self.scales: np.ndarray | None = None  # shape (C,)

    def fit(self, x_train: np.ndarray, y_train: np.ndarray) -> "NNLSReferenceBaseline":
        if x_train.ndim != 2 or y_train.ndim != 2:
            raise ValueError("Expected x_train/y_train as 2D arrays")
        if x_train.shape[0] != y_train.shape[0]:
            raise ValueError("x_train and y_train row counts do not match")

        n_samples, n_pts = x_train.shape
        n_classes = y_train.shape[1]

        basis = np.zeros((n_pts, n_classes), dtype=np.float64)
        for j in range(n_classes):
            mask = y_train[:, j] > 0
            if np.any(mask):
                basis[:, j] = np.median(x_train[mask], axis=0)

        coeff = self._solve_nnls_matrix(x_train.astype(np.float64), basis)

        scales = np.ones(n_classes, dtype=np.float64)
        for j in range(n_classes):
            c = coeff[:, j]
            t = y_train[:, j].astype(np.float64)
            denom = float(np.dot(c, c))
            if denom > 0:
                scales[j] = max(0.0, float(np.dot(c, t) / denom))
            else:
                scales[j] = 0.0

        self.basis = basis
        self.scales = scales
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        if self.basis is None or self.scales is None:
            raise RuntimeError("Baseline must be fit() before predict()")
        coeff = self._solve_nnls_matrix(x.astype(np.float64), self.basis)
        pred = coeff * self.scales[None, :]
        return np.clip(pred, 0.0, None).astype(np.float32)

    def evaluate(self, x: np.ndarray, y_true: np.ndarray) -> BaselineMetrics:
        pred = self.predict(x)
        ae = np.abs(pred - y_true)
        return BaselineMetrics(mae=ae.mean(axis=0), median_ae=np.median(ae, axis=0))

    def _solve_nnls_matrix(self, x: np.ndarray, basis: np.ndarray) -> np.ndarray:
        """Solve non-negative least squares for each row in x.

        Uses projected gradient descent with fixed step derived from basis spectral norm.
        """
        btb = basis.T @ basis
        btx = basis.T @ x.T  # shape (C, N)

        l2 = np.linalg.norm(basis, ord=2)
        step = 1.0 / (l2 * l2 + 1e-12)

        c, n = btx.shape
        coeff = np.zeros((c, n), dtype=np.float64)

        for _ in range(self.max_iter):
            grad = btb @ coeff - btx
            coeff_next = np.maximum(0.0, coeff - step * grad)
            delta = np.max(np.abs(coeff_next - coeff))
            coeff = coeff_next
            if delta < self.tol:
                break

        return coeff.T  # shape (N, C)
