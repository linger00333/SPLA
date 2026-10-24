"""Online, per-sequence estimator of the IR->RGB transform (the SPLA prior).

The prior is a small Bayesian filter over the transform parameters:

* state: the parameter vector of a model from spla.geometry (translation,
  scale_translation, similarity or affine), with covariance P;
* dynamics: a random walk with small process noise, so slow drift (thermal
  expansion, a loose bracket) is tracked but frame-to-frame noise is not;
* measurement: one RGB-IR correspondence per confident detection or confirmed
  track, p_rgb = D(p_ir) @ params + noise, with noise variance sigma^2 / weight;
* robustness, in three layers:
  1. warm-up: the first ``n_init`` correspondences are buffered and fitted
     jointly with a robust (Huber) fit pulled toward the dataset prior, so a
     single wrong first match cannot capture the filter;
  2. chi-square gating plus Huber-style down-weighting of large innovations;
  3. a guard: after ``max_rejections`` consecutive rejections the filter
     re-fits robustly on the recent window and restarts from that fit, which
     lets a wrong prior recover.

Because the estimator starts from a dataset-level prior (mean and covariance
of the per-sequence transforms seen in training), a few correspondences near
one image location are enough: the poorly observed directions (for example
scale, when the UAV stays near the image centre) stay close to the dataset
prior instead of drifting.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from .geometry import N_PARAMS, apply_theta, const_term, design_rows, fit_transform, identity_params, params_to_theta

CHI2_2DOF_99 = 9.21
CHI2_2DOF_95 = 5.99


@dataclass
class PriorConfig:
    model: str = "similarity"
    meas_sigma: float = 0.004          # normalised units; about 2.5 px on a 640 px IR frame
    process_sigma: float = 2e-4        # per-frame random-walk std of each parameter
    gate_chi2: float = CHI2_2DOF_99    # reject innovations beyond this
    huber_chi2: float = CHI2_2DOF_95   # down-weight beyond this
    min_weight: float = 0.3            # ignore observations with weight below this
    n_init: int = 8                    # correspondences buffered before the first estimate
    init_reg: float = 1e-3             # pull of the warm-up fit toward the dataset prior
    max_rejections: int = 15           # consecutive rejections before the guard fires
    window: int = 60                   # recent observations kept for the guard's re-fit


@dataclass
class PriorState:
    params: np.ndarray
    cov: np.ndarray
    initialised: bool = False
    n_updates: int = 0
    n_rejected: int = 0
    n_guard: int = 0
    consecutive_rejections: int = 0
    history: list = field(default_factory=list)


class OnlinePriorEstimator:
    """Sequence-level prior for SPLA. One instance per video sequence."""

    def __init__(self, cfg: PriorConfig = PriorConfig(),
                 init_params: Optional[np.ndarray] = None,
                 init_cov: Optional[np.ndarray] = None):
        self.cfg = cfg
        k = N_PARAMS[cfg.model]
        p0 = identity_params(cfg.model) if init_params is None else np.asarray(init_params, float)
        c0 = np.eye(k) * 0.05 ** 2 if init_cov is None else np.asarray(init_cov, float)
        self._init = (p0.copy(), c0.copy())
        self.state = PriorState(p0.copy(), c0.copy())
        self._recent = deque(maxlen=cfg.window)

    # -- public API ---------------------------------------------------------
    @property
    def theta(self) -> np.ndarray:
        """Current IR->RGB transform (2x3). Before warm-up it is the dataset prior."""
        return params_to_theta(self.state.params, self.cfg.model)

    @property
    def ready(self) -> bool:
        return self.state.initialised

    def reset(self) -> None:
        p0, c0 = self._init
        self.state = PriorState(p0.copy(), c0.copy())
        self._recent.clear()

    def predict(self) -> None:
        """Advance one frame: random-walk process noise."""
        k = len(self.state.params)
        self.state.cov = self.state.cov + np.eye(k) * self.cfg.process_sigma ** 2

    def update(self, p_ir: np.ndarray, p_rgb: np.ndarray, weight: float = 1.0) -> bool:
        """Fuse one correspondence (normalised coordinates). Returns True if used."""
        cfg, st = self.cfg, self.state
        if weight < cfg.min_weight:
            return False
        p_ir = np.asarray(p_ir, float).reshape(2)
        p_rgb = np.asarray(p_rgb, float).reshape(2)
        self._recent.append((p_ir, p_rgb, float(weight)))
        if not st.initialised:
            if len(self._recent) >= cfg.n_init:
                self._robust_restart()
            return st.initialised
        return self._kalman_update(p_ir, p_rgb, weight)

    def offset_at(self, p_ir: np.ndarray) -> np.ndarray:
        """Predicted RGB-minus-IR offset at IR points (N, 2), normalised units."""
        p = np.asarray(p_ir, float).reshape(-1, 2)
        return apply_theta(self.theta, p) - p

    def offset_std_at(self, p_ir: np.ndarray) -> np.ndarray:
        """Std of the predicted offset (N, 2): the prior's own uncertainty (D4)."""
        p = np.asarray(p_ir, float).reshape(-1, 2)
        out = []
        for q in p:
            d = design_rows(q.reshape(1, 2), self.cfg.model)
            out.append(np.sqrt(np.clip(np.diag(d @ self.state.cov @ d.T), 0, None)))
        return np.array(out)

    # -- internals ----------------------------------------------------------
    def _robust_restart(self) -> None:
        cfg, st = self.cfg, self.state
        p0, c0 = self._init
        pts = list(self._recent)
        a = np.array([q[0] for q in pts])
        b = np.array([q[1] for q in pts])
        w = np.array([q[2] for q in pts])
        fit = fit_transform(a, b, cfg.model, w=w, reg_identity=cfg.init_reg, prior_params=p0)
        if fit.inlier_fraction < 0.5:          # no consistent majority yet: keep waiting
            return
        st.params = fit.params
        # posterior-like covariance: information from the inliers plus the prior
        inl = fit.weights > 0.5
        d = design_rows(a[inl], cfg.model)
        info = d.T @ d / cfg.meas_sigma ** 2 + np.linalg.inv(c0)
        st.cov = np.linalg.inv(info)
        st.initialised = True
        st.consecutive_rejections = 0
        st.history.append(st.params.copy())

    def _kalman_update(self, p_ir, p_rgb, weight) -> bool:
        cfg, st = self.cfg, self.state
        d = design_rows(p_ir.reshape(1, 2), cfg.model)               # (2, k)
        r = np.eye(2) * cfg.meas_sigma ** 2 / max(weight, 1e-6)
        nu = p_rgb - const_term(p_ir.reshape(1, 2), cfg.model) - d @ st.params
        s = d @ st.cov @ d.T + r
        m2 = float(nu @ np.linalg.solve(s, nu))
        if m2 > cfg.gate_chi2:
            st.n_rejected += 1
            st.consecutive_rejections += 1
            if st.consecutive_rejections >= cfg.max_rejections:
                st.n_guard += 1
                st.initialised = False
                recent = list(self._recent)[-2 * cfg.max_rejections:]
                self._recent.clear()
                self._recent.extend(recent)
                self._robust_restart()
            return False
        if m2 > cfg.huber_chi2:                                      # soft down-weighting
            r = r * (m2 / cfg.huber_chi2)
            s = d @ st.cov @ d.T + r
        gain = st.cov @ d.T @ np.linalg.inv(s)
        st.params = st.params + gain @ nu
        st.cov = (np.eye(len(st.params)) - gain @ d) @ st.cov
        st.cov = 0.5 * (st.cov + st.cov.T)
        st.n_updates += 1
        st.consecutive_rejections = 0
        st.history.append(st.params.copy())
        return True


def dataset_prior(train_params: np.ndarray, shrink: float = 1.0):
    """Mean and covariance of per-sequence parameters fitted on training data.

    Use the result as init_params / init_cov of OnlinePriorEstimator. `shrink`
    > 1 widens the covariance (less confident start).
    """
    a = np.asarray(train_params, float)
    mean = a.mean(0)
    cov = np.cov(a.T) if len(a) > 1 else np.eye(a.shape[1]) * 1e-3
    cov = np.atleast_2d(cov) * shrink + np.eye(a.shape[1]) * 1e-8
    return mean, cov
