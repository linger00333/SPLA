"""Geometry for the sequence-level RGB-IR prior (direction D1, SPLA).

Conventions used everywhere in this kit
---------------------------------------
* Every point is in NORMALISED image coordinates of its own modality:
  x_n = x_px / W, y_n = y_px / H, so (0, 0) is the top-left corner and (1, 1)
  the bottom-right corner of that image.
* The sequence transform ``theta`` maps an IR point to the RGB point that
  shows the same scene location:  p_rgb = theta(p_ir).
  This direction is the one needed for warping: to bring RGB features onto
  the IR grid you sample the RGB feature map at theta(p_ir).
* ``theta`` is stored as a 2x3 affine matrix [[a11, a12, tx], [a21, a22, ty]].
  Simpler models (translation, scale+translation, similarity) are written as
  constrained affine matrices, so the rest of the code never needs to know
  which model was fitted.

Why an affine-family transform and not a single offset?
The two cameras of a rig usually differ in field of view and resolution.
Then the RGB-IR displacement of a target depends on WHERE the target is in the
image (offset = (A - I) p + t), so a constant offset is only correct near one
image position. For distant targets (UAVs at tens to hundreds of metres) the
parallax between two cameras a few centimetres apart is negligible, and a
fixed affine or homography relation between the images holds while the rig
pans and tilts as one body. The audit (tools/audit_offsets.py) decides which
model the data support.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

MODELS = ("translation", "zoom", "scale_translation", "similarity", "affine")
N_PARAMS = {"translation": 2, "zoom": 3, "scale_translation": 4, "similarity": 4, "affine": 6}

# "zoom" = one PIXEL scale plus a shift: the geometry of two cameras on one
# gimbal that differ only in focal length. In normalised coordinates a pixel
# scale s becomes diag(ZOOM_ASPECT * s, s) with
#   ZOOM_ASPECT = (W_ir / W_rgb) / (H_ir / H_rgb)     (0.703 for 640x512 vs 1920x1080)
# Set it once per data set with set_zoom_aspect(); every tool does this from
# the frame sizes, and convert_antiuav.py stores it in seq_theta_*.json.
ZOOM_ASPECT = 1.0


def set_zoom_aspect(ir_wh, rgb_wh) -> float:
    """Set the normalised-coordinate aspect factor of the zoom model; returns it."""
    global ZOOM_ASPECT
    ZOOM_ASPECT = (ir_wh[0] / rgb_wh[0]) / (ir_wh[1] / rgb_wh[1])
    return ZOOM_ASPECT


# --------------------------------------------------------------------------
# Linear parameterisation: p_rgb = J(p_ir) @ params
# --------------------------------------------------------------------------
def design_rows(p_ir: np.ndarray, model: str) -> np.ndarray:
    """Return the (2N, k) design matrix so that p_rgb.ravel() = D @ params + c,
    with c from const_term (non-zero only for the translation model).

    Parameter vectors per model:
      translation        [tx, ty]                      (A = I)
      zoom               [s, tx, ty]                   (A = diag(ZOOM_ASPECT * s, s))
      scale_translation  [sx, sy, tx, ty]              (A = diag(sx, sy))
      similarity         [a, b, tx, ty]                (A = [[a, -b], [b, a]])
      affine             [a11, a12, a21, a22, tx, ty]
    """
    p_ir = np.asarray(p_ir, dtype=float).reshape(-1, 2)
    n = len(p_ir)
    x, y = p_ir[:, 0], p_ir[:, 1]
    one, zero = np.ones(n), np.zeros(n)
    if model == "translation":
        rx = np.stack([one, zero], 1)
        ry = np.stack([zero, one], 1)
    elif model == "zoom":
        rx = np.stack([ZOOM_ASPECT * x, one, zero], 1)
        ry = np.stack([y, zero, one], 1)
    elif model == "scale_translation":
        rx = np.stack([x, zero, one, zero], 1)
        ry = np.stack([zero, y, zero, one], 1)
    elif model == "similarity":
        rx = np.stack([x, -y, one, zero], 1)
        ry = np.stack([y, x, zero, one], 1)
    elif model == "affine":
        rx = np.stack([x, y, zero, zero, one, zero], 1)
        ry = np.stack([zero, zero, x, y, zero, one], 1)
    else:
        raise ValueError(f"unknown model {model!r}; choose from {MODELS}")
    d = np.empty((2 * n, rx.shape[1]))
    d[0::2], d[1::2] = rx, ry
    return d


def const_term(p_ir: np.ndarray, model: str) -> np.ndarray:
    """Constant part c of the model, so that p_rgb.ravel() = D @ params + c.

    Only the translation model has one (A = I is fixed, so p_rgb = p_ir + t).
    """
    p_ir = np.asarray(p_ir, dtype=float).reshape(-1, 2)
    return p_ir.ravel().copy() if model == "translation" else np.zeros(2 * len(p_ir))


def params_to_theta(params: np.ndarray, model: str) -> np.ndarray:
    p = np.asarray(params, dtype=float)
    if model == "translation":
        return np.array([[1.0, 0.0, p[0]], [0.0, 1.0, p[1]]])
    if model == "zoom":
        return np.array([[ZOOM_ASPECT * p[0], 0.0, p[1]], [0.0, p[0], p[2]]])
    if model == "scale_translation":
        return np.array([[p[0], 0.0, p[2]], [0.0, p[1], p[3]]])
    if model == "similarity":
        return np.array([[p[0], -p[1], p[2]], [p[1], p[0], p[3]]])
    if model == "affine":
        return np.array([[p[0], p[1], p[4]], [p[2], p[3], p[5]]])
    raise ValueError(model)


def theta_to_params(theta: np.ndarray, model: str) -> np.ndarray:
    t = np.asarray(theta, dtype=float)
    if model == "translation":
        return np.array([t[0, 2], t[1, 2]])
    if model == "zoom":
        return np.array([0.5 * (t[0, 0] / ZOOM_ASPECT + t[1, 1]), t[0, 2], t[1, 2]])
    if model == "scale_translation":
        return np.array([t[0, 0], t[1, 1], t[0, 2], t[1, 2]])
    if model == "similarity":
        return np.array([0.5 * (t[0, 0] + t[1, 1]), 0.5 * (t[1, 0] - t[0, 1]), t[0, 2], t[1, 2]])
    if model == "affine":
        return np.array([t[0, 0], t[0, 1], t[1, 0], t[1, 1], t[0, 2], t[1, 2]])
    raise ValueError(model)


def identity_params(model: str) -> np.ndarray:
    return theta_to_params(np.array([[1.0, 0, 0], [0, 1.0, 0]]), model)


def apply_theta(theta: np.ndarray, p: np.ndarray) -> np.ndarray:
    """Map IR points (N, 2) to RGB points with a 2x3 affine matrix."""
    p = np.asarray(p, dtype=float).reshape(-1, 2)
    return p @ np.asarray(theta)[:, :2].T + np.asarray(theta)[:, 2]


def to3x3(theta: np.ndarray) -> np.ndarray:
    m = np.eye(3)
    m[:2] = theta
    return m


# --------------------------------------------------------------------------
# Robust fitting
# --------------------------------------------------------------------------
@dataclass
class Fit:
    model: str
    params: np.ndarray
    theta: np.ndarray
    residuals: np.ndarray      # (N, 2) residual p_rgb - theta(p_ir)
    weights: np.ndarray        # (N,) final IRLS weights in [0, 1]
    rms: float                 # weighted RMS residual, normalised units
    sigma: float               # robust scale (MAD-based)
    bic: float                 # Gaussian BIC on inliers, for model choice

    @property
    def inlier_fraction(self) -> float:
        return float(np.mean(self.weights > 0.5))


def fit_transform(p_ir: np.ndarray, p_rgb: np.ndarray, model: str = "similarity",
                  w: Optional[np.ndarray] = None, huber_k: float = 1.345,
                  iters: int = 20, reg_identity: float = 0.0,
                  prior_params: Optional[np.ndarray] = None) -> Fit:
    """Fit p_rgb ~ theta(p_ir) by iteratively reweighted least squares (Huber).

    Args:
      p_ir, p_rgb: (N, 2) corresponding points, normalised coordinates.
      model: one of MODELS.
      w: optional prior weights per correspondence (for example detection score).
      huber_k: Huber threshold in units of the robust residual scale.
      reg_identity: ridge strength pulling the parameters toward
        ``prior_params`` (the identity transform if None); useful when all
        points lie close together and scale or rotation is poorly determined.
      prior_params: parameter vector the ridge pulls toward, for example the
        dataset-level mean transform.
    """
    p_ir = np.asarray(p_ir, float).reshape(-1, 2)
    p_rgb = np.asarray(p_rgb, float).reshape(-1, 2)
    n = len(p_ir)
    k = N_PARAMS[model]
    if n * 2 < k and reg_identity <= 0:
        raise ValueError(f"{n} correspondences are too few for a {model} fit")
    base_w = np.ones(n) if w is None else np.asarray(w, float).reshape(-1)
    d = design_rows(p_ir, model)
    y = p_rgb.ravel() - const_term(p_ir, model)
    p0 = identity_params(model) if prior_params is None else np.asarray(prior_params, float)
    rob = np.ones(n)
    params = p0.copy()
    sigma = 1.0
    for _ in range(iters):
        ww = np.repeat(base_w * rob, 2)
        a = d.T @ (d * ww[:, None]) + reg_identity * np.eye(k)
        b = d.T @ (ww * y) + reg_identity * p0
        params = np.linalg.solve(a, b)
        r = (y - d @ params).reshape(-1, 2)
        rn = np.linalg.norm(r, axis=1)
        sigma = 1.4826 * np.median(rn) + 1e-12
        u = rn / (huber_k * sigma)
        new_rob = np.where(u <= 1.0, 1.0, 1.0 / np.maximum(u, 1e-12))
        if np.allclose(new_rob, rob, atol=1e-6):
            rob = new_rob
            break
        rob = new_rob
    r = (y - d @ params).reshape(-1, 2)
    wfin = base_w * rob
    rms = float(np.sqrt(np.sum(wfin * np.sum(r ** 2, 1)) / max(np.sum(wfin), 1e-12)))
    inl = rob > 0.5
    m = max(int(inl.sum()), 1)
    rss = float(np.sum(r[inl] ** 2))
    bic = 2 * m * np.log(max(rss / (2 * m), 1e-30)) + k * np.log(2 * m)
    return Fit(model, params, params_to_theta(params, model), r, rob, rms, float(sigma), float(bic))


def choose_model(p_ir: np.ndarray, p_rgb: np.ndarray, models=MODELS, **kw) -> dict[str, Fit]:
    """Fit every model and return them keyed by name; the lowest BIC wins."""
    out = {}
    for m in models:
        try:
            out[m] = fit_transform(p_ir, p_rgb, m, **kw)
        except ValueError:
            pass
    return out


# --------------------------------------------------------------------------
# From a transform to a dense offset field on the network canvas
# --------------------------------------------------------------------------
def canvas_matrix(img_w: int, img_h: int, canvas: int) -> np.ndarray:
    """3x3 map from normalised image coordinates to letter-boxed canvas pixels.

    Matches Ultralytics' LetterBox with auto=False, scaleup=True and centred
    padding: the long side becomes ``canvas`` and the short side is padded.
    Both modalities share this matrix because the conversion script resizes
    RGB to the IR frame size before stacking (tools/convert_antiuav.py).
    """
    r = canvas / max(img_w, img_h)
    # Ultralytics' load_image rounds up (math.ceil) and caps at the canvas size
    new_w = min(int(np.ceil(img_w * r)), canvas)
    new_h = min(int(np.ceil(img_h * r)), canvas)
    pad_x, pad_y = (canvas - new_w) / 2.0, (canvas - new_h) / 2.0
    return np.array([[new_w, 0.0, pad_x], [0.0, new_h, pad_y], [0.0, 0.0, 1.0]])


def theta_on_canvas(theta: np.ndarray, img_w: int, img_h: int, canvas: int) -> np.ndarray:
    """Express the IR->RGB transform in canvas pixels: L @ theta @ L^-1."""
    lm = canvas_matrix(img_w, img_h, canvas)
    return (lm @ to3x3(theta) @ np.linalg.inv(lm))[:2]


def offset_field(theta_canvas: np.ndarray, h: int, w: int, stride: int = 1) -> np.ndarray:
    """Dense RGB-minus-IR offset field (2, h, w) in canvas pixels.

    Cell (i, j) of a feature map with the given stride has its centre at
    canvas pixel ((j + 0.5) * stride, (i + 0.5) * stride).
    """
    jj, ii = np.meshgrid(np.arange(w), np.arange(h))
    u = (jj + 0.5) * stride
    v = (ii + 0.5) * stride
    pts = np.stack([u.ravel(), v.ravel()], 1)
    mapped = apply_theta(theta_canvas, pts)
    off = (mapped - pts).T.reshape(2, h, w)
    return off


def compose_augmentation(theta_canvas: np.ndarray, aug_joint: np.ndarray,
                         aug_rgb: Optional[np.ndarray] = None) -> np.ndarray:
    """Update the canvas transform after GPU augmentation.

    aug_joint: 2x3 affine applied to BOTH modalities (canvas -> canvas), e.g. a
      flip, scale or translation used for data augmentation.
    aug_rgb:   2x3 affine applied to the RGB channels only (synthetic
      misalignment jitter), applied after aug_joint.

    If a pixel u in the original canvas moves to G u, the IR point G u pairs
    with the RGB point R G theta u, so the new transform is R G theta G^-1.
    """
    g = to3x3(aug_joint)
    r = np.eye(3) if aug_rgb is None else to3x3(aug_rgb)
    return (r @ g @ to3x3(theta_canvas) @ np.linalg.inv(g))[:2]
