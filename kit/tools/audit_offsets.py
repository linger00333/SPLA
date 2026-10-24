#!/usr/bin/env python3
"""Offset audit for Anti-UAV RGB-IR video pairs (objective O1, Gate G1, D3).

For every sequence it measures how the RGB box relates to the IR box over
time, using only frames where both modalities say the UAV exists.

It reports, per sequence and overall:
  * translation-only statistics: per-sequence mean offset, within-sequence
    and between-sequence variance, and their ratio (the H1 criterion);
  * robust fits of four transform models (translation, scale+translation,
    similarity, affine) with residual RMS and BIC, so you can see whether a
    single offset is enough or the fields of view differ;
  * a time-lag estimate tau (frames; positive = RGB lags IR), from the
    dependence of residuals on target velocity (direction D3);
  * the correlation between residual size and target speed;
  * an interpolation flag: the share of frames whose boxes lie on straight
    lines between key frames, which marks coarse (interpolated) labels.

Expected layout (the official Anti-UAV release):
  ROOT/<split>/<sequence>/{IR_label.json | infrared.json}, {RGB_label.json | visible.json}
  ROOT/<split>/<sequence>/{infrared.mp4 | IR.mp4}   (optional, used to read frame size)
  ROOT/<split>/<sequence>/{visible.mp4 | RGB.mp4}   (optional)
(file names are detected by spla/antiuav_io.py)
Each label file has "exist": [0/1, ...] and "gt_rect": [[x, y, w, h] or [], ...].

Example:
  python tools/audit_offsets.py --root /data/Anti-UAV-RGBT --split train \
      --out audit_train --ir-size 640 512 --rgb-size 1920 1080
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from spla.antiuav_io import KNOWN_LABEL_ISSUES, find_files, list_sequences  # noqa: E402
from spla.antiuav_io import read_labels as _read_labels  # noqa: E402
from spla.geometry import MODELS, N_PARAMS, apply_theta, fit_transform, set_zoom_aspect  # noqa: E402

BOX_FORMAT = "xywh"


# --------------------------------------------------------------------------
def read_labels(path: Path):
    return _read_labels(path, BOX_FORMAT)


def video_size(path: Path):
    try:
        import cv2
    except ImportError:
        return None
    if not path.exists():
        return None
    cap = cv2.VideoCapture(str(path))
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    return (w, h) if w > 0 and h > 0 else None


def centres(boxes: np.ndarray, size) -> np.ndarray:
    w, h = size
    c = np.stack([boxes[:, 0] + boxes[:, 2] / 2, boxes[:, 1] + boxes[:, 3] / 2], 1)
    return c / np.array([w, h], float)


def interpolated_mask(c_px: np.ndarray, valid: np.ndarray, eps_px: float = 0.25,
                      min_run: int = 5) -> np.ndarray:
    """Frames inside runs of near-zero second difference (piecewise-linear tracks)."""
    n = len(c_px)
    flat = np.zeros(n, bool)
    for t in range(1, n - 1):
        if valid[t - 1] and valid[t] and valid[t + 1]:
            dd = c_px[t + 1] - 2 * c_px[t] + c_px[t - 1]
            flat[t] = np.all(np.abs(dd) < eps_px)
    out = np.zeros(n, bool)
    t = 0
    while t < n:
        if flat[t]:
            s = t
            while t < n and flat[t]:
                t += 1
            if t - s >= min_run:
                out[max(s - 1, 0):min(t + 1, n)] = True
        else:
            t += 1
    return out


def central_velocity(c: np.ndarray, valid: np.ndarray) -> np.ndarray:
    v = np.full_like(c, np.nan)
    for t in range(1, len(c) - 1):
        if valid[t - 1] and valid[t + 1]:
            v[t] = (c[t + 1] - c[t - 1]) / 2.0
    return v


def estimate_lag(c_ir, c_rgb, v_ir, model="similarity", iters=4):
    """Joint fit of theta and a scalar time lag tau (frames).

    Model: c_rgb(t) = theta(c_ir(t - tau)) ~= theta(c_ir(t) - tau * v_ir(t)).
    Returns (tau, se_tau, rms_without_lag, rms_with_lag, fit).
    """
    ok = np.all(np.isfinite(v_ir), 1)
    if ok.sum() < 10:
        return np.nan, np.nan, np.nan, np.nan, None
    a, b, v = c_ir[ok], c_rgb[ok], v_ir[ok]
    fit0 = fit_transform(a, b, model)
    tau = 0.0
    fit = fit0
    for _ in range(iters):
        fit = fit_transform(a - tau * v, b, model)
        av = v @ fit.theta[:, :2].T                     # A v
        r = b - apply_theta(fit.theta, a)               # residual without lag
        w = fit.weights[:, None]
        denom = float(np.sum(w * av * av))
        if denom < 1e-15:
            break
        tau = float(-np.sum(w * r * av) / denom)
    r_lag = b - apply_theta(fit.theta, a - tau * v)
    dof = max(2 * len(a) - fit.params.size - 1, 1)
    s2 = float(np.sum(r_lag ** 2) / dof)
    av = v @ fit.theta[:, :2].T
    se = float(np.sqrt(s2 / max(np.sum(av * av), 1e-15)))
    rms_lag = float(np.sqrt(np.mean(np.sum(r_lag ** 2, 1))))
    return tau, se, fit0.rms, rms_lag, fit


def spearman(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 5:
        return np.nan
    rx = np.argsort(np.argsort(x[ok]))
    ry = np.argsort(np.argsort(y[ok]))
    return float(np.corrcoef(rx, ry)[0, 1])



# --------------------------------------------------------------------------
# Residual diagnostics: where does the per-frame residual come from?
# --------------------------------------------------------------------------
def _err_px(res_norm, size):
    """Per-pair error in IR pixels (x scaled by width, y by height)."""
    return np.linalg.norm(res_norm * np.asarray(size, float), axis=1)


def label_noise_px(c_norm, valid, interp, size):
    """Upper bound on box-centre annotation noise (IR px) from second differences.

    For white noise of std s per axis, the second difference has std s*sqrt(6).
    True target acceleration adds to it, so this is an upper bound.
    """
    t = np.arange(1, len(c_norm) - 1)
    ok = valid[t - 1] & valid[t] & valid[t + 1] & ~interp[t]
    if ok.sum() < 10:
        return np.nan
    dd = (c_norm[t + 1] - 2 * c_norm[t] + c_norm[t - 1])[ok] * np.asarray(size, float)
    mad = np.median(np.abs(dd - np.median(dd, 0)), 0)
    return float(np.linalg.norm(1.4826 * mad / np.sqrt(6)))


def windowed_errors(a, b, t, model, size, window, prior_params, min_pairs=15):
    """Median error (IR px) when the transform is re-fitted in windows of `window` frames.

    A small ridge toward the whole-video fit keeps windows where the target
    barely moves from producing wild scale estimates.
    """
    errs, scales = [], []
    for k in range(int(t.min() // window), int(t.max() // window) + 1):
        sel = (t >= k * window) & (t < (k + 1) * window)
        if sel.sum() < max(min_pairs, N_PARAMS[model]):
            continue
        f = fit_transform(a[sel], b[sel], model, reg_identity=1e-3, prior_params=prior_params)
        errs.append(_err_px(f.residuals, size))
        scales.append(float(np.sqrt(abs(np.linalg.det(f.theta[:, :2])))))
    if not errs:
        return np.nan, np.nan
    e = np.concatenate(errs)
    zoom = float(max(scales) / min(scales)) if len(scales) > 1 else 1.0
    return float(np.median(e)), zoom

# --------------------------------------------------------------------------
def audit_sequence(seq_dir: Path, ir_size, rgb_size, keyframe_stride=0, model_for_lag="auto"):
    fs = find_files(seq_dir)
    ir_exist, ir_boxes = read_labels(fs["ir_label"])
    rgb_exist, rgb_boxes = read_labels(fs["rgb_label"])
    ir_sz = (video_size(fs["ir_video"]) if fs["ir_video"] else None) or tuple(ir_size)
    rgb_sz = (video_size(fs["rgb_video"]) if fs["rgb_video"] else None) or tuple(rgb_size)
    set_zoom_aspect(ir_sz, rgb_sz)
    n = min(len(ir_exist), len(rgb_exist))
    ir_exist, rgb_exist = ir_exist[:n], rgb_exist[:n]
    ir_boxes, rgb_boxes = ir_boxes[:n], rgb_boxes[:n]
    valid = (ir_exist == 1) & (rgb_exist == 1) & np.all(np.isfinite(ir_boxes), 1) & np.all(np.isfinite(rgb_boxes), 1)
    if keyframe_stride > 0:
        key = np.zeros(n, bool)
        key[::keyframe_stride] = True
        valid_used = valid & key
    else:
        valid_used = valid

    c_ir_px = np.stack([ir_boxes[:, 0] + ir_boxes[:, 2] / 2, ir_boxes[:, 1] + ir_boxes[:, 3] / 2], 1)
    c_rgb_px = np.stack([rgb_boxes[:, 0] + rgb_boxes[:, 2] / 2, rgb_boxes[:, 1] + rgb_boxes[:, 3] / 2], 1)
    interp = interpolated_mask(c_ir_px, valid) | interpolated_mask(c_rgb_px, valid)
    c_ir = centres(ir_boxes, ir_sz)
    c_rgb = centres(rgb_boxes, rgb_sz)
    v_ir = central_velocity(c_ir, valid)

    row = {"sequence": seq_dir.name, "frames": n, "both_exist": int(valid.sum()),
           "used": int(valid_used.sum()), "interp_share": float(interp[valid].mean()) if valid.any() else np.nan,
           "ir_w": ir_sz[0], "ir_h": ir_sz[1], "rgb_w": rgb_sz[0], "rgb_h": rgb_sz[1]}
    if valid_used.sum() < 10:
        row["status"] = "too_few_frames"
        return row, None
    a, b = c_ir[valid_used], c_rgb[valid_used]
    off = b - a
    row.update({"mean_dx": float(off[:, 0].mean()), "mean_dy": float(off[:, 1].mean()),
                "within_var": float(np.sum(off.var(0))),
                "speed_resid_spearman": np.nan, "status": "ok"})
    for m in MODELS:
        try:
            f = fit_transform(a, b, m)
            row[f"rms_{m}"] = f.rms
            row[f"rms_px_{m}"] = f.rms * ir_sz[0]
            row[f"bic_{m}"] = f.bic
        except ValueError:
            row[f"rms_{m}"] = row[f"rms_px_{m}"] = row[f"bic_{m}"] = np.nan
    # share of the within-sequence offset variance that one fixed transform per
    # sequence explains (1.0 = the geometry is perfectly persistent)
    for m in MODELS:
        rms = row.get(f"rms_{m}", np.nan)
        row[f"explained_{m}"] = float(1 - rms ** 2 / row["within_var"]) if row["within_var"] > 0 else np.nan
    bics = {m: row.get(f"bic_{m}", np.nan) for m in MODELS}
    row["best_model"] = min((k for k in bics if np.isfinite(bics[k])), key=lambda k: bics[k], default="")
    lag_model = row["best_model"] if model_for_lag == "auto" else model_for_lag
    lag_model = lag_model or "scale_translation"
    row["lag_model"] = lag_model
    f_best = fit_transform(a, b, lag_model)
    row["theta"] = json.dumps(np.round(f_best.theta, 6).tolist())
    size = (ir_sz[0], ir_sz[1])
    e_whole = _err_px(f_best.residuals, size)
    row["median_err_px"] = float(np.median(e_whole))            # typical per-frame residual
    row["plain_rms_px"] = float(np.sqrt(np.mean(e_whole ** 2)))  # includes outliers
    ctr = np.array([[0.5, 0.5]])
    row["centre_dx_px"], row["centre_dy_px"] = ((apply_theta(f_best.theta, ctr) - ctr)[0] * size).tolist()
    row["scale"] = float(np.sqrt(abs(np.linalg.det(f_best.theta[:, :2]))))
    # lag, fitted with the same model; both RMS values are plain (unweighted)
    vu = v_ir[valid_used]
    tau, se, _, _, f_lag = estimate_lag(a, b, vu, lag_model)
    okv = np.all(np.isfinite(vu), 1)
    if f_lag is not None and np.isfinite(tau):
        r0 = _err_px(b[okv] - apply_theta(f_best.theta, a[okv]), size)
        r1 = _err_px(b[okv] - apply_theta(f_lag.theta, a[okv] - tau * vu[okv]), size)
        row.update({"lag_frames": tau, "lag_se": se, "median_err_px_no_lag": float(np.median(r0)),
                    "median_err_px_with_lag": float(np.median(r1)),
                    "lag_gain": float(1 - np.median(r1) / max(np.median(r0), 1e-9))})
    else:
        row.update({"lag_frames": np.nan, "lag_se": np.nan, "lag_gain": np.nan})
    speed = np.linalg.norm(vu, axis=1)
    row["speed_resid_spearman"] = spearman(speed, e_whole)
    # does the geometry drift within the video? (re-fit in time windows)
    t_used = np.where(valid_used)[0]
    for wnd in (50, 100, 200):
        row[f"median_err_px_win{wnd}"], z = windowed_errors(a, b, t_used, lag_model, size, wnd, f_best.params)
        if wnd == 100:
            row["zoom_ratio_win100"] = z
    # how much of the residual could be annotation noise?
    li = label_noise_px(c_ir, valid, interp, size)
    lr = label_noise_px(c_rgb, valid, interp, size)      # RGB normalised coords, expressed at IR size
    row["label_noise_ir_px"], row["label_noise_rgb_px"] = li, lr
    row["label_floor_px"] = float(np.hypot(li * row["scale"], lr)) if np.isfinite(li) and np.isfinite(lr) else np.nan
    return row, off.mean(0)


def summarise(rows, means, interp_threshold=0.3):
    flagged = [r for r in rows if r.get("sequence") in KNOWN_LABEL_ISSUES]
    ok = [r for r in rows if r.get("status") == "ok" and r.get("sequence") not in KNOWN_LABEL_ISSUES]
    out = {"sequences": len(rows), "usable": len(ok),
           "excluded_known_label_issues": {r["sequence"]: KNOWN_LABEL_ISSUES[r["sequence"]] for r in flagged}}
    for stratum, sel in (("all", ok),
                         ("fine", [r for r in ok if r["interp_share"] < interp_threshold]),
                         ("coarse", [r for r in ok if r["interp_share"] >= interp_threshold])):
        if len(sel) < 2:
            out[stratum] = {"n": len(sel)}
            continue
        mu = np.array([[r["mean_dx"], r["mean_dy"]] for r in sel])
        within = float(np.mean([r["within_var"] for r in sel]))
        between = float(np.sum(mu.var(0, ddof=1)))
        best = {}
        for r in sel:
            best[r["best_model"]] = best.get(r["best_model"], 0) + 1
        lags = np.array([r["lag_frames"] for r in sel], float)
        out[stratum] = {
            "n": len(sel),
            "within_var_mean": within,
            "between_var": between,
            "within_over_between": within / between if between > 0 else np.inf,
            "H1_pass(<=0.30)": bool(between > 0 and within / between <= 0.30),
            "median_rms_px": {m: float(np.nanmedian([r[f"rms_px_{m}"] for r in sel])) for m in MODELS},
            "median_explained_by_sequence_transform": {
                m: float(np.nanmedian([r[f"explained_{m}"] for r in sel])) for m in MODELS},
            "best_model_counts": best,
            "median_lag_frames": float(np.nanmedian(lags)),
            "share_lag_significant(|tau|>2se)": float(np.nanmean(
                [abs(r["lag_frames"]) > 2 * r["lag_se"] for r in sel if np.isfinite(r["lag_se"])])),
            "median_speed_resid_spearman": float(np.nanmedian([r["speed_resid_spearman"] for r in sel])),
        }
        out[stratum]["diagnostics"] = diagnose(sel)
    return out


def _med(sel, k):
    v = np.array([r.get(k, np.nan) for r in sel], float)
    return float(np.nanmedian(v)) if np.any(np.isfinite(v)) else float("nan")


def diagnose(sel):
    """Split the per-frame residual into drift, time lag and label noise (IR px).

    Rules of thumb, not statistical tests; the thresholds are stated so they
    can be argued about in the thesis.
    """
    whole = _med(sel, "median_err_px")
    d = {
        "median_err_px_whole_video": whole,
        "median_err_px_window": {w: _med(sel, f"median_err_px_win{w}") for w in (50, 100, 200)},
        "median_label_floor_px": _med(sel, "label_floor_px"),
        "median_label_noise_ir_px": _med(sel, "label_noise_ir_px"),
        "median_label_noise_rgb_px": _med(sel, "label_noise_rgb_px"),
        "median_lag_frames(best model)": _med(sel, "lag_frames"),
        "share_abs_lag_gt_1_frame": float(np.nanmean([abs(r.get("lag_frames", np.nan)) > 1 for r in sel
                                                     if np.isfinite(r.get("lag_frames", np.nan))])),
        "median_lag_gain": _med(sel, "lag_gain"),
        "median_zoom_ratio_win100": _med(sel, "zoom_ratio_win100"),
        "share_zoom_change_gt_5pct": float(np.nanmean([r.get("zoom_ratio_win100", np.nan) > 1.05 for r in sel
                                                      if np.isfinite(r.get("zoom_ratio_win100", np.nan))])),
        "share_videos_err_gt_24px": float(np.mean([r.get("median_err_px", 0) > 24 for r in sel])),
    }
    ctr = np.array([[r.get("centre_dx_px", np.nan), r.get("centre_dy_px", np.nan)] for r in sel], float)
    ctr = ctr[np.all(np.isfinite(ctr), 1)]
    between = float(np.sqrt(np.sum(ctr.var(0, ddof=1)))) if len(ctr) > 1 else float("nan")
    d["between_video_spread_px"] = between          # what a per-video prior removes beyond one calibration
    d["residual_over_between"] = whole / between if between > 0 else float("nan")
    e = np.array([r.get("median_err_px", np.nan) for r in sel], float)
    d["p90_err_px_whole_video"] = float(np.nanpercentile(e, 90)) if np.any(np.isfinite(e)) else float("nan")
    wins = [d["median_err_px_window"][w] for w in (50, 100)]
    best_win = float(np.nanmin(wins)) if np.any(np.isfinite(wins)) else float("nan")
    floor, gain = d["median_label_floor_px"], d["median_lag_gain"]
    d["drift_share"] = float(1 - best_win / whole) if whole > 0 and np.isfinite(best_win) else float("nan")
    at_noise = np.isfinite(floor) and whole <= max(3.0, 1.5 * floor)
    drift = np.isfinite(d["drift_share"]) and d["drift_share"] >= 0.3
    lag = np.isfinite(gain) and gain >= 0.3
    # Gate G1 (revised): does the per-video prior remove most of the misalignment,
    # leaving a residual SPLA's bounded correction (24 px at P3) can handle?
    gate = at_noise or (d["residual_over_between"] <= 0.30 and d["share_videos_err_gt_24px"] <= 0.10)
    d["G1_pass"] = bool(gate)
    parts = []
    if at_noise:
        parts.append("the residual is about the label-noise level")
    if drift:
        parts.append(f"the geometry drifts within videos ({100 * d['drift_share']:.0f} % of the residual): "
                     "use a time-varying prior and tune PriorConfig.process_sigma on train")
    if lag:
        parts.append("a time lag between the cameras explains much of the residual: add the lag to the prior (D3)")
    if gate:
        verdict = ("G1 pass, D1 go: the per-video prior removes most of the misalignment "
                   f"(residual {whole:.1f} px vs {d['between_video_spread_px']:.0f} px between videos)")
        verdict += "; " + "; ".join(parts) if parts else "; the remaining per-frame residual is for SPLA's residual head"
    elif drift or lag:
        verdict = "G1 borderline: " + "; ".join(parts)
    else:
        verdict = ("G1 fail: the per-frame residual is comparable to the between-video spread and not explained by "
                   "drift, lag or label noise; keep the prior only as initialisation of a per-image aligner")
    d["verdict"] = verdict
    return d


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True)
    ap.add_argument("--split", default="train")
    ap.add_argument("--out", default="audit_out")
    ap.add_argument("--ir-size", nargs=2, type=int, default=[640, 512],
                    help="IR frame size if the video cannot be read (W H)")
    ap.add_argument("--rgb-size", nargs=2, type=int, default=[1920, 1080],
                    help="RGB frame size if the video cannot be read (W H)")
    ap.add_argument("--keyframe-stride", type=int, default=0,
                    help="use only every k-th frame (25 = Anti-UAV coarse key frames)")
    ap.add_argument("--lag-model", default="auto", choices=["auto", *MODELS],
                    help="model for the lag fit and diagnostics; auto = each video's best model by BIC")
    ap.add_argument("--box-format", default="xywh", choices=["xywh", "xyxy"],
                    help="gt_rect layout; tools/inspect_dataset.py tells you which")
    args = ap.parse_args(argv)
    global BOX_FORMAT
    BOX_FORMAT = args.box_format

    split_dir = Path(args.root) / args.split
    seqs = list_sequences(split_dir)
    if not seqs:
        raise SystemExit(f"no sequences with IR and RGB labels under {split_dir}; run tools/inspect_dataset.py")
    rows, means = [], []
    for s in seqs:
        row, mu = audit_sequence(s, args.ir_size, args.rgb_size, args.keyframe_stride, args.lag_model)
        rows.append(row)
        if mu is not None:
            means.append(mu)
    os.makedirs(args.out, exist_ok=True)
    keys = sorted({k for r in rows for k in r})
    with open(Path(args.out) / f"sequences_{args.split}.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    summary = summarise(rows, means)
    with open(Path(args.out) / f"summary_{args.split}.json", "w") as f:
        json.dump(summary, f, indent=2, default=float)
    print(json.dumps(summary, indent=2, default=float))


if __name__ == "__main__":
    main()
