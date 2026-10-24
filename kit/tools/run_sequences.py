#!/usr/bin/env python3
"""Sequential evaluation of an SPLA detector with the ONLINE prior (the real test).

NOT EXECUTED HERE (needs PyTorch, OpenCV and a trained model); its NumPy helpers are tested.

Stress-test options: --shift-mode persistent|jitter --shift-px P (E4), --prior-noise-px S
(oracle-noise sweep), --init-offset-px P (wrong start), --zero-prior-every K (observations
from a zero-prior pass every K frames, against confirmation bias).

For every test sequence, frames are processed in order. Before each frame
the prior is the estimator's current transform; after the forward pass, each
confident detection gives one RGB-IR correspondence (IR box centre, plus the
offset SPLA predicted there), which updates the estimator. Nothing from the
test labels reaches the model or the estimator.

Prior modes (same weights, different prior):
  online   OnlinePriorEstimator started from the training-set prior  -> the method
  dataset  the training-set mean transform, never updated            -> "fixed calibration"
  oracle   the per-sequence transform fitted on the test labels      -> upper bound
  zero     identity transform (no prior)                             -> lower bound

Writes, per sequence, Anti-UAV style results (one box or [] per frame,
IR frame coordinates) and a CSV of the prior error against the oracle
transform per frame (convergence curve, experiment E3).

Example:
  python tools/run_sequences.py --weights runs/detect/spla/weights/best.pt \
     --root /data/Anti-UAV-RGBT --split test --out results/online \
     --seq-theta /data/antiuav_yolo/seq_theta_test.json \
     --train-theta /data/antiuav_yolo/seq_theta_train.json --prior online
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import zlib
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from spla.antiuav_io import find_files, list_sequences, read_labels  # noqa: E402
from spla.geometry import canvas_matrix, params_to_theta, set_zoom_aspect, theta_to_params  # noqa: E402
from spla.prior_estimator import OnlinePriorEstimator, PriorConfig, dataset_prior  # noqa: E402


def letterbox(img, imgsz):
    """Resize the long side to imgsz (ceil, capped) and pad to a centred square, like the fork."""
    import cv2
    h0, w0 = img.shape[:2]
    lm = canvas_matrix(w0, h0, imgsz)
    nw, nh, px, py = int(round(lm[0, 0])), int(round(lm[1, 1])), lm[0, 2], lm[1, 2]
    im = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA if nw < w0 else cv2.INTER_LINEAR)
    top, left = int(round(py - 0.1)), int(round(px - 0.1))
    out = np.full((imgsz, imgsz) + im.shape[2:], 114, im.dtype)
    out[top:top + nh, left:left + nw] = im
    return out, lm


def observations_to_normalised(lm, ctr_px, off_px):
    """Turn detector outputs on the canvas into estimator observations.

    ctr_px: (N, 2) detection centres in canvas pixels (IR grid).
    off_px: (N, 2) SPLA total offsets there, canvas pixels (RGB minus IR).
    lm:     3x3 map normalised image coords -> canvas px (canvas_matrix).
    Returns p_ir, p_rgb (N, 2) in NORMALISED image coordinates, the units of
    the estimator's state (spla.geometry / PriorConfig.meas_sigma).
    """
    li = np.linalg.inv(lm)
    ctr = np.asarray(ctr_px, float).reshape(-1, 2)
    off = np.asarray(off_px, float).reshape(-1, 2)
    p_ir = ctr @ li[:2, :2].T + li[:2, 2]
    p_rgb = (ctr + off) @ li[:2, :2].T + li[:2, 2]
    return p_ir, p_rgb


def theta_error_px(th_a, th_b, w, h, grid=5):
    """Mean distance (IR px) between two transforms over a grid of image points."""
    u = np.stack(np.meshgrid(np.linspace(0.05, 0.95, grid), np.linspace(0.05, 0.95, grid)), -1).reshape(-1, 2)
    a = u @ th_a[:, :2].T + th_a[:, 2]
    b = u @ th_b[:, :2].T + th_b[:, 2]
    return float(np.mean(np.linalg.norm((a - b) * [w, h], axis=1)))


def seq_rng(name: str, salt: str = ""):
    """Deterministic per-sequence random generator (same draw in every run)."""
    return np.random.default_rng(zlib.crc32((salt + name).encode()))


def synthetic_shift(mode: str, px: float, seq_rng_, frame_rng):
    """RGB-only shift (IR px) for one frame: persistent = one vector per video, jitter = new each frame."""
    if mode == "none" or px <= 0:
        return np.zeros(2)
    rng = seq_rng_ if mode == "persistent" else frame_rng
    a = rng.uniform(0, 2 * np.pi)
    return px * np.array([np.cos(a), np.sin(a)])


def translate_theta(th, d_px, w, h):
    """IR->RGB transform after the RGB image is shifted by d_px (IR px of the stretched RGB frame)."""
    out = np.array(th, float).copy()
    out[:, 2] += np.asarray(d_px, float) / np.array([w, h])
    return out


def model_input(rgb, ir, in_ch, input_mode='auto', blank='none'):
    """Build the channel tensor without mutating either source image."""
    if in_ch not in (3, 4, 6) or (input_mode != 'auto' and in_ch != 3):
        raise ValueError('Explicit modality selection requires a 3-channel detector')
    if blank == 'rgb':
        rgb = np.full_like(rgb, 114)
    elif blank == 'ir':
        ir = np.full_like(ir, 114)
    if in_ch == 3:
        return rgb if input_mode == 'rgb' else np.repeat(ir[..., None], 3, -1)
    return np.concatenate([rgb, ir[..., None]], -1)


def frame_box(box, lm, ir_width, output_wh):
    """Convert a canvas xyxy box to xywh in the specified original frame."""
    # The released converter and this runner stretch RGB to the IR dimensions.
    return [(box[0] - lm[0, 2]) / lm[0, 0] * output_wh[0],
            (box[1] - lm[1, 2]) / lm[1, 1] * output_wh[1],
            (box[2] - box[0]) / lm[0, 0] * output_wh[0],
            (box[3] - box[1]) / lm[1, 1] * output_wh[1]]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--root", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--out", required=True)
    ap.add_argument("--seq-theta", required=True, help="seq_theta_<split>.json (oracle; used only for mode oracle and for scoring the prior)")
    ap.add_argument("--train-theta", required=True, help="seq_theta_train.json (dataset prior)")
    ap.add_argument("--prior", default="online", choices=["online", "dataset", "oracle", "zero"])
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--conf", type=float, default=0.25, help="detection threshold for the output")
    ap.add_argument("--obs-conf", type=float, default=0.5, help="detections above this update the prior")
    ap.add_argument("--level", type=int, default=0, help="which SPLA layer (0=P3) supplies observations")
    ap.add_argument("--zero-prior-every", type=int, default=0,
                    help="every k frames take the prior's observations from a zero-prior pass (anti confirmation bias); 0 = off")
    ap.add_argument("--box-format", default="xywh", choices=["xywh", "xyxy"])
    ap.add_argument("--process-sigma", type=float, default=None, help="online prior drift; from tools/tune_prior.py")
    ap.add_argument("--meas-sigma", type=float, default=None, help="online prior observation noise; from tools/tune_prior.py")
    # stress tests (E4, prior-noise sweep, wrong start)
    ap.add_argument("--shift-mode", default="none", choices=["none", "persistent", "jitter"],
                    help="synthetic RGB-only shift: one vector per video, or a new one every frame")
    ap.add_argument("--shift-px", type=float, default=0.0, help="shift size in IR px")
    ap.add_argument("--prior-noise-px", type=float, default=0.0, help="oracle mode: per-video translation error (IR px)")
    ap.add_argument("--init-offset-px", type=float, default=0.0, help="online mode: start the prior this far off (IR px)")
    ap.add_argument("--max-seq", type=int, default=0, help="only the first k sequences (smoke tests)")
    ap.add_argument('--max-frames', type=int, default=0, help='per-sequence frame limit for runtime validation only')
    ap.add_argument("--blank", default="none", choices=["none", "rgb", "ir"],
                    help="modality ablation: replace one input with flat grey (the letterbox fill, 114) on every "
                         "frame, to measure how much the model relies on it")
    ap.add_argument("--input", default="auto", choices=["auto", "ir", "rgb"],
                    help="3-channel models only: feed the IR frame (A0) or the stretched RGB frame (an RGB-only "
                         "detector). With rgb, saved boxes are in ORIGINAL RGB pixels, for scoring against RGB labels")
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    import cv2
    import torch
    from spla.checkpoints import load_inference_model
    from ultralytics.utils.ops import non_max_suppression
    from spla.modules import SPLA, prior_field_from_theta, sample_at

    model = load_inference_model(args.weights, device=torch.device(args.device))
    model = model.float().eval()
    spla_layers = [m for m in model.modules() if isinstance(m, SPLA)]
    has_spla = bool(spla_layers)
    obs_layer = spla_layers[args.level] if has_spla else None
    in_ch = int(getattr(model, "yaml", {}).get("ch", 6))
    model_input(np.zeros((1, 1, 3), np.uint8), np.zeros((1, 1), np.uint8), in_ch, args.input, args.blank)
    args.output_coordinates = 'RGB_pixels' if in_ch == 3 and args.input == 'rgb' else 'IR_pixels'
    if not has_spla:
        # A0 (IR only, 3 channels) or A1 (RGB + IR, 4 channels): no prior input exists, so
        # these runs measure detection only; the prior is reported as identity
        if args.prior != "zero":
            print(f"model has no SPLA layers ({in_ch} input channels): running with --prior zero")
            args.prior = "zero"

    oracle = json.loads(Path(args.seq_theta).read_text())
    train = json.loads(Path(args.train_theta).read_text())
    model_name = next(iter(train.values()))["model"]
    v0 = next(iter(train.values()))
    if "zoom_aspect" in v0:
        import spla.geometry as g
        g.ZOOM_ASPECT = float(v0["zoom_aspect"])
    elif "ir_wh" in v0:
        set_zoom_aspect(v0["ir_wh"], v0["rgb_wh"])
    mu, cov = dataset_prior(np.array([v["params"] for v in train.values() if v["n_pairs"] > 0]))
    cfg = PriorConfig(model=model_name)
    if args.process_sigma is not None:
        cfg.process_sigma = args.process_sigma
    if args.meas_sigma is not None:
        cfg.meas_sigma = args.meas_sigma
    ident = np.array([[1.0, 0, 0], [0, 1.0, 0]])
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    curve_rows = []

    def forward(rgb, ir, th, lm):
        th_c = (lm @ np.vstack([th, [0, 0, 1]]) @ np.linalg.inv(lm))[:2]
        x = model_input(rgb, ir, in_ch, args.input, args.blank)
        x = torch.from_numpy(x.transpose(2, 0, 1)[None].astype(np.float32) / 255).to(args.device)
        if in_ch == 6:
            thc = torch.as_tensor(th_c[None], dtype=torch.float32, device=args.device)
            x = torch.cat([x, prior_field_from_theta(thc, args.imgsz, args.imgsz, 1)], 1)
        with torch.no_grad():
            pred = model(x)
            det = non_max_suppression(pred, conf_thres=0.001, iou_thres=0.6, max_det=10)[0]
        return det, (obs_layer.cache["total_px"].clone() if has_spla else None)

    def offset_at(field, pts_canvas):
        p = torch.as_tensor(np.asarray(pts_canvas, np.float32).reshape(-1, 2), device=field.device)
        return sample_at(field, p, torch.zeros(len(p), device=field.device), obs_layer.stride).cpu().numpy()

    seqs = list_sequences(Path(args.root) / args.split, need=("ir_video", "rgb_video", "ir_label", "rgb_label"))
    for seq in seqs[: args.max_seq or None]:
        fs = find_files(seq)
        ir_e, ir_b = read_labels(fs["ir_label"], args.box_format)
        rgb_e, rgb_b = read_labels(fs["rgb_label"], args.box_format)
        cap_i, cap_v = cv2.VideoCapture(str(fs["ir_video"])), cv2.VideoCapture(str(fs["rgb_video"]))
        wi, hi = int(cap_i.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap_i.get(cv2.CAP_PROP_FRAME_HEIGHT))
        wv, hv = int(cap_v.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap_v.get(cv2.CAP_PROP_FRAME_HEIGHT))
        s_rng, f_rng = seq_rng(seq.name, "shift"), seq_rng(seq.name, "jitter")
        d_persist = synthetic_shift(args.shift_mode, args.shift_px, s_rng, None) if args.shift_mode == "persistent" else np.zeros(2)
        th_seq = np.asarray(oracle[seq.name]["theta"]) if seq.name in oracle else None
        th_oracle = None if th_seq is None else translate_theta(th_seq, d_persist, wi, hi)
        if th_oracle is not None and args.prior_noise_px > 0:
            th_oracle = translate_theta(th_oracle, seq_rng(seq.name, "noise").normal(0, args.prior_noise_px, 2), wi, hi)
        init = mu
        if args.init_offset_px > 0:
            a = seq_rng(seq.name, "init").uniform(0, 2 * np.pi)
            th0 = translate_theta(params_to_theta(mu, model_name), args.init_offset_px * np.array([np.cos(a), np.sin(a)]), wi, hi)
            init = theta_to_params(th0, model_name)
        est = OnlinePriorEstimator(cfg, init_params=init, init_cov=cov)
        results, dets_all, t = [], [], 0
        while True:
            if args.max_frames and t >= args.max_frames:
                break
            ok_i, fi = cap_i.read()
            ok_v, fv = cap_v.read()
            if not (ok_i and ok_v):
                break
            fv = cv2.resize(fv, (wi, hi), interpolation=cv2.INTER_AREA)
            d_t = d_persist if args.shift_mode != "jitter" else synthetic_shift("jitter", args.shift_px, None, f_rng)
            if np.any(d_t):
                fv = cv2.warpAffine(fv, np.float32([[1, 0, d_t[0]], [0, 1, d_t[1]]]), (wi, hi))
            ir, lm = letterbox(cv2.cvtColor(fi, cv2.COLOR_BGR2GRAY), args.imgsz)
            rgb, _ = letterbox(cv2.cvtColor(fv, cv2.COLOR_BGR2RGB), args.imgsz)
            # the prior for THIS frame (never from this frame's labels, except oracle mode)
            if args.prior == "online":
                est.predict()
                th = est.theta
            elif args.prior == "dataset":
                th = params_to_theta(mu, model_name)
            elif args.prior == "oracle":
                th = th_oracle if th_oracle is not None else ident
            else:
                th = ident
            det, total = forward(rgb, ir, th, lm)
            s, px, py = lm[0, 0] / wi, lm[0, 2], lm[1, 2]
            to_ir = lambda b: [(b[0] - px) / s, (b[1] - py) / s, (b[2] - b[0]) / s, (b[3] - b[1]) / s]
            if in_ch == 3 and args.input == "rgb":       # stretched-RGB frame -> original RGB pixels
                to_ir = lambda b: frame_box(b, lm, wi, (wv, hv))
            dets_all.append([to_ir(d[:4].tolist()) + [float(d[4])] for d in det])
            results.append(to_ir(det[0, :4].tolist()) if len(det) and float(det[0, 4]) >= args.conf else [])
            # observations for the prior: confident detections and the offset SPLA found there
            if args.prior == "online" and has_spla and len(det):
                good = det[det[:, 4] >= args.obs_conf]
                if len(good):
                    ctr = ((good[:, :2] + good[:, 2:4]) / 2).cpu().numpy()
                    field = total
                    if args.zero_prior_every > 0 and t % args.zero_prior_every == 0:
                        _, field = forward(rgb, ir, ident, lm)
                    p_ir, p_rgb = observations_to_normalised(lm, ctr, offset_at(field, ctr))
                    for a_, b_, conf in zip(p_ir, p_rgb, good[:, 4].cpu().numpy()):
                        est.update(a_, b_, weight=float(conf))
            # alignment error at the ground-truth target (evaluation only)
            align_err, prior_tgt = float("nan"), float("nan")
            if t < min(len(ir_e), len(rgb_e)) and ir_e[t] and rgb_e[t]:
                # the prior's error AT THE TARGET (what the network must correct there), IR px;
                # prior_error_px below is over the whole image, where scale errors dominate
                c_ir0 = (ir_b[t, :2] + ir_b[t, 2:] / 2) / [wi, hi]
                c_rgb0 = (rgb_b[t, :2] + rgb_b[t, 2:] / 2) / [wv, hv] + d_t / [wi, hi]
                prior_tgt = float(np.linalg.norm((c_ir0 @ th[:, :2].T + th[:, 2] - c_rgb0) * [wi, hi]))
            if has_spla and t < min(len(ir_e), len(rgb_e)) and ir_e[t] and rgb_e[t]:
                c_ir = (ir_b[t, :2] + ir_b[t, 2:] / 2) / [wi, hi]
                c_rgb = (rgb_b[t, :2] + rgb_b[t, 2:] / 2) / [wv, hv] + d_t / [wi, hi]
                ci = c_ir @ lm[:2, :2].T + lm[:2, 2]
                true_off = (c_rgb @ lm[:2, :2].T + lm[:2, 2]) - ci
                align_err = float(np.linalg.norm(offset_at(total, ci)[0] - true_off) / s)
            prior_err = theta_error_px(th, th_oracle, wi, hi) if th_oracle is not None else float("nan")
            curve_rows.append([seq.name, t, prior_err, align_err, prior_tgt, int(est.ready),
                               est.state.n_updates, est.state.n_rejected, est.state.n_guard])
            t += 1
        cap_i.release(), cap_v.release()
        (out / f"{seq.name}_IR.txt").write_text(json.dumps({"res": results}))
        (out / f"{seq.name}_dets.json").write_text(json.dumps(dets_all))
        print(f"{seq.name}: frames={t} prior_updates={est.state.n_updates} guard={est.state.n_guard}")

    with open(out / "prior_curve.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["seq", "frame", "prior_error_px", "align_error_px", "prior_target_error_px", "ready",
                    "n_updates", "n_rejected", "n_guard"])
        w.writerows(curve_rows)
    (out / "run_config.json").write_text(json.dumps(vars(args), indent=1))


if __name__ == "__main__":
    main()
