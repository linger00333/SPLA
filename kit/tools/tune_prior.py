#!/usr/bin/env python3
"""Tune the online prior on labelled videos, without any network (NumPy only).

Replays each video's ground-truth RGB-IR centre pairs through
OnlinePriorEstimator, as if a perfect detector supplied them, and measures
the ONE-STEP-AHEAD error: before frame t is used, how far is theta_t(c_ir)
from the labelled RGB centre (IR px)? That is the error the network would
have to correct at frame t.

To mimic a real detector, a pair is observed only with probability --obs-rate
and gets Gaussian noise of --obs-noise-px. Run it on TRAIN to choose
process_sigma (how fast the prior may drift) and meas_sigma; confirm on val.

Reference rows: "dataset" = training-set mean transform, never updated;
"oracle" = the video's own whole-video fit (uses that video's labels: a bound).

Usage:
  python tools/tune_prior.py --root /data/Anti-UAV-RGBT --split train \
      --train-theta /data/antiuav_yolo/seq_theta_train.json --out reports/tune_prior_train.json
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from spla.antiuav_io import KNOWN_LABEL_ISSUES, find_files, list_sequences, read_labels  # noqa: E402
from spla.geometry import apply_theta, fit_transform, params_to_theta, set_zoom_aspect  # noqa: E402
from spla.prior_estimator import OnlinePriorEstimator, PriorConfig, dataset_prior  # noqa: E402


def set_aspect_from(theta_json: dict) -> None:
    """Restore the zoom model's aspect factor from a seq_theta_*.json table."""
    v = next(iter(theta_json.values()))
    if "zoom_aspect" in v:
        import spla.geometry as g
        g.ZOOM_ASPECT = float(v["zoom_aspect"])
    elif "ir_wh" in v and "rgb_wh" in v:
        set_zoom_aspect(v["ir_wh"], v["rgb_wh"])


def video_wh(path, default):
    try:
        import cv2
        cap = cv2.VideoCapture(str(path))
        wh = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        cap.release()
        return wh if wh[0] > 0 and wh[1] > 0 else tuple(default)
    except Exception:
        return tuple(default)


def load_pairs(seq, box_format, ir_default=(640, 512), rgb_default=(1920, 1080)):
    """Normalised IR and RGB centres on frames where both exist, plus frame indices and IR size."""
    fs = find_files(seq)
    ei, bi = read_labels(fs["ir_label"], box_format)
    er, br = read_labels(fs["rgb_label"], box_format)
    (wi, hi) = video_wh(fs["ir_video"], ir_default) if fs["ir_video"] else tuple(ir_default)
    (wv, hv) = video_wh(fs["rgb_video"], rgb_default) if fs["rgb_video"] else tuple(rgb_default)
    n = min(len(ei), len(er))
    ok = (ei[:n] == 1) & (er[:n] == 1)
    t = np.where(ok)[0]
    c_ir = (bi[t, :2] + bi[t, 2:] / 2) / [wi, hi]
    c_rgb = (br[t, :2] + br[t, 2:] / 2) / [wv, hv]
    return t, c_ir, c_rgb, (wi, hi)


def replay(t, c_ir, c_rgb, size, cfg, mu, cov, obs_rate, obs_noise_px, rng, return_thetas=False,
           query_frames=None):
    """One-step-ahead errors (IR px) of the online prior over one video.

    With return_thetas, also the transform the filter held BEFORE using frame
    t[k] (causal: what online inference would feed the network at that frame).
    """
    est = OnlinePriorEstimator(cfg, init_params=mu, init_cov=cov)
    errs = np.empty(len(t))
    thetas = []
    post_thetas = []
    noise = obs_noise_px / np.asarray(size, float)
    last = t[0] - 1
    for k in range(len(t)):
        for _ in range(int(t[k] - last)):          # one predict per elapsed frame
            est.predict()
        last = t[k]
        errs[k] = np.linalg.norm((apply_theta(est.theta, c_ir[k:k + 1])[0] - c_rgb[k]) * size)
        if return_thetas:
            thetas.append(est.theta.copy())
        if rng.random() < obs_rate:
            est.update(c_ir[k] + rng.normal(0, noise), c_rgb[k] + rng.normal(0, noise))
        if return_thetas and query_frames is not None:
            post_thetas.append(est.theta.copy())
    if return_thetas and query_frames is not None:
        selected = []
        for frame in query_frames:
            k = int(np.searchsorted(t, frame, side="left"))
            if k < len(t) and t[k] == frame:
                selected.append(thetas[k])
            else:
                selected.append(post_thetas[k - 1] if k > 0 else params_to_theta(mu, cfg.model))
        thetas = selected
    return (errs, thetas) if return_thetas else errs


def write_meta_priors(meta_path, data, cfg, mu, cov, th_ds, args):
    """Store the causal replayed prior of every image as "theta_prior" in a spla_meta file.

    Training on this prior (instead of the frozen per-video fit) gives the
    network the kind of prior error it will meet at test time.
    """
    meta_path = Path(meta_path)
    meta = json.loads(meta_path.read_text())
    rng = np.random.default_rng(1)
    by_seq = {}
    frames_by_seq = {}
    for m in meta.values():
        frames_by_seq.setdefault(m["seq"], []).append(m["frame"])
    for name, t, a, b, size in data:
        frames = sorted(set(frames_by_seq.get(name, [])))
        _, th = replay(t, a, b, size, cfg, mu, cov, args.obs_rate, args.obs_noise_px, rng,
                       return_thetas=True, query_frames=frames)
        by_seq[name] = dict(zip(frames, th))
    n_set = 0
    for stem, m in meta.items():
        if m["seq"] in by_seq:
            m["theta_prior"] = by_seq[m["seq"]][m["frame"]].tolist()
            n_set += 1
        else:
            m["theta_prior"] = th_ds.tolist()      # videos skipped (few pairs, known label issues)
    meta_path.write_text(json.dumps(meta))
    merged = meta_path.with_name("spla_meta.json")
    if merged.exists() and merged != meta_path:
        mm = json.loads(merged.read_text())
        for stem, m in meta.items():
            if stem in mm:
                mm[stem]["theta_prior"] = m["theta_prior"]
        merged.write_text(json.dumps(mm))
    print(f"theta_prior written for {n_set} of {len(meta)} images in {meta_path}"
          + (f" (and {merged.name})" if merged.exists() else ""))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True)
    ap.add_argument("--split", default="train")
    ap.add_argument("--train-theta", required=True, help="seq_theta_train.json (dataset prior and model)")
    ap.add_argument("--box-format", default="xywh", choices=["xywh", "xyxy"])
    ap.add_argument("--process-sigma", nargs="+", type=float, default=[1e-4, 2e-4, 5e-4, 1e-3, 2e-3, 5e-3])
    ap.add_argument("--meas-sigma", nargs="+", type=float, default=[0.004, 0.008, 0.016, 0.032])
    ap.add_argument("--obs-rate", type=float, default=0.5, help="share of frames that yield an observation")
    ap.add_argument("--obs-noise-px", type=float, default=2.0, help="noise added to observed centres (IR px)")
    ap.add_argument("--max-seq", type=int, default=0)
    ap.add_argument("--out", default="tune_prior.json")
    ap.add_argument("--tolerance", type=float, default=0.10,
                    help="'chosen' = the SLOWEST prior (smallest process_sigma) within this share of the best "
                         "median error: less prone to chasing its own detections online")
    ap.add_argument("--write-meta", default=None,
                    help="spla_meta_<split>.json to receive the replayed prior ('theta_prior') with the chosen setting")
    args = ap.parse_args(argv)

    train = json.loads(Path(args.train_theta).read_text())
    model = next(iter(train.values()))["model"]
    set_aspect_from(train)
    mu, cov = dataset_prior(np.array([v["params"] for v in train.values() if v["n_pairs"] > 0]))
    th_ds = params_to_theta(mu, model)
    seqs = [s for s in list_sequences(Path(args.root) / args.split,
                                      need=("ir_label", "rgb_label"))
            if s.name not in KNOWN_LABEL_ISSUES][: args.max_seq or None]
    data = [(s.name, *load_pairs(s, args.box_format)) for s in seqs]
    data = [d for d in data if len(d[1]) >= 20]

    def summarise(per_seq):
        med = np.array([np.median(e) for e in per_seq])
        late = np.array([np.median(e[len(e) // 2:]) for e in per_seq])    # after warm-up
        allf = np.concatenate(per_seq)
        return {"median_err_px": float(np.median(med)), "median_err_px_second_half": float(np.median(late)),
                "p90_err_px_all_frames": float(np.percentile(allf, 90)), "n_videos": len(per_seq)}

    report = {"split": args.split, "model": model, "obs_rate": args.obs_rate, "obs_noise_px": args.obs_noise_px,
              "references": {}, "grid": []}
    ds, orc = [], []
    for _, t, a, b, size in data:
        ds.append(np.linalg.norm((apply_theta(th_ds, a) - b) * size, axis=1))
        f = fit_transform(a, b, model)
        orc.append(np.linalg.norm(f.residuals * size, axis=1))
    report["references"]["dataset_prior"] = summarise(ds)
    report["references"]["oracle_per_video"] = summarise(orc)
    for ps, ms in itertools.product(args.process_sigma, args.meas_sigma):
        cfg = PriorConfig(model=model, process_sigma=ps, meas_sigma=ms)
        rng = np.random.default_rng(0)
        per = [replay(t, a, b, size, cfg, mu, cov, args.obs_rate, args.obs_noise_px, rng) for _, t, a, b, size in data]
        report["grid"].append({"process_sigma": ps, "meas_sigma": ms, **summarise(per)})
    report["grid"].sort(key=lambda r: r["median_err_px"])
    report["best"] = report["grid"][0]
    ok = [g for g in report["grid"] if g["median_err_px"] <= (1 + args.tolerance) * report["best"]["median_err_px"]]
    report["chosen"] = min(ok, key=lambda g: (g["process_sigma"], g["median_err_px"]))
    edge = []
    if report["best"]["process_sigma"] == max(args.process_sigma):
        edge.append("process_sigma")
    if report["best"]["meas_sigma"] == max(args.meas_sigma):
        edge.append("meas_sigma")
    report["best_at_grid_edge"] = edge
    Path(args.out).write_text(json.dumps(report, indent=1))
    r = report["references"]
    print(f"videos: {len(data)}  model: {model}")
    print(f"dataset prior      median {r['dataset_prior']['median_err_px']:6.2f} px")
    print(f"oracle per video   median {r['oracle_per_video']['median_err_px']:6.2f} px  (bound, uses the labels)")
    for g in report["grid"][:5]:
        print(f"online ps={g['process_sigma']:.0e} ms={g['meas_sigma']:.3f}  median {g['median_err_px']:6.2f} px"
              f"  second half {g['median_err_px_second_half']:6.2f}  p90 {g['p90_err_px_all_frames']:6.1f}")
    c = report["chosen"]
    print(f"chosen (slowest within {100 * args.tolerance:.0f} % of best): ps={c['process_sigma']:.0e} "
          f"ms={c['meas_sigma']:.3f}  median {c['median_err_px']:.2f} px")
    if edge:
        print(f"WARNING: best value at the grid edge for {edge}; extend the grid")
    print("written:", args.out)
    if args.write_meta:
        cfg = PriorConfig(model=model, process_sigma=c["process_sigma"], meas_sigma=c["meas_sigma"])
        write_meta_priors(args.write_meta, data, cfg, mu, cov, th_ds, args)


if __name__ == "__main__":
    main()
