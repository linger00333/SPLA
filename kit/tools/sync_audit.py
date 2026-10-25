#!/usr/bin/env python3
"""Audit temporal synchronisation of the IR and RGB streams, per video and per window.

For each window of --window frames and each candidate lag k in [-max-lag, max-lag],
pair the IR centre at frame t with the RGB centre at frame t + k, fit one
transform (--model, spla.geometry, robust Huber fit) to the pairs, and record
the median residual in IR pixels. The lag with the smallest residual is the
window's estimate. A video is flagged when the best lag beats lag 0 by both
--min-gain-px and the factor --min-ratio.

Positive k means the RGB frame that matches IR frame t is RGB frame t + k (the
RGB stream is late). A per-video spatial transform cannot absorb a lag: the
induced offset is target or camera motion times the lag, and changes sign
with the motion.

Centre sources (choose one or both):
  --ir-run/--rgb-run  saved predictions of two single-modality detectors (A0 and
                      an --input rgb run): label-free, top box with confidence >= --conf
  --root              the dataset annotations (IR and RGB label centres); this is
                      a data audit, it scores no detector

Usage:
  python kit/tools/sync_audit.py --ir-run results/A0_s0 --rgb-run results/A0rgb_s0 \
     --out reports/sync_audit_val.json
  python kit/tools/sync_audit.py --root data/raw --split val --out reports/sync_audit_val_labels.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from spla.antiuav_io import find_files, list_sequences, read_labels  # noqa: E402
from spla.geometry import fit_transform  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from decision_fusion import _read_json, run_sequences  # noqa: E402


def centres_from_dets(dets, wh, conf):
    c = np.full((len(dets), 2), np.nan)
    for t, d in enumerate(dets):
        if d and d[0][4] >= conf:
            x, y, w, h = d[0][:4]
            c[t] = [(x + w / 2) / wh[0], (y + h / 2) / wh[1]]
    return c


def centres_from_labels(exist, boxes, wh):
    c = np.full((len(exist), 2), np.nan)
    ok = exist == 1
    c[ok] = (boxes[ok, :2] + boxes[ok, 2:] / 2) / np.asarray(wh, float)
    return c


def residual_px(a, b, model, ir_wh, min_pairs):
    ok = np.all(np.isfinite(a), 1) & np.all(np.isfinite(b), 1)
    if ok.sum() < min_pairs:
        return None
    fit = fit_transform(a[ok], b[ok], model)
    lin = fit.theta[:, :2]
    # residual in RGB-normalised units -> IR-normalised (through the fitted linear part) -> IR px
    r = np.linalg.solve(lin, fit.residuals.T).T * np.asarray(ir_wh, float)
    return float(np.median(np.linalg.norm(r, axis=1))), int(ok.sum())


def audit_video(ir_c, rgb_c, args, ir_wh):
    n = min(len(ir_c), len(rgb_c))
    out = []
    starts = range(0, n, args.window) if args.window > 0 else [0]
    for s in starts:
        e = n if args.window <= 0 else min(s + args.window, n)
        scores = {}
        for k in range(-args.max_lag, args.max_lag + 1):
            idx = np.arange(s, e)
            j = idx + k
            keep = (j >= 0) & (j < n)
            r = residual_px(ir_c[idx[keep]], rgb_c[j[keep]], args.model, ir_wh, args.min_pairs)
            if r is not None:
                scores[k] = r
        if 0 not in scores:
            out.append({"start": s, "end": e, "best_lag": None})
            continue
        k_best = min(scores, key=lambda k: scores[k][0])
        out.append({"start": s, "end": e, "best_lag": k_best, "resid_px_best": scores[k_best][0],
                    "resid_px_lag0": scores[0][0], "pairs_lag0": scores[0][1]})
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ir-run", type=Path)
    ap.add_argument("--rgb-run", type=Path)
    ap.add_argument("--root", type=Path, help="dataset root with <split>/<video>/ label files")
    ap.add_argument("--split", default="val")
    ap.add_argument("--box-format", default="xywh", choices=["xywh", "xyxy"])
    ap.add_argument("--ir-wh", nargs=2, type=int, default=[640, 512])
    ap.add_argument("--rgb-wh", nargs=2, type=int, default=[1920, 1080])
    ap.add_argument("--conf", type=float, default=0.5)
    ap.add_argument("--model", default="scale_translation")
    ap.add_argument("--max-lag", type=int, default=15)
    ap.add_argument("--window", type=int, default=0, help="frames per window; 0 = whole video")
    ap.add_argument("--min-pairs", type=int, default=40)
    ap.add_argument("--min-gain-px", type=float, default=1.0)
    ap.add_argument("--min-ratio", type=float, default=1.5)
    ap.add_argument("--allow-test", action="store_true")
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args(argv)
    if (args.ir_run is None) != (args.rgb_run is None) or (args.ir_run is None and args.root is None):
        ap.error("give --ir-run and --rgb-run together, or --root")
    if args.split == "test" and not args.allow_test:
        raise SystemExit("refusing split=test (protocol); pass --allow-test to override")

    report = {"split": args.split, "settings": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
              "sources": {}}
    sources = []
    if args.ir_run is not None:
        seqs = run_sequences(args.ir_run)
        if seqs != run_sequences(args.rgb_run):
            raise ValueError("the two runs must contain the same videos")
        for name, d in (("ir", args.ir_run), ("rgb", args.rgb_run)):
            split = _read_json(d / "run_config.json").get("split")
            if split != args.split:
                raise ValueError(f"--{name}-run is split {split}, expected {args.split}")

        def pred_centres(seq):
            di = _read_json(args.ir_run / f"{seq}_dets.json")
            dr = _read_json(args.rgb_run / f"{seq}_dets.json")
            if len(di) != len(dr):
                raise ValueError(f"{seq}: frame counts differ")
            return centres_from_dets(di, args.ir_wh, args.conf), centres_from_dets(dr, args.rgb_wh, args.conf)
        sources.append(("predictions", seqs, pred_centres))
    if args.root is not None:
        dirs = {p.name: p for p in list_sequences(args.root / args.split)}

        def label_centres(seq):
            fs = find_files(dirs[seq])
            ei, bi = read_labels(fs["ir_label"], args.box_format)
            er, br = read_labels(fs["rgb_label"], args.box_format)
            return centres_from_labels(ei, bi, args.ir_wh), centres_from_labels(er, br, args.rgb_wh)
        sources.append(("labels", sorted(dirs), label_centres))

    for name, seqs, get in sources:
        per, flagged = {}, []
        for seq in seqs:
            ir_c, rgb_c = get(seq)
            wins = audit_video(ir_c, rgb_c, args, args.ir_wh)
            per[seq] = wins
            whole = [w for w in wins if w["best_lag"] is not None]
            if whole:
                best = float(np.median([w["resid_px_best"] for w in whole]))
                zero = float(np.median([w["resid_px_lag0"] for w in whole]))
                if zero - best > args.min_gain_px and zero > args.min_ratio * best:
                    flagged.append(seq)
        report["sources"][name] = {"videos": len(seqs), "flagged": flagged, "per_video": per}
        print(f"{name}: {len(flagged)} of {len(seqs)} videos flagged as desynchronised")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=1, allow_nan=False))
    print("written:", args.out)


if __name__ == "__main__":
    main()
