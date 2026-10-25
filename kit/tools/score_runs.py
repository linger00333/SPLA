#!/usr/bin/env python3
"""Score run_sequences.py outputs and compare runs with a paired sequence bootstrap.

Metrics, per run (seeds averaged per sequence first):
  SA       state accuracy as in Anti-UAV410's Evaluation_for_SA.py: IoU on frames
           with a target, 1 for an empty prediction on frames without one
  SA_pen   the challenge formula: SA - 0.2 * (share of target frames predicted empty)^0.3
  AP50     IR-frame average precision at IoU 0.5 from all saved detections
  align_px median alignment error at ground-truth targets (IR px), from prior_curve.csv
  prior_px median prior error against the oracle transform (online runs), and
  conv_frm median frames until the prior stays within 2 px of the oracle

Comparisons: every run against --baseline, resampling SEQUENCES (frames in one
video are not independent) with the same draw for both runs; prints the mean
difference, its 95 % interval and the share of draws with a difference <= 0.

Usage:
  python tools/score_runs.py --root /data/Anti-UAV-RGBT --split test \
     --run A3=results/A3_s0,results/A3_s1,results/A3_s2 \
     --run A4=results/A4_s0,results/A4_s1,results/A4_s2 --baseline A3 --out reports/test_scores.json
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from spla.antiuav_io import find_files, list_sequences, read_labels  # noqa: E402
from spla.prediction_io import read_json, validate_frames  # noqa: E402


def iou_xywh(a, b):
    ax2, ay2, bx2, by2 = a[0] + a[2], a[1] + a[3], b[0] + b[2], b[1] + b[3]
    iw, ih = min(ax2, bx2) - max(a[0], b[0]), min(ay2, by2) - max(a[1], b[1])
    if iw <= 0 or ih <= 0:
        return 0.0
    inter = iw * ih
    return inter / (a[2] * a[3] + b[2] * b[3] - inter)


def state_accuracy(pred, exist, boxes):
    """Return (SA, SA_pen) for one sequence."""
    if len(pred) != len(exist) or len(boxes) != len(exist):
        raise ValueError('Prediction and annotation lengths must match exactly')
    m, empty_on_target, n_target = [], 0, 0
    for t, p in enumerate(pred[: len(exist)]):
        if not exist[t]:
            m.append(1.0 if len(p) < 4 else 0.0)
        else:
            n_target += 1
            if len(p) == 4:
                m.append(iou_xywh(p, boxes[t]))
            else:
                m.append(0.0)
                empty_on_target += 1
    sa = float(np.mean(m)) if m else float("nan")
    pen = 0.2 * (empty_on_target / n_target) ** 0.3 if n_target else 0.0
    return sa, sa - pen


def match_dets(dets, exist, boxes, thr=0.5):
    """Scores and TP flags of all detections in one sequence, plus the number of targets."""
    if len(dets) != len(exist) or len(boxes) != len(exist):
        raise ValueError('Detection and annotation lengths must match exactly')
    scores, tps = [], []
    for t, frame in enumerate(dets[: len(exist)]):
        matched = False
        for d in sorted(frame, key=lambda x: -x[4]):
            ok = bool(exist[t]) and not matched and iou_xywh(d[:4], boxes[t]) >= thr
            matched |= ok
            scores.append(d[4])
            tps.append(ok)
    return np.array(scores), np.array(tps, bool), int(np.sum(exist))


def average_precision(scores, tps, n_gt):
    if n_gt == 0 or len(scores) == 0:
        return float("nan")
    o = np.argsort(-scores)
    tp = np.cumsum(tps[o])
    fp = np.cumsum(~tps[o])
    rec = tp / n_gt
    prec = tp / np.maximum(tp + fp, 1e-12)
    mrec = np.r_[0, rec, 1]
    mpre = np.r_[1, prec, 0]
    mpre = np.maximum.accumulate(mpre[::-1])[::-1]
    i = np.where(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1]))


def frames_to_converge(errs, tol=2.0):
    """First frame after which the prior error stays below tol (NaN if never)."""
    e = np.asarray(errs, float)
    bad = np.where(~(e < tol))[0]
    if len(bad) == 0:
        return 0.0
    return float(bad[-1] + 1) if bad[-1] + 1 < len(e) else float("nan")


def load_run(dirs, labels, require_dets=False):
    """Per-sequence metrics for one model, averaged over its seed directories."""
    per = defaultdict(lambda: defaultdict(list))
    for d in dirs:
        d = Path(d)
        if not d.is_dir() or not labels:
            raise ValueError('Run directory and nonempty labels are required')
        cfg = d / 'run_config.json'
        if cfg.exists() or cfg.with_name(cfg.name + '.gz').exists():
            meta = read_json(cfg)
            if meta.get('output_coordinates', 'RGB_pixels' if meta.get('input') == 'rgb' else 'IR_pixels') != 'IR_pixels':
                raise ValueError('The IR-label scorer requires IR-pixel predictions')
        curve = defaultdict(lambda: ([], [], []))
        cp = d / "prior_curve.csv"
        if cp.exists():
            with open(cp) as f:
                for r in csv.DictReader(f):
                    curve[r["seq"]][0].append(float(r["prior_error_px"]))
                    curve[r["seq"]][1].append(float(r.get("align_error_px", "nan")))
                    curve[r["seq"]][2].append(float(r.get("prior_target_error_px") or "nan"))
        for seq, (exist, boxes) in labels.items():
            pf, df = d / f"{seq}_IR.txt", d / f"{seq}_dets.json"
            pred = read_json(pf)["res"]
            validate_frames(pred, len(exist))
            sa, sa_pen = state_accuracy(pred, exist, boxes)
            per[seq]["SA"].append(sa)
            per[seq]["SA_pen"].append(sa_pen)
            if df.exists() or df.with_name(df.name + '.gz').exists() or require_dets:
                dets = read_json(df)
                validate_frames(dets, len(exist), detections=True)
                per[seq]["_dets"].append(match_dets(dets, exist, boxes))
            if seq in curve:
                pe, ae, te = curve[seq]
                per[seq]["align_px"].append(float(np.nanmedian(ae)) if np.any(np.isfinite(ae)) else np.nan)
                per[seq]["prior_px"].append(float(np.nanmedian(pe)) if np.any(np.isfinite(pe)) else np.nan)
                per[seq]["conv_frm"].append(frames_to_converge(pe))
                te_ok = [x for x in te if np.isfinite(x)]
                if te_ok:
                    per[seq]["prior_tgt_px"].append(float(np.median(te_ok)))
                    per[seq]["conv_tgt_frm"].append(frames_to_converge(te_ok, tol=5.0))
    return per


def summarise(per, seqs):
    out = {}
    for k in ("SA", "SA_pen", "align_px", "prior_px", "conv_frm", "prior_tgt_px", "conv_tgt_frm"):
        v = [np.nanmean(per[s][k]) for s in seqs if per[s].get(k)]
        if v:
            out[k] = float(np.nanmean(v)) if k in ("SA", "SA_pen") else float(np.nanmedian(v))
    out["AP50"] = pooled_ap(per, seqs)
    # spread between training seeds (the bootstrap only covers which videos were sampled)
    n_seeds = max((len(per[s]["SA"]) for s in seqs if per[s].get("SA")), default=0)
    if n_seeds > 1:
        seed_sa = [np.mean([per[s]["SA"][k] for s in seqs if len(per[s].get("SA", [])) > k]) for k in range(n_seeds)]
        out["n_seeds"], out["SA_seed_sd"] = n_seeds, float(np.std(seed_sa, ddof=1))
    else:
        out["n_seeds"] = n_seeds
    return out


def pooled_ap(per, seqs):
    """AP50 pooled over sequences, averaged over seeds."""
    n_seeds = max((len(per[s]["_dets"]) for s in seqs if per[s].get("_dets")), default=0)
    aps = []
    for k in range(n_seeds):
        sc, tp, ng = [], [], 0
        for s in seqs:
            if len(per[s].get("_dets", [])) > k:
                a, b, n = per[s]["_dets"][k]
                sc.append(a), tp.append(b)
                ng += n
        if sc:
            aps.append(average_precision(np.concatenate(sc), np.concatenate(tp), ng))
    return float(np.nanmean(aps)) if aps else float("nan")


def bootstrap(per_a, per_b, seqs, n_boot, rng):
    """Paired bootstrap over sequences of metric(a) - metric(b)."""
    seqs = [s for s in seqs if per_a[s].get("SA") and per_b[s].get("SA")]
    res = defaultdict(list)
    for _ in range(n_boot):
        draw = [seqs[i] for i in rng.integers(0, len(seqs), len(seqs))]
        for k in ("SA", "SA_pen"):
            res[k].append(np.mean([np.nanmean(per_a[s][k]) - np.nanmean(per_b[s][k]) for s in draw]))
        res["AP50"].append(pooled_ap(per_a, draw) - pooled_ap(per_b, draw))
    out = {}
    for k, v in res.items():
        v = np.asarray(v)
        v = v[np.isfinite(v)]
        if len(v):
            out[k] = {"mean": float(v.mean()), "ci95": [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))],
                      "share_le_0": float(np.mean(v <= 0)), "n_seq": len(seqs)}
    return out


def write_per_sequence(runs, seqs, output):
    """Write seed-mean SA for each video; unavailable scores remain blank."""
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["sequence", *(f"{name}_SA" for name in runs)])
        for seq in seqs:
            row = [seq]
            for per in runs.values():
                values = np.asarray(per.get(seq, {}).get("SA", []), dtype=float)
                values = values[np.isfinite(values)]
                row.append(float(values.mean()) if len(values) else "")
            writer.writerow(row)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--box-format", default="xywh", choices=["xywh", "xyxy"])
    ap.add_argument("--run", action="append", required=True, help="NAME=dir[,dir...] (one dir per seed)")
    ap.add_argument("--baseline", help="NAME of the run every other run is compared with")
    ap.add_argument("--boot", type=int, default=10000)
    ap.add_argument("--out", default="scores.json")
    ap.add_argument("--per-seq", type=Path, help="optional CSV of per-video SA (averaged across seeds)")
    ap.add_argument('--require-dets', action='store_true', help='require complete candidate files for AP')
    ap.add_argument('--bootstrap-mode', choices=['per_comparison', 'legacy_shared'], default='per_comparison',
                    help='per_comparison restarts seed 0 for order-invariant draws; legacy_shared reproduces old reports')
    args = ap.parse_args(argv)
    if args.boot < 1: raise ValueError('Bootstrap count must be positive')

    labels = {}
    for seq in list_sequences(Path(args.root) / args.split, need=("ir_label",)):
        labels[seq.name] = read_labels(find_files(seq)["ir_label"], args.box_format)
    seqs = sorted(labels)
    if not seqs: raise ValueError('No evaluation sequences found')
    runs = {}
    for spec in args.run:
        name, dirs = spec.split("=", 1)
        if name in runs: raise ValueError(f'Duplicate run: {name}')
        runs[name] = load_run(dirs.split(","), labels, args.require_dets)
    if args.baseline and args.baseline not in runs: raise ValueError('Baseline run not supplied')
    if args.per_seq is not None:
        write_per_sequence(runs, seqs, args.per_seq)
    report = {"split": args.split, "n_sequences": len(seqs), "runs": {}, "vs_baseline": {}}
    report.update(bootstrap_mode=args.bootstrap_mode, bootstrap_seed=0, bootstrap_resamples=args.boot)
    print(f"{'run':<14}{'SA':>8}{'SA_pen':>8}{'AP50':>8}{'align_px':>10}{'prior_px':>10}{'prior_tgt':>10}{'conv_frm':>10}")
    for name, per in runs.items():
        s = summarise(per, seqs)
        report["runs"][name] = s
        print(f"{name:<14}" + "".join(f"{s.get(k, float('nan')):>{w}.3f}" for k, w in
                                      (("SA", 8), ("SA_pen", 8), ("AP50", 8), ("align_px", 10), ("prior_px", 10),
                                       ("prior_tgt_px", 10), ("conv_frm", 10))))
    if args.baseline:
        rng = np.random.default_rng(0)
        for name, per in runs.items():
            if name == args.baseline:
                continue
            comparison_rng = np.random.default_rng(0) if args.bootstrap_mode == 'per_comparison' else rng
            b = bootstrap(per, runs[args.baseline], seqs, args.boot, comparison_rng)
            report["vs_baseline"][f"{name}-{args.baseline}"] = b
            for k, v in b.items():
                print(f"{name}-{args.baseline} {k}: {v['mean']:+.4f}  95% CI [{v['ci95'][0]:+.4f}, {v['ci95'][1]:+.4f}]"
                      f"  P(diff<=0)={v['share_le_0']:.3f}  n_seq={v['n_seq']}")
    Path(args.out).write_text(json.dumps(report, indent=1))
    print("written:", args.out)


if __name__ == "__main__":
    main()
