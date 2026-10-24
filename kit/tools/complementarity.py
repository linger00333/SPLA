#!/usr/bin/env python3
"""Measure complementarity of two fixed detectors in their own label coordinates.

Compares an IR detector (A0, boxes in IR pixels) with an RGB-only detector
(run_sequences.py --input rgb, boxes in RGB pixels), each against its OWN
modality's labels, on frames where the drone is labelled in both. A frame is
a success for a modality when its top box has IoU >= --iou with that
modality's label. Then:

  rescue share  = frames RGB succeeds but IR fails / frames IR fails
  oracle gain   = success rate if each frame took the better modality,
                  minus IR's success rate

The oracle selects the better of these two fixed outputs on each frame.
It is not an upper bound on learned fusion or IR-coordinate detection gains.
Intervals: paired bootstrap over videos.

Usage:
  python tools/complementarity.py --root /data/Anti-UAV-RGBT --split val \
      --ir-run results/val/A0_s0 --rgb-run results/val/A0rgb_s0 --out reports/complementarity_val.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from spla.antiuav_io import find_files, read_labels, KNOWN_LABEL_ISSUES  # noqa: E402

try:
    from score_runs import iou_xywh
except ImportError:  # run from elsewhere
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from score_runs import iou_xywh


def frame_success(pred, exist, boxes, thr):
    """Per-frame success (bool) of a run's top boxes against labels; NaN-free arrays."""
    if len(pred) != len(exist) or len(boxes) != len(exist):
        raise ValueError('Prediction and annotation lengths must match exactly')
    n = len(exist)
    s = np.zeros(n, bool)
    for t in range(n):
        if exist[t] and len(pred[t]) == 4:
            s[t] = iou_xywh(pred[t], boxes[t]) >= thr
    return s


def counts_for(s_ir, s_rgb, both_exist):
    if len(s_ir) != len(s_rgb) or len(s_ir) != len(both_exist):
        raise ValueError('Modality and mask lengths must match exactly')
    m = both_exist
    a, b = s_ir[: len(m)][m], s_rgb[: len(m)][m]
    return {"n": int(m.sum()), "both": int(np.sum(a & b)), "ir_only": int(np.sum(a & ~b)),
            "rgb_only": int(np.sum(~a & b)), "neither": int(np.sum(~a & ~b))}


def summarise(c):
    n = c["n"]
    ir_fail = c["rgb_only"] + c["neither"]
    return {"frames": c["n"], "ir_success": (c["both"] + c["ir_only"]) / n if n else None,
            "rgb_success": (c["both"] + c["rgb_only"]) / n if n else None,
            "oracle_success": (c["both"] + c["ir_only"] + c["rgb_only"]) / n if n else None,
            "oracle_gain": c["rgb_only"] / n if n else None,
            "rescue_share": c["rgb_only"] / ir_fail if ir_fail else None,
            "ir_failures": ir_fail}


def add(c1, c2):
    return {k: c1[k] + c2[k] for k in c1}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True)
    ap.add_argument("--split", default="val")
    ap.add_argument("--box-format", default="xywh", choices=["xywh", "xyxy"])
    ap.add_argument("--ir-run", required=True, help="run_sequences output of the IR model (boxes in IR px)")
    ap.add_argument("--rgb-run", required=True, help="run_sequences --input rgb output (boxes in RGB px)")
    ap.add_argument("--iou", type=float, default=0.5)
    ap.add_argument("--boot", type=int, default=10000)
    ap.add_argument("--out", default="complementarity.json")
    args = ap.parse_args(argv)
    if not 0 < args.iou <= 1 or args.boot < 1:
        raise ValueError('Require 0 < IoU <= 1 and positive bootstrap count')

    per = {}
    seqs = sorted(p for p in (Path(args.root) / args.split).iterdir() if p.is_dir())
    for directory, expected in ((args.ir_run, 'IR_pixels'), (args.rgb_run, 'RGB_pixels')):
        cfg = json.loads((Path(directory) / 'run_config.json').read_text())
        actual = cfg.get('output_coordinates', 'RGB_pixels' if cfg.get('input') == 'rgb' else 'IR_pixels')
        if actual != expected or cfg.get('split') != args.split:
            raise ValueError(f'Incorrect coordinates or split in {directory}')
    for seq in seqs:
        pi, pr = Path(args.ir_run) / f"{seq.name}_IR.txt", Path(args.rgb_run) / f"{seq.name}_IR.txt"
        if not (pi.exists() and pr.exists()):
            raise FileNotFoundError(f'Missing prediction for {seq.name}')
        fs = find_files(seq)
        for role in ('ir_label', 'rgb_label'):
            obj = json.loads(fs[role].read_text())
            if len(obj['exist']) != len(obj['gt_rect']):
                raise ValueError(f'Annotation lengths differ: {seq.name}/{role}')
        ei, bi = read_labels(fs["ir_label"], args.box_format)
        er, br = read_labels(fs["rgb_label"], args.box_format)
        if len(ei) != len(er):
            raise ValueError(f'Modality annotation lengths differ: {seq.name}')
        both = (ei == 1) & (er == 1)
        s_ir = frame_success(json.loads(pi.read_text())["res"], ei, bi, args.iou)
        s_rgb = frame_success(json.loads(pr.read_text())["res"], er, br, args.iou)
        per[seq.name] = counts_for(s_ir, s_rgb, both)
    if not per:
        raise SystemExit("no video has predictions in both runs")
    names = sorted(per)
    total = {"n": 0, "both": 0, "ir_only": 0, "rgb_only": 0, "neither": 0}
    for k in names:
        total = add(total, per[k])
    report = {"split": args.split, "iou": args.iou, "videos": len(names), "pooled": summarise(total),
              'scope': 'Fixed-detector frame selection, each scored in its own modality coordinates; not a bound on learned fusion',
              'known_label_issues': {k: KNOWN_LABEL_ISSUES[k] for k in names if k in KNOWN_LABEL_ISSUES},
              'coverage': {'all_sequences_required': True, 'lengths_exact': True,
                           'total_frames': sum(len(json.loads((Path(args.ir_run) / f'{k}_IR.txt').read_text())['res']) for k in names)}}
    rng = np.random.default_rng(0)
    boot = {"oracle_gain": [], "rescue_share": []}
    for _ in range(args.boot):
        c = {"n": 0, "both": 0, "ir_only": 0, "rgb_only": 0, "neither": 0}
        for i in rng.integers(0, len(names), len(names)):
            c = add(c, per[names[i]])
        s = summarise(c)
        for k in boot:
            boot[k].append(s[k] if s[k] is not None else float('nan'))
    for k, v in boot.items():
        v = np.asarray(v, float)
        v = v[np.isfinite(v)]
        if len(v):
            report[f"{k}_ci95"] = [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]
        else:
            report[f"{k}_ci95"] = None
        report[f'{k}_bootstrap_valid'] = int(len(v))
    clean = dict.fromkeys(total, 0)
    for k in names:
        if k not in KNOWN_LABEL_ISSUES:
            clean = add(clean, per[k])
    report['excluding_known_label_issues'] = summarise(clean)
    report['bootstrap'] = {'unit': 'sequence', 'seed': 0, 'replicates': args.boot,
                           'aggregation': 'pooled counts after sampling sequences with replacement'}
    report["per_video"] = {k: dict(per[k], **summarise(per[k])) for k in names}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=1, allow_nan=False))
    p = report["pooled"]
    print(f"videos {len(names)}, frames with both labels {p['frames']}")
    print('Pooled metrics:', json.dumps(p, allow_nan=False))
    print("written:", args.out)


if __name__ == "__main__":
    main()
