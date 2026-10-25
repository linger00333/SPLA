#!/usr/bin/env python3
"""Break down the frames a decision-fusion run replaced, against the IR labels.

Run only AFTER the fusion protocol is registered: this reads evaluation labels.
For every frame that decision_fusion.py replaced, it records whether the IR
label is positive and, if so, the IoU of the IR detector's own output and of
the fused output. SA counts a replaced frame on an IR-negative frame as a loss
of 1, so the "negative" row is the false-positive cost.

Usage:
  python kit/tools/fusion_breakdown.py --root data/raw --split val \
     --ir-run results/A0_s0 --fused results/fusion/P_train_mean --out reports/fusion_P_breakdown.json
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from spla.antiuav_io import find_files, read_labels  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from decision_fusion import _read_json, iou_xywh  # noqa: E402


def frame_score(pred, exist, box):
    """The per-frame SA term of score_runs.state_accuracy."""
    if not exist:
        return 1.0 if len(pred) < 4 else 0.0
    return iou_xywh(pred, box) if len(pred) == 4 else 0.0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True, type=Path)
    ap.add_argument("--split", default="val")
    ap.add_argument("--box-format", default="xywh", choices=["xywh", "xyxy"])
    ap.add_argument("--ir-run", required=True, type=Path)
    ap.add_argument("--fused", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args(argv)
    if args.split == "test":
        raise SystemExit("refusing split=test (protocol)")
    replaced = defaultdict(list)
    with open(args.fused / "fusion_log.csv") as f:
        for r in csv.DictReader(f):
            if r["action"] == "replace":
                replaced[r["seq"]].append(int(r["frame"]))
    rows = {"ir_positive": defaultdict(float), "ir_negative": defaultdict(float)}
    per_video = {}
    for seq, frames in sorted(replaced.items()):
        exist, boxes = read_labels(find_files(args.root / args.split / seq)["ir_label"], args.box_format)
        base = _read_json(args.ir_run / f"{seq}_IR.txt")["res"]
        fused = _read_json(args.fused / f"{seq}_IR.txt")["res"]
        if not (len(exist) == len(base) == len(fused)):
            raise ValueError(f"{seq}: label and prediction lengths differ")
        delta = 0.0
        for t in frames:
            key = "ir_positive" if exist[t] else "ir_negative"
            b, u = frame_score(base[t], exist[t], boxes[t]), frame_score(fused[t], exist[t], boxes[t])
            rows[key]["frames"] += 1
            rows[key]["sum_score_ir"] += b
            rows[key]["sum_score_fused"] += u
            if exist[t]:
                rows[key]["fused_iou_ge_0.5"] += float(u >= 0.5)
            delta += u - b
        per_video[seq] = {"replaced": len(frames), "frames": len(exist),
                          "sa_change_in_video": delta / len(exist)}
    out = {"split": args.split, "fused_run": str(args.fused),
           "replaced_frames": {k: dict(v) for k, v in rows.items()}, "per_video": per_video,
           "note": "per-frame SA terms; sa_change_in_video is the change of that video's SA"}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=1))
    print(json.dumps(out["replaced_frames"], indent=1))
    print("written:", args.out)


if __name__ == "__main__":
    main()
