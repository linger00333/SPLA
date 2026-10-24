#!/usr/bin/env python3
"""Check an Anti-UAV RGB-T download before anything else runs on it.

Answers, for your copy of the data:
  * which split folders and how many sequences each (official: 160 / 67 / 91);
  * which file names it uses (infrared.* / visible.* or IR.* / RGB.*);
  * frame sizes and frame counts per modality, and whether the label length
    matches the video length (a mismatch breaks frame pairing);
  * the gt_rect layout: x, y, w, h (official eval code) or x1, y1, x2, y2
    (v1 paper text). Pass the answer to the other tools as --box-format;
  * how many frames have both an IR and an RGB box (the alignment pairs);
  * the share of frames whose boxes are linearly interpolated (coarse labels).

Usage:
  python tools/inspect_dataset.py --root /data/Anti-UAV-RGBT [--max-seq 0] [--out report.json]
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from spla.antiuav_io import CANDIDATES, find_files  # noqa: E402


def video_info(path):
    try:
        import cv2
    except ImportError:
        return None
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return None
    info = dict(w=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), h=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                n=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), fps=float(cap.get(cv2.CAP_PROP_FPS)))
    cap.release()
    return info


def raw_rects(path):
    d = json.loads(Path(path).read_text())
    ex = np.asarray(d.get("exist", []), int)
    rects = [r for e, r in zip(ex, d.get("gt_rect", [])) if e and isinstance(r, (list, tuple)) and len(r) == 4]
    return d, ex, np.asarray(rects, float).reshape(-1, 4)


def interpolated_share(rects):
    if len(rects) < 5:
        return float("nan")
    c = rects[:, :2]
    d2 = np.abs(c[2:] - 2 * c[1:-1] + c[:-2]).max(1)
    return float(np.mean(d2 < 1e-3))


def draw_examples(root, k, box_format, out_dir):
    """IR frame with its box | RGB frame (stretched to IR size) with its box (green) and the
    IR box at the same normalised position (red). The red-green gap is the misalignment."""
    import cv2
    from spla.antiuav_io import list_sequences, read_labels
    split = next((root / n for n in ("train", "Train", "training") if (root / n).exists()), None)
    if split is None:
        print("draw: no train folder")
        return
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    for seq in list_sequences(split, need=("ir_label", "rgb_label", "ir_video", "rgb_video"))[:k]:
        fs = find_files(seq)
        ei, bi = read_labels(fs["ir_label"], box_format)
        er, br = read_labels(fs["rgb_label"], box_format)
        both = np.where((ei[: len(er)] == 1) & (er[: len(ei)] == 1))[0]
        if not len(both):
            continue
        t = int(both[len(both) // 2])
        frames = []
        for vid in (fs["ir_video"], fs["rgb_video"]):
            cap = cv2.VideoCapture(str(vid))
            cap.set(cv2.CAP_PROP_POS_FRAMES, t)
            ok, f = cap.read()
            cap.release()
            frames.append(f if ok else None)
        if frames[0] is None or frames[1] is None:
            continue
        hi, wi = frames[0].shape[:2]
        hv, wv = frames[1].shape[:2]
        ir, rgb = frames[0].copy(), cv2.resize(frames[1], (wi, hi))
        def rect(img, b, sx, sy, col):
            x, y, w, h = b
            cv2.rectangle(img, (int(x * sx), int(y * sy)), (int((x + w) * sx), int((y + h) * sy)), col, 2)
        rect(ir, bi[t], 1, 1, (0, 255, 0))
        rect(rgb, br[t], wi / wv, hi / hv, (0, 255, 0))
        rect(rgb, bi[t], 1, 1, (0, 0, 255))
        cv2.imwrite(f"{out_dir}/{seq.name}_{t:06d}.jpg", np.hstack([ir, rgb]))
    print("pictures:", out_dir)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True)
    ap.add_argument("--max-seq", type=int, default=0, help="inspect at most this many per split (0 = all)")
    ap.add_argument("--out", default="dataset_report.json")
    ap.add_argument("--draw", type=int, default=0,
                    help="save side-by-side IR | RGB pictures with boxes for this many train sequences")
    ap.add_argument("--box-format", default="xywh", choices=["xywh", "xyxy"], help="layout used for --draw")
    args = ap.parse_args()

    root = Path(args.root)
    report = {"root": str(root), "splits": {}}
    names = Counter()
    third_lt_first, n_rects = 0, 0
    for split in sorted(p for p in root.iterdir() if p.is_dir()):
        seqs = sorted(p for p in split.iterdir() if p.is_dir())
        if not seqs:
            continue
        rows, problems = [], []
        for seq in seqs[: args.max_seq or None]:
            fs = find_files(seq)
            for role, p in fs.items():
                names[(role, p.name if p else None)] += 1
            row = {"seq": seq.name}
            missing = [r for r, p in fs.items() if p is None]
            if missing:
                problems.append(f"{seq.name}: missing {missing}; files: {sorted(x.name for x in seq.iterdir())[:8]}")
                rows.append(row)
                continue
            for mod in ("ir", "rgb"):
                d, ex, rects = raw_rects(fs[f"{mod}_label"])
                vi = video_info(fs[f"{mod}_video"])
                row[f"{mod}_labels"] = len(ex)
                row[f"{mod}_exist"] = int(ex.sum())
                row[f"{mod}_interp"] = interpolated_share(rects)
                if vi:
                    row.update({f"{mod}_{k}": v for k, v in vi.items()})
                    if vi["n"] and abs(vi["n"] - len(ex)) > 1:
                        problems.append(f"{seq.name}: {mod} video has {vi['n']} frames, labels {len(ex)}")
                third_lt_first += int(np.sum((rects[:, 2] < rects[:, 0]) | (rects[:, 3] < rects[:, 1])))
                n_rects += len(rects)
            _, ei, _ = raw_rects(fs["ir_label"])
            _, er, _ = raw_rects(fs["rgb_label"])
            m = min(len(ei), len(er))
            row["pairs"] = int(np.sum((ei[:m] == 1) & (er[:m] == 1)))
            rows.append(row)
        def med(k):
            v = [r[k] for r in rows if k in r and np.isfinite(r[k])]
            return float(np.median(v)) if v else None
        sizes = Counter((r.get("ir_w"), r.get("ir_h"), r.get("rgb_w"), r.get("rgb_h")) for r in rows if "ir_w" in r)
        report["splits"][split.name] = {
            "n_sequences": len(seqs), "inspected": len(rows),
            "frame_sizes_ir_wh_rgb_wh": {str(k): v for k, v in sizes.most_common(5)},
            "median_fps_ir": med("ir_fps"), "median_frames": med("ir_labels"),
            "median_pairs_per_seq": med("pairs"), "total_pairs": int(sum(r.get("pairs", 0) for r in rows)),
            "median_interpolated_share_ir": med("ir_interp"), "median_interpolated_share_rgb": med("rgb_interp"),
            "problems": problems[:50], "n_problems": len(problems),
        }
    share = third_lt_first / max(n_rects, 1)
    report["file_names"] = {f"{r}:{n}": c for (r, n), c in names.most_common()}
    report["box_format"] = {
        "share_third_lt_first": share,
        "verdict": "xywh" if share > 0.5 else ("xyxy" if share < 0.01 else "unclear: draw a few boxes and look"),
        "rule": "UAV boxes are small, so in x,y,w,h the 3rd value (w) is almost always smaller than x",
    }
    Path(args.out).write_text(json.dumps(report, indent=1))
    if args.draw:
        draw_examples(root, args.draw, args.box_format, Path(args.out).with_suffix("").as_posix() + "_frames")
    for s, v in report["splits"].items():
        print(f"{s:>12}: {v['n_sequences']} sequences, sizes {v['frame_sizes_ir_wh_rgb_wh']}, "
              f"pairs {v['total_pairs']}, interp IR {v['median_interpolated_share_ir']}, problems {v['n_problems']}")
    print("file names:", report["file_names"])
    print("box format:", report["box_format"]["verdict"], f"(share {share:.3f})")
    print("expected names:", CANDIDATES)
    print("report:", args.out)


if __name__ == "__main__":
    main()
