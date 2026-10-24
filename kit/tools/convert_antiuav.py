#!/usr/bin/env python3
"""Convert Anti-UAV RGB-T video pairs into the YOLOv11-RGBT layout plus SPLA metadata.

Input (official Anti-UAV RGB-T release, 318 sequences):
  ROOT/<split>/<seq>/ IR and RGB videos + label JSONs (names detected by spla/antiuav_io.py)

Output:
  OUT/images/visible/<split>/<split>_<seq>_<frame>.jpg   RGB frame RESIZED TO THE IR SIZE
  OUT/images/infrared/<split>/<split>_<seq>_<frame>.jpg  IR frame (grayscale)
  OUT/labels/visible/<split>/<split>_<seq>_<frame>.txt   YOLO labels from the IR box
  OUT/spla_meta_<split>.json                     per-image SPLA metadata
  OUT/seq_theta_<split>.json                     per-sequence oracle transform + fit stats
  OUT/antiuav_rgbt.yaml                          data yaml for the fork

Why the RGB frame is resized to the IR size: the fork merges the two images
with cv2.merge, which needs equal sizes, and resizing each to the training
size separately would break when the aspect ratios differ (16:9 vs 5:4).
Stretching keeps NORMALISED coordinates unchanged, so a transform fitted on
normalised coordinates still applies. Note that a similarity in pixels
becomes an anisotropic scale in normalised coordinates when the aspect
ratios differ; this is why the default model here is scale_translation
(use the model the audit picks most often).

Boxes are the IR boxes: the fused feature map lives on the IR grid, so the
detector predicts IR-frame boxes, and the RGB box only supplies the
alignment target (its centre). Frames without an IR target are kept as
negatives (empty label) at a lower rate (--neg-stride).

The word "visible" must not appear anywhere else in OUT, because the fork
finds the IR file by replacing it in the path.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from spla.antiuav_io import find_files, list_sequences, read_labels  # noqa: E402
from spla.geometry import MODELS, fit_transform, identity_params, set_zoom_aspect  # noqa: E402




def centres(boxes, w, h):
    return np.stack([(boxes[:, 0] + boxes[:, 2] / 2) / w, (boxes[:, 1] + boxes[:, 3] / 2) / h], 1)


def frames(path: Path):
    import cv2
    cap = cv2.VideoCapture(str(path))
    while True:
        ok, f = cap.read()
        if not ok:
            break
        yield f
    cap.release()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    ap.add_argument("--stride", type=int, default=5, help="keep every k-th frame with a target")
    ap.add_argument("--neg-stride", type=int, default=50, help="keep every k-th frame without a target")
    ap.add_argument("--model", default="zoom", choices=list(MODELS),
                    help="zoom = one pixel scale + shift (3 params): the camera physics; see spla/geometry.py")
    ap.add_argument("--theta-only", action="store_true",
                    help="only refit the per-video transforms and update theta_seq in existing spla_meta_*.json "
                         "(no frame extraction; use after changing --model)")
    ap.add_argument("--min-pairs", type=int, default=20, help="pairs needed to fit a sequence transform")
    ap.add_argument("--box-format", default="xywh", choices=["xywh", "xyxy"],
                    help="gt_rect layout; tools/inspect_dataset.py tells you which")
    args = ap.parse_args()

    import cv2
    out = Path(args.out)
    assert "visible" not in str(out.resolve()), "OUT must not contain the word 'visible'"
    for split in args.splits:
        sdir = Path(args.root) / split
        if not sdir.exists():
            print(f"skip {split}: {sdir} missing")
            continue
        if not args.theta_only:
            for sub in ("images/visible", "images/infrared", "labels/visible"):
                (out / sub / split).mkdir(parents=True, exist_ok=True)
        meta, seq_info = {}, {}
        old_meta = json.loads((out / f"spla_meta_{split}.json").read_text()) if args.theta_only else {}
        for seq in list_sequences(sdir, need=("ir_label", "rgb_label", "ir_video", "rgb_video")):
            fs = find_files(seq)
            ir_e, ir_b = read_labels(fs["ir_label"], args.box_format)
            rgb_e, rgb_b = read_labels(fs["rgb_label"], args.box_format)
            cap_i, cap_v = cv2.VideoCapture(str(fs["ir_video"])), cv2.VideoCapture(str(fs["rgb_video"]))
            wi, hi = int(cap_i.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap_i.get(cv2.CAP_PROP_FRAME_HEIGHT))
            wv, hv = int(cap_v.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap_v.get(cv2.CAP_PROP_FRAME_HEIGHT))
            cap_i.release(), cap_v.release()
            n = min(len(ir_e), len(rgb_e))
            c_ir, c_rgb = centres(ir_b[:n], wi, hi), centres(rgb_b[:n], wv, hv)
            aspect = set_zoom_aspect((wi, hi), (wv, hv))
            both = (ir_e[:n] == 1) & (rgb_e[:n] == 1)
            # oracle per-sequence transform (used for training targets and as the
            # oracle upper bound at test time; NEVER fed to the online test run)
            if both.sum() >= args.min_pairs:
                fit = fit_transform(c_ir[both], c_rgb[both], args.model)
                params, rms, inl = fit.params, fit.rms, fit.inlier_fraction
            else:
                params, rms, inl = identity_params(args.model), float("nan"), float("nan")
            fit_theta = fit.theta if both.sum() >= args.min_pairs else np.array([[1.0, 0, 0], [0, 1.0, 0]])
            seq_info[seq.name] = {"model": args.model, "params": params.tolist(), "theta": fit_theta.tolist(),
                                  "n_pairs": int(both.sum()), "rms_norm": rms, "inlier_fraction": inl,
                                  "ir_wh": [wi, hi], "rgb_wh": [wv, hv], "zoom_aspect": aspect}
            if args.theta_only:
                for stem, m in old_meta.items():
                    if m["seq"] == seq.name:
                        meta[stem] = dict(m, theta_seq=fit_theta.tolist())
                print(f"{split}/{seq.name}: pairs={both.sum()} rms={rms:.4f} (theta only)")
                continue
            k_pos = k_neg = 0
            for t, (fi, fv) in enumerate(zip(frames(fs["ir_video"]), frames(fs["rgb_video"]))):
                if t >= n:
                    break
                has = ir_e[t] == 1
                if has:
                    k_pos += 1
                    if (k_pos - 1) % args.stride:
                        continue
                else:
                    k_neg += 1
                    if (k_neg - 1) % args.neg_stride:
                        continue
                stem = f"{split}_{seq.name}_{t:06d}"   # split prefix: train/val clips can share a video name
                fv = cv2.resize(fv, (wi, hi), interpolation=cv2.INTER_AREA)
                cv2.imwrite(str(out / "images/visible" / split / f"{stem}.jpg"), fv)
                cv2.imwrite(str(out / "images/infrared" / split / f"{stem}.jpg"), cv2.cvtColor(fi, cv2.COLOR_BGR2GRAY))
                lab = ""
                if has:
                    x, y, w, h = ir_b[t]
                    cx, cy = np.clip([(x + w / 2) / wi, (y + h / 2) / hi], 0, 1)
                    lab = f"0 {cx:.6f} {cy:.6f} {min(w / wi, 1):.6f} {min(h / hi, 1):.6f}\n"
                (out / "labels/visible" / split / f"{stem}.txt").write_text(lab)
                pair = bool(both[t])
                meta[stem] = {"seq": seq.name, "frame": t, "img_wh": [wi, hi],
                              "theta_seq": fit_theta.tolist(),
                              "ir_ctr": c_ir[t:t + 1].tolist() if pair else [],
                              "rgb_ctr": c_rgb[t:t + 1].tolist() if pair else []}
            print(f"{split}/{seq.name}: pairs={both.sum()} rms={rms:.4f} kept={sum(1 for k in meta if k.startswith(f'{split}_{seq.name}_'))}")
        (out / f"spla_meta_{split}.json").write_text(json.dumps(meta))
        (out / f"seq_theta_{split}.json").write_text(json.dumps(seq_info, indent=1))
    # one sidecar for the trainer and its validator: train + val only (stems carry
    # the split, so they never collide). Test-split transforms stay in their own
    # files, so no test label can reach training or model selection.
    merged = {}
    for split in [s for s in args.splits if s != "test"]:
        p = out / f"spla_meta_{split}.json"
        if p.exists():
            merged.update(json.loads(p.read_text()))
    (out / "spla_meta.json").write_text(json.dumps(merged))
    # A0 (IR-only baseline): the IR images with the same labels, read as 3-channel images
    lab_ir = out / "labels" / "infrared"
    if not lab_ir.exists():
        lab_ir.symlink_to(out / "labels" / "visible", target_is_directory=True)
    (out / "antiuav_ir.yaml").write_text(
        f"path: {out.resolve()}\ntrain: images/infrared/train\nval: images/infrared/val\n"
        f"test: images/infrared/test\nnc: 1\nnames: ['uav']\n")
    (out / "antiuav_rgbt.yaml").write_text(
        f"path: {out.resolve()}\ntrain: images/visible/train\nval: images/visible/val\n"
        f"test: images/visible/test\nnc: 1\nnames: ['uav']\n")
    print("done:", out)


if __name__ == "__main__":
    main()
