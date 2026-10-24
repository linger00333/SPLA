#!/usr/bin/env python3
"""Build an RGB-only detection set from an existing conversion (no frame extraction).

It reuses the stretched RGB frames that convert_antiuav.py already wrote and
labels them with the RGB boxes (the converter's labels are IR boxes). Train a
plain YOLO11 on it to measure how much the RGB stream can see on its own:

  OUT/rgbonly/images/<split>/*  hard links to OUT/images/visible/<split>/* (no copies)
  OUT/rgbonly/labels/<split>/<stem>.txt   YOLO labels from the RGB box at that frame
  OUT/antiuav_rgbonly.yaml

The stretched frame keeps normalised RGB coordinates, so the RGB box
normalised by the original RGB size is also correct in the stretched image.

Usage:
  python tools/make_rgb_only.py --root /data/Anti-UAV-RGBT --out /data/antiuav_yolo
  python tools/train_spla.py --stage 0 --data /data/antiuav_yolo/antiuav_rgbonly.yaml --seed 0 --name A0rgb_s0
"""
from __future__ import annotations

import argparse
import json
import os
import hashlib
import shutil
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from spla.antiuav_io import find_files, read_labels  # noqa: E402


def video_wh(path, default):
    try:
        import cv2
        cap = cv2.VideoCapture(str(path))
        wh = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        cap.release()
        if min(wh) > 0:
            return wh
    except Exception:
        pass
    if default is not None and min(default) > 0:
        return tuple(default)
    raise ValueError(f'Cannot establish original RGB dimensions: {path}')


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True, help="original Anti-UAV RGB-T folder")
    ap.add_argument("--out", required=True, help="folder written by convert_antiuav.py")
    ap.add_argument('--source', help='read-only existing conversion; defaults to --out')
    ap.add_argument("--splits", nargs="+", choices=['train', 'val'], default=["train", "val"])
    ap.add_argument("--box-format", default="xywh", choices=["xywh", "xyxy"])
    ap.add_argument("--rgb-size", nargs=2, type=int, default=None, help="explicit dimensions for synthetic/offline inputs only")
    args = ap.parse_args(argv)

    out = Path(args.out).resolve()
    source = Path(args.source).resolve() if args.source else out
    base = out / "rgbonly"
    if base.exists() or base.is_symlink():
        raise FileExistsError(f'Refusing to modify existing dataset: {base}')
    counts = {}
    for split in args.splits:
        meta_p = source / f"spla_meta_{split}.json"
        src = source / "images" / "visible" / split
        if not meta_p.exists() or not src.exists():
            raise FileNotFoundError(f'Missing metadata or images for {split}')
        meta = json.loads(meta_p.read_text())
        expected = {stem + '.jpg' for stem in meta}
        actual = {p.name for p in src.iterdir() if p.is_file()}
        if expected != actual:
            raise ValueError(f'Image/metadata mismatch: {split}; missing={len(expected-actual)}, extra={len(actual-expected)}')
        # A real folder of per-file links, NOT a folder symlink: Ultralytics resolves the split path,
        # and a resolved folder symlink would make it read labels/visible/<split> (the converter's
        # labels) instead of the RGB labels written here.
        img_dir = base / "images" / split
        if img_dir.is_symlink():
            raise SystemExit(f"{img_dir} is a folder symlink from an older run; delete it and rerun")
        img_dir.mkdir(parents=True, exist_ok=True)
        links = copies = 0
        for f in sorted(src.iterdir()):
            dst = img_dir / f.name
            if dst.exists() or dst.is_symlink():
                continue
            try:
                os.link(f, dst)               # hard link: no copy, and nothing for a resolve() to follow
                links += 1
            except OSError:
                shutil.copy2(f, dst)  # preserve path-based label lookup if hard links are unavailable
                copies += 1
        lab_dir = base / "labels" / split
        lab_dir.mkdir(parents=True, exist_ok=True)
        cache = {}
        n_pos = 0
        for stem, m in meta.items():
            seq = m["seq"]
            if seq not in cache:
                fs = find_files(Path(args.root) / split / seq)
                raw = json.loads(fs['rgb_label'].read_text())
                if len(raw['exist']) != len(raw['gt_rect']):
                    raise ValueError(f'RGB annotation lengths differ: {split}/{seq}')
                e, b = read_labels(fs["rgb_label"], args.box_format)
                wh = video_wh(fs['rgb_video'], args.rgb_size)
                cache[seq] = (e, b, wh)
            e, b, (wv, hv) = cache[seq]
            t = m["frame"]
            if not isinstance(t, int) or t < 0 or t >= len(e):
                raise ValueError(f'Frame outside RGB annotations: {stem}')
            line = ""
            if t < len(e) and e[t] and np.all(np.isfinite(b[t])):
                x, y, w, h = b[t]
                cx, cy = np.clip([(x + w / 2) / wv, (y + h / 2) / hv], 0, 1)
                line = f"0 {cx:.6f} {cy:.6f} {min(w / wv, 1):.6f} {min(h / hv, 1):.6f}\n"
                n_pos += 1
            (lab_dir / f"{stem}.txt").write_text(line)
        counts[split] = {"images": len(meta), "with_rgb_target": n_pos,
                         'hardlinks': links, 'copies': copies, 'file_symlinks': 0,
                         'sampling_metadata_sha256': hashlib.sha256(meta_p.read_bytes()).hexdigest(),
                         'source_rgb_dimensions': {k: list(v[2]) for k, v in cache.items()}}
        print(f"{split}: {len(meta)} images, {n_pos} with an RGB target")
    (out / "antiuav_rgbonly.yaml").write_text(
        f"path: {base.as_posix()}\n" + ''.join(f'{s}: images/{s}\n' for s in args.splits) + "nc: 1\nnames: ['uav']\n")
    (base / "summary.json").write_text(json.dumps(counts, indent=1))
    print("written:", out / "antiuav_rgbonly.yaml")


if __name__ == "__main__":
    main()
