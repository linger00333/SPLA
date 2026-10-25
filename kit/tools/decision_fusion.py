#!/usr/bin/env python3
"""No-training decision-level fusion of a fixed IR detector and a fixed RGB detector.

Inputs are two completed run_sequences.py outputs on the same split:
  --ir-run   the IR-only detector (A0), boxes in IR pixels
  --rgb-run  the RGB-only detector (run_sequences.py --input rgb), boxes in ORIGINAL RGB pixels

Rule, per frame (nothing from any label is used unless --map seq):
  keep the IR top box, unless its confidence is below --ir-conf (0.25 = only
  empty IR frames) AND the RGB top box has confidence >= --rgb-conf. Then the
  RGB box is mapped into IR pixels and replaces the IR output. A mapped box
  whose centre falls outside the IR frame is discarded (the IR output is kept),
  otherwise it is clipped to the IR frame.

Mapping (--map), theta maps NORMALISED IR points to NORMALISED RGB points
(spla.geometry), so an RGB point maps back through theta^-1:
  train_mean  mean of the training-split per-video transforms; uses no validation label
  seq         each video's own transform from --seq-theta, fitted on that split's labels.
              This leaks evaluation labels and is a reference, not a method
  online      train_mean plus a causal centre correction: the median IR-minus-mapped
              centre offset of the last --online-k frames (within --window frames)
              where both detectors were confident (>= --pair-conf). Uses only past
              predictions; falls back to train_mean when no such frame exists

Box size (--size):
  mapped      the mapped RGB box itself
  ir_recent   the mapped centre with the width and height of the most recent
              confident IR box within --window frames (falls back to mapped)

Outputs (same layout as run_sequences.py, so score_runs.py scores it unchanged):
  OUT/<seq>_IR.txt      {"res": [[x, y, w, h] | [], ...]}, IR pixels
  OUT/<seq>_dets.json   IR candidates unchanged; on replaced frames the mapped
                        box is inserted first with the RGB confidence
  OUT/run_config.json   arguments, input hashes, output_coordinates = IR_pixels
  OUT/fusion_log.csv    one row per frame: confidences, action, mapping source
  OUT/fusion_summary.json  label-free counts and a transfer-fidelity check: on
                        frames where both detectors are confident, the IoU of
                        the box this rule WOULD output against the IR box

Accepts plain or .gz prediction files. Missing videos, unequal lengths, wrong
coordinate metadata and the test split (unless --allow-test) are errors.

Example:
  python kit/tools/decision_fusion.py --ir-run results/A0_s0 --rgb-run results/A0rgb_s0 \
     --train-theta data/converted/seq_theta_train.json --map train_mean --size mapped \
     --out results/fusion/P_train_mean
  python kit/tools/score_runs.py --root data/raw --split val --run A0=results/A0_s0 \
     --run P=results/fusion/P_train_mean --baseline A0 --boot 10000 --out reports/fusion_val.json
"""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
from collections import deque
from pathlib import Path

import numpy as np

SUFFIX_RES, SUFFIX_DETS = "_IR.txt", "_dets.json"


# --------------------------------------------------------------------------
# Input
# --------------------------------------------------------------------------
def _read_json(path: Path):
    if path.exists():
        return json.loads(path.read_text())
    gz = path.with_name(path.name + ".gz")
    if gz.exists():
        return json.loads(gzip.open(gz, "rt").read())
    raise FileNotFoundError(path)


def run_sequences(directory: Path) -> list[str]:
    names = set()
    for p in Path(directory).iterdir():
        n = p.name[:-3] if p.name.endswith(".gz") else p.name
        if n.endswith(SUFFIX_RES):
            names.add(n[: -len(SUFFIX_RES)])
    return sorted(names)


def run_config(directory: Path) -> dict:
    return _read_json(Path(directory) / "run_config.json")


def coordinates(cfg: dict) -> str:
    return cfg.get("output_coordinates", "RGB_pixels" if cfg.get("input") == "rgb" else "IR_pixels")


def load_pair(ir_dir: Path, rgb_dir: Path, seq: str):
    ir_res = _read_json(ir_dir / f"{seq}{SUFFIX_RES}")["res"]
    ir_dets = _read_json(ir_dir / f"{seq}{SUFFIX_DETS}")
    rgb_dets = _read_json(rgb_dir / f"{seq}{SUFFIX_DETS}")
    if not (len(ir_res) == len(ir_dets) == len(rgb_dets)):
        raise ValueError(f"{seq}: frame counts differ (IR res {len(ir_res)}, IR dets {len(ir_dets)}, "
                         f"RGB dets {len(rgb_dets)})")
    return ir_res, ir_dets, rgb_dets


def sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


# --------------------------------------------------------------------------
# Geometry
# --------------------------------------------------------------------------
def train_mean_theta(train: dict) -> np.ndarray:
    """Mean of the stored 2x3 matrices (equal to theta(mean params) for every linear model)."""
    rows = [np.asarray(v["theta"], float) for v in train.values() if v.get("n_pairs", 0) > 0]
    if not rows:
        raise ValueError("training theta table has no fitted video")
    return np.mean(rows, 0)


def frame_sizes(table: dict) -> tuple[tuple[int, int], tuple[int, int]]:
    ir = {tuple(v["ir_wh"]) for v in table.values()}
    rgb = {tuple(v["rgb_wh"]) for v in table.values()}
    if len(ir) != 1 or len(rgb) != 1:
        raise ValueError(f"expected one IR and one RGB frame size, found {ir} and {rgb}")
    return next(iter(ir)), next(iter(rgb))


def rgb_box_to_ir(box, theta, ir_wh, rgb_wh) -> np.ndarray:
    """Map an xywh RGB-pixel box into IR pixels through theta^-1 (axis-aligned hull)."""
    x, y, w, h = map(float, box[:4])
    corners = np.array([[x, y], [x + w, y], [x, y + h], [x + w, y + h]]) / np.asarray(rgb_wh, float)
    m = np.vstack([np.asarray(theta, float), [0.0, 0.0, 1.0]])
    inv = np.linalg.inv(m)
    p = (corners @ inv[:2, :2].T + inv[:2, 2]) * np.asarray(ir_wh, float)
    x0, y0 = p.min(0)
    x1, y1 = p.max(0)
    return np.array([x0, y0, x1 - x0, y1 - y0])


def centre(b) -> np.ndarray:
    return np.array([b[0] + b[2] / 2.0, b[1] + b[3] / 2.0])


def box_from(c, wh) -> np.ndarray:
    return np.array([c[0] - wh[0] / 2.0, c[1] - wh[1] / 2.0, wh[0], wh[1]])


def clip_to_frame(b, wh):
    """Clip to the frame; None when the centre is outside it or nothing remains."""
    c = centre(b)
    if not (0 <= c[0] < wh[0] and 0 <= c[1] < wh[1]):
        return None
    x0, y0 = max(b[0], 0.0), max(b[1], 0.0)
    x1, y1 = min(b[0] + b[2], float(wh[0])), min(b[1] + b[3], float(wh[1]))
    if x1 <= x0 or y1 <= y0:
        return None
    return np.array([x0, y0, x1 - x0, y1 - y0])


def iou_xywh(a, b) -> float:
    iw = min(a[0] + a[2], b[0] + b[2]) - max(a[0], b[0])
    ih = min(a[1] + a[3], b[1] + b[3]) - max(a[1], b[1])
    if iw <= 0 or ih <= 0:
        return 0.0
    inter = iw * ih
    return float(inter / (a[2] * a[3] + b[2] * b[3] - inter))


# --------------------------------------------------------------------------
# Fusion of one video
# --------------------------------------------------------------------------
class CausalState:
    """Recent confident IR/RGB pairs of one video; only past frames are ever read."""

    def __init__(self, window: int, k: int):
        self.window, self.k = window, k
        self.pairs = deque()      # (frame, IR centre minus mapped centre)
        self.ir_wh = deque()      # (frame, IR box width and height)

    def _trim(self, t):
        for q in (self.pairs, self.ir_wh):
            while q and q[0][0] < t - self.window:
                q.popleft()

    def correction(self, t):
        self._trim(t)
        if not self.pairs:
            return None
        return np.median(np.array([d for _, d in list(self.pairs)[-self.k:]]), 0)

    def recent_wh(self, t):
        self._trim(t)
        return None if not self.ir_wh else self.ir_wh[-1][1]

    def observe(self, t, ir_box, mapped_box, both_confident: bool):
        if ir_box is None:
            return
        self.ir_wh.append((t, np.asarray(ir_box[2:4], float)))
        if both_confident:
            self.pairs.append((t, centre(ir_box) - centre(mapped_box)))


def propose(t, rgb_box, theta, state: CausalState, args, ir_wh, rgb_wh):
    """The IR-pixel box the rule would output from this RGB box, and its sources."""
    mapped = rgb_box_to_ir(rgb_box, theta, ir_wh, rgb_wh)
    c, map_src = centre(mapped), args.map
    if args.map == "online":
        corr = state.correction(t)
        if corr is None:
            map_src = "online_fallback"
        else:
            c = c + corr
    wh, size_src = mapped[2:], "mapped"
    if args.size == "ir_recent":
        rw = state.recent_wh(t)
        if rw is None:
            size_src = "mapped_fallback"
        else:
            wh, size_src = rw, "ir_recent"
    return box_from(c, wh), mapped, map_src, size_src


def fuse_video(seq, ir_res, ir_dets, rgb_dets, theta, args, ir_wh, rgb_wh):
    state = CausalState(args.window, args.online_k)
    res_out, dets_out, log, fidelity = [], [], [], []
    for t, (res, di, dr) in enumerate(zip(ir_res, ir_dets, rgb_dets)):
        ir_top = di[0] if di else None
        rgb_top = dr[0] if dr else None
        ci = float(ir_top[4]) if ir_top else 0.0
        cr = float(rgb_top[4]) if rgb_top else 0.0
        ir_box = np.asarray(ir_top[:4], float) if ir_top and ci >= args.pair_conf else None
        out_res, out_dets = res, di
        action, map_src, size_src = "keep_ir", "", ""
        mapped = proposal = None
        if rgb_top is not None:
            proposal, mapped, map_src, size_src = propose(t, rgb_top[:4], theta, state, args, ir_wh, rgb_wh)
        if ci < args.ir_conf and cr >= args.rgb_conf:
            box = clip_to_frame(proposal, ir_wh)
            if box is None:
                action = "keep_ir_outside_fov"
            else:
                action = "replace"
                out_res = [float(v) for v in box]
                out_dets = [out_res + [cr]] + list(di)
        elif ci < args.ir_conf:
            action = "keep_ir_rgb_not_confident"
        # label-free transfer fidelity: what the rule WOULD output where IR is confident
        if ir_box is not None and cr >= args.pair_conf and proposal is not None:
            fidelity.append(iou_xywh(proposal, ir_box))
        state.observe(t, ir_box, mapped, ir_box is not None and cr >= args.pair_conf and mapped is not None)
        res_out.append(out_res)
        dets_out.append(out_dets)
        log.append([seq, t, round(ci, 6), round(cr, 6), action, map_src if action == "replace" else "",
                    size_src if action == "replace" else ""])
    return res_out, dets_out, log, fidelity


def summarise(log_rows, fidelity):
    actions = {}
    for r in log_rows:
        actions[r[4]] = actions.get(r[4], 0) + 1
    rep = [r for r in log_rows if r[4] == "replace"]
    out = {"frames": len(log_rows), "actions": actions,
           "replaced_map_sources": {}, "replaced_size_sources": {}}
    for r in rep:
        out["replaced_map_sources"][r[5]] = out["replaced_map_sources"].get(r[5], 0) + 1
        out["replaced_size_sources"][r[6]] = out["replaced_size_sources"].get(r[6], 0) + 1
    f = np.asarray(fidelity, float)
    out["transfer_fidelity"] = {
        "frames": int(len(f)),
        "median_iou_vs_confident_ir": float(np.median(f)) if len(f) else None,
        "share_iou_ge_0.5": float(np.mean(f >= 0.5)) if len(f) else None,
        "note": "label-free: IoU of the rule's proposal against a confident IR box, not against labels",
    }
    return out


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ir-run", required=True, type=Path)
    ap.add_argument("--rgb-run", required=True, type=Path)
    ap.add_argument("--train-theta", required=True, type=Path, help="seq_theta_train.json")
    ap.add_argument("--seq-theta", type=Path, help="seq_theta_<split>.json; required only for --map seq")
    ap.add_argument("--map", default="train_mean", choices=["train_mean", "seq", "online"])
    ap.add_argument("--size", default="mapped", choices=["mapped", "ir_recent"])
    ap.add_argument("--ir-conf", type=float, default=0.5,
                    help="replace the IR output when its top confidence is below this (0.25 = empty frames only)")
    ap.add_argument("--rgb-conf", type=float, default=0.5, help="minimum RGB top confidence to replace")
    ap.add_argument("--pair-conf", type=float, default=0.5,
                    help="both detectors at least this confident -> a pair for the online correction")
    ap.add_argument("--window", type=int, default=30, help="online/ir_recent memory, frames")
    ap.add_argument("--online-k", type=int, default=5, help="online: median over the last k pairs")
    ap.add_argument("--allow-test", action="store_true", help="permit split=test (off by protocol)")
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args(argv)
    if not 0 < args.rgb_conf <= 1 or not 0 <= args.ir_conf <= 1 or not 0 < args.pair_conf <= 1:
        ap.error("confidence thresholds must lie in (0, 1]")
    if args.window < 1 or args.online_k < 1:
        ap.error("--window and --online-k must be positive")
    if args.map == "seq" and args.seq_theta is None:
        ap.error("--map seq requires --seq-theta")
    return args


def main(argv=None):
    args = parse_args(argv)
    cfg_ir, cfg_rgb = run_config(args.ir_run), run_config(args.rgb_run)
    if coordinates(cfg_ir) != "IR_pixels" or coordinates(cfg_rgb) != "RGB_pixels":
        raise ValueError("--ir-run must hold IR-pixel boxes and --rgb-run RGB-pixel boxes")
    split = cfg_ir.get("split")
    if split != cfg_rgb.get("split"):
        raise ValueError(f"runs come from different splits: {split} vs {cfg_rgb.get('split')}")
    if split == "test" and not args.allow_test:
        raise SystemExit("refusing split=test (protocol: no test evaluation); pass --allow-test to override")
    seqs = run_sequences(args.ir_run)
    if not seqs or seqs != run_sequences(args.rgb_run):
        raise ValueError("the two runs must contain exactly the same, non-empty set of videos")

    train = json.loads(args.train_theta.read_text())
    ir_wh, rgb_wh = frame_sizes(train)
    th_mean = train_mean_theta(train)
    seq_table = json.loads(args.seq_theta.read_text()) if args.seq_theta else None
    if seq_table is not None and frame_sizes(seq_table) != (ir_wh, rgb_wh):
        raise ValueError("training and evaluation theta tables disagree on frame sizes")
    if args.map == "seq":
        missing = [s for s in seqs if s not in seq_table or seq_table[s].get("n_pairs", 0) <= 0]
        if missing:
            raise ValueError(f"--map seq: no fitted transform for {missing}")

    args.out.mkdir(parents=True, exist_ok=True)
    all_log, all_fid, per_video = [], [], {}
    for seq in seqs:
        ir_res, ir_dets, rgb_dets = load_pair(args.ir_run, args.rgb_run, seq)
        theta = np.asarray(seq_table[seq]["theta"], float) if args.map == "seq" else th_mean
        res, dets, log, fid = fuse_video(seq, ir_res, ir_dets, rgb_dets, theta, args, ir_wh, rgb_wh)
        (args.out / f"{seq}{SUFFIX_RES}").write_text(json.dumps({"res": res}))
        (args.out / f"{seq}{SUFFIX_DETS}").write_text(json.dumps(dets))
        per_video[seq] = summarise(log, fid)
        all_log += log
        all_fid += fid
    with open(args.out / "fusion_log.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["seq", "frame", "ir_conf", "rgb_conf", "action", "map_source", "size_source"])
        w.writerows(all_log)
    summary = {"split": split, "videos": len(seqs), "pooled": summarise(all_log, all_fid),
               "per_video": per_video,
               "uses_evaluation_labels": args.map == "seq"}
    (args.out / "fusion_summary.json").write_text(json.dumps(summary, indent=1, allow_nan=False))
    config = {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()}
    config.update({"tool": "decision_fusion.py", "split": split, "output_coordinates": "IR_pixels",
                   "prior": "zero", "ir_wh": list(ir_wh), "rgb_wh": list(rgb_wh),
                   "train_mean_theta": th_mean.tolist(),
                   "uses_evaluation_labels": args.map == "seq",
                   "input_sha256": {"train_theta": sha256(args.train_theta),
                                    **({"seq_theta": sha256(args.seq_theta)} if args.seq_theta else {})},
                   "source_runs": {"ir": cfg_ir.get("weights"), "rgb": cfg_rgb.get("weights")}})
    (args.out / "run_config.json").write_text(json.dumps(config, indent=1))
    p = summary["pooled"]
    print(f"videos {len(seqs)}, frames {p['frames']}, actions {p['actions']}")
    print(f"transfer fidelity (label-free): {json.dumps(p['transfer_fidelity'])}")
    print("written:", args.out)


if __name__ == "__main__":
    main()
