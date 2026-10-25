#!/usr/bin/env python3
"""Paired sequence bootstrap of the SA difference on a declared subset of videos.

Reads the per-video SA CSV written by score_runs.py --per-seq, drops the listed
videos, and repeats score_runs.py's SA bootstrap (resample videos with
replacement, same draw for both runs, mean of per-video differences). With no
exclusions it reproduces score_runs.py's SA interval exactly for the FIRST
comparison of a score_runs.py invocation (checked against the v6 A1_blankrgb-A1
result); later comparisons there share one generator, so their draws differ.

Usage:
  python kit/tools/subset_bootstrap.py --per-seq reports/fusion_val_per_seq.csv --a P --b A0 \
     --exclude 20190925_213001_1_1 --label without_213001_1_1 --out reports/fusion_P_subset.json
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def load(path, a, b):
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    out = {}
    for r in rows:
        va, vb = r.get(f"{a}_SA", ""), r.get(f"{b}_SA", "")
        name = r.get('sequence')
        if not name or name in out or va == '' or vb == '':
            raise ValueError('Every video needs unique, complete paired SA values')
        pair = np.asarray([float(va), float(vb)])
        if not np.isfinite(pair).all(): raise ValueError('SA values must be finite')
        out[name] = float(pair[0] - pair[1])
    if not out: raise ValueError('Empty per-sequence CSV')
    return out


def bootstrap(diffs, n_boot, seed):
    if not diffs or n_boot < 1: raise ValueError('Nonempty differences and positive bootstrap count required')
    seqs = sorted(diffs)
    d = np.array([diffs[s] for s in seqs])
    rng = np.random.default_rng(seed)
    draws = np.array([d[rng.integers(0, len(d), len(d))].mean() for _ in range(n_boot)])
    return {"point": float(d.mean()), "boot_mean": float(draws.mean()),
            "ci95": [float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))],
            "share_le_0": float(np.mean(draws <= 0)), "n_seq": len(d)}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--per-seq", required=True, type=Path)
    ap.add_argument("--a", required=True, help="run name (column <a>_SA)")
    ap.add_argument("--b", required=True, help="baseline run name")
    ap.add_argument("--exclude", nargs="*", default=[])
    ap.add_argument("--exclude-file", type=Path, help="JSON list of videos to exclude")
    ap.add_argument("--label", default="subset")
    ap.add_argument("--boot", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args(argv)
    excl = set(args.exclude) | set(json.loads(args.exclude_file.read_text()) if args.exclude_file else [])
    diffs = load(args.per_seq, args.a, args.b)
    unknown = excl - set(diffs)
    if unknown:
        raise ValueError(f"excluded videos not in the CSV: {sorted(unknown)}")
    kept = {k: v for k, v in diffs.items() if k not in excl}
    if not kept:
        raise ValueError("no video left")
    res = {"label": args.label, "comparison": f"{args.a}-{args.b}", "metric": "SA",
           "excluded": sorted(excl), **bootstrap(kept, args.boot, args.seed)}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(res, indent=1))
    print(json.dumps(res))


if __name__ == "__main__":
    main()
