"""File discovery for Anti-UAV RGB-T sequences (several releases use different names).

Known spellings:
  official eval code (Eval/test.py):      IR.mp4, RGB.mp4, IR_label.json, RGB_label.json
  modality names in the release notes:    infrared.*, visible.*
Run tools/inspect_dataset.py once on your download to see which one you have.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

# Sequences listed in the annotation-correction table of the Anti-UAV release
# (github.com/ucas-vg/Anti-UAV README: sequence, modality, start frame). Their
# labels need special handling, so the audit reports them apart.
KNOWN_LABEL_ISSUES = {
    "20190925_130434_1_2": "infrared labels: start frame 16",
    "20190926_134054_1_1": "visible labels: start frame 150",
    "20190925_194211_1_3": "visible labels: start frame 150",
    "20190925_194211_1_4": "visible labels: start frame 1",
}

CANDIDATES = {
    "ir_video": ["infrared.mp4", "IR.mp4", "ir.mp4"],
    "rgb_video": ["visible.mp4", "RGB.mp4", "rgb.mp4"],
    "ir_label": ["IR_label.json", "infrared.json", "IR.json"],
    "rgb_label": ["RGB_label.json", "visible.json", "RGB.json"],
}


def find_files(seq_dir: Path) -> dict:
    """Map role -> existing path (or None) for one sequence folder."""
    seq_dir = Path(seq_dir)
    out = {}
    for role, names in CANDIDATES.items():
        out[role] = next((seq_dir / n for n in names if (seq_dir / n).exists()), None)
    return out


def list_sequences(split_dir: Path, need=("ir_label", "rgb_label")) -> list:
    """Sequence folders that contain every role in `need`."""
    split_dir = Path(split_dir)
    if not split_dir.exists():
        return []
    return sorted(p for p in split_dir.iterdir()
                  if p.is_dir() and all(find_files(p)[r] is not None for r in need))


def read_labels(path: Path, box_format: str = "xywh"):
    """Return exist (T,) int and boxes (T, 4) as x, y, w, h (NaN where absent).

    box_format: "xywh" (what the official evaluation code assumes) or "xyxy"
    (what the v1 paper's text says); tools/inspect_dataset.py tells you which.
    """
    d = json.loads(Path(path).read_text())
    exist = np.asarray(d["exist"], int)
    boxes = np.full((len(exist), 4), np.nan)
    for i, r in enumerate(d["gt_rect"]):
        if exist[i] and r and len(r) == 4:
            x, y, a, b = map(float, r)
            w, h = (a - x, b - y) if box_format == "xyxy" else (a, b)
            if w > 0 and h > 0:
                boxes[i] = (x, y, w, h)
                continue
        exist[i] = 0
    return exist, boxes
