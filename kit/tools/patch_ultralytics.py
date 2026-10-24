#!/usr/bin/env python3
"""Register the SPLA layers in a YOLOv11-RGBT checkout (idempotent).

Makes two edits to ultralytics/nn/tasks.py:
  1. imports SPLA, GlobalOffsetHead and Pick so that parse_model can find
     them by name (it looks modules up in the globals of tasks.py);
  2. tells parse_model how many channels each produces and passes the input
     channel counts as the first constructor arguments.

Usage:
  python tools/patch_ultralytics.py /path/to/YOLOv11-RGBT
The spla package must be importable (pip install -e /path/to/spla_kit, or
add it to PYTHONPATH).
"""
import re
import sys
from pathlib import Path

MARK = "# SPLA-PATCH"
IMPORT = f"from spla.modules import SPLA, GlobalOffsetHead, Pick  {MARK}\n\n\n"
BRANCH = (
    f"        elif m in {{SPLA, GlobalOffsetHead}}:  {MARK}\n"
    "            c2 = ch[f[0]] if m is SPLA else 2\n"
    "            args = [ch[f[0]], ch[f[1]], *args]\n"
    f"        elif m is Pick:  {MARK}\n"
    "            c2 = ch[f[0]] if isinstance(f, list) else ch[f]\n"
)


def patch(repo: Path) -> None:
    p = repo / "ultralytics" / "nn" / "tasks.py"
    src = p.read_text(encoding="utf-8")
    if MARK in src:
        print(f"{p}: already patched")
        return
    i = src.index("def parse_model(")
    src = src[:i] + IMPORT + src[i:]
    # insert the branch before the SilenceChannel branch inside parse_model
    m = re.search(r"^        elif m is SilenceChannel:\n", src, flags=re.M)
    if m is None:
        raise SystemExit("could not find 'elif m is SilenceChannel:' in parse_model; patch by hand")
    src = src[:m.start()] + BRANCH + src[m.start():]
    p.write_text(src, encoding="utf-8")
    print(f"{p}: patched")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit(__doc__)
    patch(Path(sys.argv[1]))
