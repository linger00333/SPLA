#!/usr/bin/env python3
"""Evaluate the retention gate fixed before the low-LR A1c pilot."""
import argparse
import csv
import json
from pathlib import Path


def evaluate(directory):
    baseline = json.loads((directory / "pretrain_validation.json").read_text())
    baseline = float(baseline["metrics"]["metrics/mAP50-95(B)"])
    with (directory / "results.csv").open() as stream:
        rows = [{key.strip(): value for key, value in row.items()}
                for row in csv.DictReader(stream)]
    if len(rows) != 10:
        raise ValueError("The retention gate requires exactly 10 completed epochs")
    values = [float(row["metrics/mAP50-95(B)"]) for row in rows]
    checks = {
        "constant_lr": all(abs(float(row["lr/pg0"]) - 0.0001) < 1e-12 for row in rows),
        "final_within_0.003": values[-1] >= baseline - 0.003,
        "last3_within_0.003": sum(values[-3:]) / 3 >= baseline - 0.003,
        "minimum_within_0.005": min(values) >= baseline - 0.005,
        "first3_to_last3_drop_at_most_0.003": sum(values[-3:]) / 3 >= sum(values[:3]) / 3 - 0.003,
    }
    return {"baseline_mAP50_95": baseline, "per_epoch_mAP50_95": values,
            "checks": checks, "passed": all(checks.values()),
            "criterion_fixed_before_training": True}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    report = evaluate(args.run)
    text = json.dumps(report, indent=2, allow_nan=False) + "\n"
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text, encoding="ascii")
    print(text, end="")
    raise SystemExit(0 if report["passed"] else 2)
