"""Regression checks for optional per-video SA export."""
import csv
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

spec = importlib.util.spec_from_file_location("score_runs", Path(__file__).parents[1] / "tools/score_runs.py")
score = importlib.util.module_from_spec(spec)
spec.loader.exec_module(score)


class PerSequenceExportTests(unittest.TestCase):
    def test_seed_means_and_missing_scores(self):
        runs = {"A0": {"v1": {"SA": [0.2, 0.4]}, "v2": {"SA": [np.nan]}},
                "A1": {"v1": {"SA": [0.5]}, "v2": {"SA": [0.7]}}}
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder) / "reports/scores.csv"
            score.write_per_sequence(runs, ["v1", "v2", "missing"], output)
            with output.open(newline="") as stream:
                rows = list(csv.DictReader(stream))
        self.assertEqual(list(rows[0]), ["sequence", "A0_SA", "A1_SA"])
        self.assertAlmostEqual(float(rows[0]["A0_SA"]), 0.3)
        self.assertEqual(rows[1]["A0_SA"], "")
        self.assertEqual(rows[2], {"sequence": "missing", "A0_SA": "", "A1_SA": ""})

    def test_csv_quotes_sequence_and_run_names(self):
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder) / "scores.csv"
            score.write_per_sequence({"model,one": {"video,one": {"SA": [0.8]}}}, ["video,one"], output)
            with output.open(newline="") as stream:
                row = next(csv.DictReader(stream))
        self.assertEqual(row["sequence"], "video,one")
        self.assertEqual(float(row["model,one_SA"]), 0.8)

    def test_optional_export_preserves_aggregate_and_bootstrap(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            for name, pred in (("A0", [[], []]), ("A1", [[0, 0, 2, 2], []])):
                directory = root / name
                directory.mkdir()
                (directory / "video_IR.txt").write_text(json.dumps({"res": pred}))
            args = ["--root", str(root), "--split", "val", "--run", "A0=" + str(root / "A0"),
                    "--run", "A1=" + str(root / "A1"), "--baseline", "A0", "--boot", "8"]
            labels = (np.array([1, 0]), np.array([[0, 0, 2, 2], [np.nan] * 4]))
            with patch.object(score, "list_sequences", return_value=[root / "video"]), \
                    patch.object(score, "find_files", return_value={"ir_label": root / "label.json"}), \
                    patch.object(score, "read_labels", return_value=labels):
                score.main(args + ["--out", str(root / "before.json")])
                score.main(args + ["--out", str(root / "after.json"), "--per-seq", str(root / "per.csv")])
            self.assertEqual((root / "before.json").read_bytes(), (root / "after.json").read_bytes())
            with (root / "per.csv").open(newline="") as stream:
                row = next(csv.DictReader(stream))
            self.assertEqual(float(row["A0_SA"]), 0.5)
            self.assertEqual(float(row["A1_SA"]), 1.0)


if __name__ == "__main__":
    unittest.main()
