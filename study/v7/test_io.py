"""Additional publication-format checks without real labels or videos."""
from pathlib import Path
import gzip
import importlib.util
import json
import sys
import tempfile
import unittest
import numpy as np
V = Path(__file__).resolve().parent
repo = V.parents[1] / 'github/SPLA'
if not repo.exists(): repo = V.parents[1]
sys.path[:0] = [str(repo / 'kit'), str(repo / 'kit/tools')]
from spla.prediction_io import read_json
import score_runs
from fusion_breakdown import frame_score

class Formats(unittest.TestCase):
    def test_gzip_folder_is_scored(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            with gzip.open(d / 'v_IR.txt.gz', 'wt') as f: json.dump({'res': [[0, 0, 2, 2]]}, f)
            out = score_runs.load_run([d], {'v': (np.array([1]), np.array([[0, 0, 2, 2]]))})
            self.assertEqual(out['v']['SA'], [1.0])
    def test_conflicting_dual_files_raise(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / 'v.json'; p.write_text('[1]')
            with gzip.open(str(p) + '.gz', 'wt') as f: f.write('[2]')
            with self.assertRaises(ValueError): read_json(p)
    def test_rgb_metadata_refused_by_ir_scorer(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp); (d / 'run_config.json').write_text(json.dumps({'output_coordinates': 'RGB_pixels'}))
            with self.assertRaises(ValueError): score_runs.load_run([d], {'v': (np.array([0]), np.full((1, 4), np.nan))})
    def test_existing_negative_false_positive_has_zero_added_cost(self):
        wrong = [0, 0, 2, 2]; box = np.full(4, np.nan)
        self.assertEqual(frame_score(wrong, 0, box) - frame_score(wrong, 0, box), 0)
        self.assertEqual(frame_score(wrong, 0, box) - frame_score([], 0, box), -1)

if __name__ == '__main__': unittest.main()
