"""Regression checks for strict coverage, causal fusion and repeatable inference."""
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
import numpy as np

TOOLS = Path(__file__).parents[1] / 'tools'
sys.path.insert(0, str(TOOLS))
def load(name):
    spec = importlib.util.spec_from_file_location(name, TOOLS / (name + '.py'))
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module

class StrictScorer(unittest.TestCase):
    def test_missing_predictions_raise(self):
        score = load('score_runs')
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(FileNotFoundError):
                score.load_run([tmp], {'v': (np.array([1]), np.array([[1., 1., 2., 2.]]))})

    def test_truncated_or_extra_frames_raise(self):
        score = load('score_runs')
        for pred in ([], [[], []]):
            with self.assertRaises(ValueError):
                score.state_accuracy(pred, np.array([0]), np.full((1, 4), np.nan))

    def test_comparison_intervals_do_not_depend_on_argument_order(self):
        score = load('score_runs')
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for i in range(7):
                seq = root / 'raw/val' / str(i); seq.mkdir(parents=True)
                (seq / 'infrared.json').write_text(json.dumps({'exist': [1], 'gt_rect': [[0, 0, 10, 10]]}))
                for name, dx in (('A0', 1), ('P', i), ('S', 6-i)):
                    d = root / name; d.mkdir(exist_ok=True)
                    (d / f'{i}_IR.txt').write_text(json.dumps({'res': [[dx, 0, 10, 10]]}))
            results = []
            for order in (('A0', 'P', 'S'), ('A0', 'S', 'P')):
                out = root / ('_'.join(order) + '.json')
                argv = ['--root', str(root / 'raw'), '--split', 'val', '--boot', '100', '--baseline', 'A0', '--out', str(out)]
                for name in order: argv += ['--run', name + '=' + str(root / name)]
                score.main(argv)
                results.append(json.loads(out.read_text())['vs_baseline'])
            self.assertEqual(results[0], results[1])

class FusionCoverage(unittest.TestCase):
    def test_direct_fusion_refuses_truncation(self):
        from argparse import Namespace
        df = load('decision_fusion')
        args = Namespace(window=30, online_k=5, map='train_mean', size='mapped', pair_conf=.5, ir_conf=.5, rgb_conf=.5)
        with self.assertRaises(ValueError):
            df.fuse_video('v', [[]], [], [[]], np.array([[1, 0, 0], [0, 1, 0]]), args, (640, 512), (1920, 1080))

    def test_rgb_result_length_is_checked(self):
        df = load('decision_fusion')
        with tempfile.TemporaryDirectory() as tmp:
            a, b = Path(tmp) / 'ir', Path(tmp) / 'rgb'; a.mkdir(); b.mkdir()
            (a / 'v_IR.txt').write_text(json.dumps({'res': [[]]}))
            (a / 'v_dets.json').write_text('[[]]')
            (b / 'v_IR.txt').write_text(json.dumps({'res': []}))
            (b / 'v_dets.json').write_text('[[]]')
            with self.assertRaises(ValueError): df.load_pair(a, b, 'v')

class SyncAndSubsets(unittest.TestCase):
    def test_sync_unequal_length_is_an_error(self):
        from argparse import Namespace
        sa = load('sync_audit')
        with self.assertRaises(ValueError):
            sa.audit_video(np.zeros((100, 2)), np.zeros((90, 2)), Namespace(window=0, max_lag=2, model='scale_translation', min_pairs=40), (640, 512))

    def test_subset_missing_cells_and_duplicates_raise(self):
        sb = load('subset_bootstrap')
        for body in ('sequence,P_SA,A0_SA\nv1,,0.8\n', 'sequence,P_SA,A0_SA\nv1,.9,.8\nv1,.9,.8\n'):
            with tempfile.TemporaryDirectory() as tmp:
                path = Path(tmp) / 'p.csv'; path.write_text(body)
                with self.assertRaises(ValueError): sb.load(path, 'P', 'A0')

if __name__ == '__main__': unittest.main()
