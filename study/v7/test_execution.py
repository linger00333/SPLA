"""Checks for the additional held-out trajectory diagnostic."""
import importlib.util
from pathlib import Path
import sys
import unittest
import numpy as np
V = Path(__file__).resolve().parent
repo = V.parents[1] / 'github/SPLA'
sys.path[:0] = [str(repo / 'kit'), str(repo / 'kit/tools')]
spec = importlib.util.spec_from_file_location('heldout_sync', V / 'heldout_sync.py')
module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)

class Heldout(unittest.TestCase):
    def fixture(self):
        rng = np.random.default_rng(37)
        a = np.cumsum(rng.normal(0, .01, (360, 2)), axis=0) + .5
        b = np.empty_like(a); b[:4] = .5
        b[4:] = a[:-4] * [.8, 1.2] + [.1, -.2]
        return a, b
    def test_recovers_synthetic_lag_on_later_block(self):
        a, b = self.fixture(); out = module.evaluate(a, b)
        self.assertEqual(out['selected_training_lag'], 4)
        self.assertLess(out['heldout_residual_selected_px'], 1e-8)
        self.assertTrue(out['heldout_candidate'])
    def test_later_block_does_not_choose_lag(self):
        a, b = self.fixture(); before = module.evaluate(a, b)
        b[len(b)*2//3:] += [.3, -.4]
        after = module.evaluate(a, b)
        self.assertEqual(before['selected_training_lag'], after['selected_training_lag'])
        self.assertGreater(after['heldout_residual_selected_px'], before['heldout_residual_selected_px'])
    def test_static_track_is_not_sync_ground_truth(self):
        a = np.ones((360, 2)) * .5
        out = module.evaluate(a, a)
        self.assertEqual(out['status'], 'singular_spatial_fit')

if __name__ == '__main__': unittest.main()
