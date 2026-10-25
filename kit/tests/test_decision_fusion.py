"""Checks for the decision-level fusion tool (no video, no torch)."""
import gzip
import importlib.util
import json
import sys
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path

import numpy as np

TOOLS = Path(__file__).parents[1] / "tools"
sys.path.insert(0, str(TOOLS))


def load(name):
    spec = importlib.util.spec_from_file_location(name, TOOLS / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


df = load("decision_fusion")
IR_WH, RGB_WH = (640, 512), (1920, 1080)
THETA = np.array([[0.8, 0.0, 0.1], [0.0, 1.2, -0.2]])


def ir_to_rgb(box):
    """Forward map of an IR-pixel box through THETA (the inverse of what the tool does)."""
    x, y, w, h = box
    p = np.array([[x, y], [x + w, y + h]]) / IR_WH
    q = (p @ THETA[:, :2].T + THETA[:, 2]) * RGB_WH
    return [q[0, 0], q[0, 1], q[1, 0] - q[0, 0], q[1, 1] - q[0, 1]]


def args(**kw):
    base = dict(map="train_mean", size="mapped", ir_conf=0.5, rgb_conf=0.5, pair_conf=0.5,
                window=30, online_k=5)
    base.update(kw)
    return Namespace(**base)


def theta_table(theta=THETA):
    return {"a": {"model": "scale_translation", "theta": np.asarray(theta).tolist(), "n_pairs": 50,
                  "ir_wh": list(IR_WH), "rgb_wh": list(RGB_WH)}}


def write_run(d, split, coords, seqs, gz=False):
    d.mkdir(parents=True)
    cfg = {"split": split, "output_coordinates": coords, "weights": "w.pt"}
    files = {"run_config.json": cfg}
    for name, (res, dets) in seqs.items():
        files[f"{name}_IR.txt"] = {"res": res}
        files[f"{name}_dets.json"] = dets
    for fname, obj in files.items():
        if gz:
            with gzip.open(d / (fname + ".gz"), "wt") as f:
                f.write(json.dumps(obj))
        else:
            (d / fname).write_text(json.dumps(obj))


class Geometry(unittest.TestCase):
    def test_rgb_box_maps_back_to_the_ir_box(self):
        box = [300.0, 200.0, 20.0, 16.0]
        back = df.rgb_box_to_ir(ir_to_rgb(box), THETA, IR_WH, RGB_WH)
        np.testing.assert_allclose(back, box, atol=1e-9)

    def test_clip_and_outside(self):
        self.assertIsNone(df.clip_to_frame(np.array([700.0, 10, 20, 20]), IR_WH))
        np.testing.assert_allclose(df.clip_to_frame(np.array([-5.0, 10, 20, 20]), IR_WH), [0, 10, 15, 20])

    def test_train_mean_ignores_unfitted_videos(self):
        t = {"a": {"theta": [[1, 0, 0], [0, 1, 0]], "n_pairs": 10},
             "b": {"theta": [[3, 0, 1], [0, 3, 1]], "n_pairs": 30},
             "c": {"theta": [[9, 0, 9], [0, 9, 9]], "n_pairs": 0}}
        np.testing.assert_allclose(df.train_mean_theta(t), [[2, 0, 0.5], [0, 2, 0.5]])


class Rule(unittest.TestCase):
    def run_one(self, ir_dets, rgb_dets, **kw):
        res = [d[0][:4] if d and d[0][4] >= 0.25 else [] for d in ir_dets]
        return df.fuse_video("s", res, ir_dets, rgb_dets, THETA, args(**kw), IR_WH, RGB_WH)

    def test_gates(self):
        gt = [300.0, 200.0, 20.0, 16.0]
        rgb = ir_to_rgb(gt)
        ir_dets = [[gt + [0.9]], [], [gt + [0.3]], [], [[10, 10, 5, 5, 0.1]]]
        rgb_dets = [[rgb + [0.9]], [rgb + [0.9]], [rgb + [0.9]], [rgb + [0.4]], []]
        res, dets, log, _ = self.run_one(ir_dets, rgb_dets)
        self.assertEqual([r[4] for r in log],
                         ["keep_ir", "replace", "replace", "keep_ir_rgb_not_confident", "keep_ir_rgb_not_confident"])
        self.assertEqual(res[0], gt)
        np.testing.assert_allclose(res[1], gt, atol=1e-9)
        self.assertEqual(res[3], [])
        self.assertEqual(dets[1][0][4], 0.9)                 # mapped box inserted with RGB confidence
        self.assertEqual(dets[4], ir_dets[4])                # untouched candidates

    def test_empty_only_threshold_keeps_low_confidence_ir(self):
        gt = [300.0, 200.0, 20.0, 16.0]
        res, _, log, _ = self.run_one([[gt + [0.3]]], [[ir_to_rgb([100, 100, 20, 16]) + [0.9]]], ir_conf=0.25)
        self.assertEqual(log[0][4], "keep_ir")
        self.assertEqual(res[0], gt)

    def test_outside_fov_keeps_ir(self):
        far = [[1.0, 0, 5.0], [0, 1.0, 0]]                    # RGB content maps far left of the IR frame
        res, _, log, _ = df.fuse_video("s", [[]], [[]], [[[900, 500, 30, 20, 0.9]]], np.array(far), args(),
                                       IR_WH, RGB_WH)
        self.assertEqual(log[0][4], "keep_ir_outside_fov")
        self.assertEqual(res[0], [])

    def test_online_correction_is_causal(self):
        """The offset learnt from past pairs is used; the current frame's own pair is not."""
        n = 12
        ir_dets, rgb_dets = [], []
        for t in range(n):
            gt = [300.0 + t, 200.0, 20.0, 16.0]
            true_rgb = ir_to_rgb(gt)
            shifted = [true_rgb[0] + 30.0, *true_rgb[1:]]      # constant +30 RGB px bias in the mapping
            rgb_dets.append([shifted + [0.9]])
            ir_dets.append([gt + [0.9]] if t != n - 1 else [])
        _, _, log, fid = self.run_one(ir_dets, rgb_dets, map="online")
        res, _, _, _ = self.run_one(ir_dets, rgb_dets, map="online")
        self.assertEqual(log[-1][4:6], ["replace", "online"])
        np.testing.assert_allclose(res[-1], [300.0 + n - 1, 200, 20, 16], atol=1e-6)
        self.assertLess(fid[0], 1.0)                           # first pair: no history yet
        self.assertAlmostEqual(fid[-1], 1.0, places=6)

    def test_ir_recent_size(self):
        gt = [300.0, 200.0, 20.0, 16.0]
        wide = ir_to_rgb([295.0, 203.0, 30.0, 10.0])           # same centre, other shape
        _, _, log, _ = self.run_one([[gt + [0.9]], []], [[wide + [0.9]], [wide + [0.9]]], size="ir_recent")
        res, _, _, _ = self.run_one([[gt + [0.9]], []], [[wide + [0.9]], [wide + [0.9]]], size="ir_recent")
        self.assertEqual(log[1][6], "ir_recent")
        np.testing.assert_allclose(res[1], gt, atol=1e-6)


class _Runs(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        gt = [300.0, 200.0, 20.0, 16.0]
        self.ir = {"v1": ([gt, []], [[gt + [0.9]], []]), "v2": ([[]], [[]])}
        self.rgb = {"v1": ([[], []], [[], [ir_to_rgb(gt) + [0.8]]]), "v2": ([[]], [[]])}
        (self.root / "train.json").write_text(json.dumps(theta_table()))

    def tearDown(self):
        self.tmp.cleanup()

    def call(self, split="val", gz=False, rgb_coords="RGB_pixels", extra=()):
        write_run(self.root / "ir", split, "IR_pixels", self.ir, gz)
        write_run(self.root / "rgb", split, rgb_coords, self.rgb, gz)
        df.main(["--ir-run", str(self.root / "ir"), "--rgb-run", str(self.root / "rgb"),
                 "--train-theta", str(self.root / "train.json"), "--out", str(self.root / "out"), *extra])
        return self.root / "out"


class EndToEnd(_Runs):
    def test_outputs_and_compressed_inputs(self):
        out = self.call(gz=True)
        res = json.loads((out / "v1_IR.txt").read_text())["res"]
        self.assertEqual(res[0], [300.0, 200.0, 20.0, 16.0])
        np.testing.assert_allclose(res[1], [300.0, 200.0, 20.0, 16.0], atol=1e-6)
        cfg = json.loads((out / "run_config.json").read_text())
        self.assertEqual(cfg["output_coordinates"], "IR_pixels")
        self.assertFalse(cfg["uses_evaluation_labels"])
        summ = json.loads((out / "fusion_summary.json").read_text())
        self.assertEqual(summ["pooled"]["actions"], {"keep_ir": 1, "replace": 1, "keep_ir_rgb_not_confident": 1})

    def test_refusals(self):
        with self.assertRaises(SystemExit):
            self.call(split="test")
        self.tmp.cleanup()
        self.setUp()
        with self.assertRaises(ValueError):
            self.call(rgb_coords="IR_pixels")
        self.tmp.cleanup()
        self.setUp()
        self.rgb["v1"] = ([[]], [[]])                           # length mismatch
        with self.assertRaises(ValueError):
            self.call()
        self.tmp.cleanup()
        self.setUp()
        del self.rgb["v2"]                                      # missing video
        with self.assertRaises(ValueError):
            self.call()

    def test_unchanged_scorer_reads_the_output(self):
        out = self.call()
        sr = load("score_runs")
        labels = {"v1": (np.array([1, 1]), np.array([[300.0, 200, 20, 16]] * 2)),
                  "v2": (np.array([0]), np.full((1, 4), np.nan))}
        base, fused = sr.load_run([str(self.root / "ir")], labels), sr.load_run([str(out)], labels)
        self.assertAlmostEqual(np.mean(base["v1"]["SA"]), 0.5)
        self.assertAlmostEqual(np.mean(fused["v1"]["SA"]), 1.0, places=6)
        self.assertEqual(fused["v2"]["SA"], [1.0])


if __name__ == "__main__":
    unittest.main()


class SyncAudit(unittest.TestCase):
    def test_recovers_a_known_lag(self):
        sa = load("sync_audit")
        rng = np.random.default_rng(0)
        n, lag = 300, 4                                         # RGB frame t+4 shows IR frame t
        path = np.cumsum(rng.normal(0, 0.01, (n + lag, 2)), 0) * 0.3 + 0.5
        ir_c = path[:n]
        rgb_c = np.full((n, 2), np.nan)
        rgb_c[lag:] = path[: n - lag] @ THETA[:, :2].T + THETA[:, 2]
        a = Namespace(window=0, max_lag=8, model="scale_translation", min_pairs=40)
        win = sa.audit_video(ir_c, rgb_c, a, IR_WH)[0]
        self.assertEqual(win["best_lag"], lag)
        self.assertLess(win["resid_px_best"], 1e-6)
        self.assertGreater(win["resid_px_lag0"], 0.5)


class Breakdown(_Runs):
    def test_replaced_frames_split_by_label_state(self):
        out = self.call()
        for name, exist, rect in (("v1", [1, 0], [[300, 200, 20, 16], []]), ("v2", [0], [[]])):
            d = self.root / "raw" / "val" / name
            d.mkdir(parents=True)
            (d / "infrared.json").write_text(json.dumps({"exist": exist, "gt_rect": rect}))
        fb = load("fusion_breakdown")
        fb.main(["--root", str(self.root / "raw"), "--ir-run", str(self.root / "ir"), "--fused", str(out),
                 "--out", str(self.root / "b.json")])
        rep = json.loads((self.root / "b.json").read_text())
        self.assertEqual(rep["replaced_frames"]["ir_negative"]["frames"], 1)
        self.assertEqual(rep["replaced_frames"]["ir_negative"]["sum_score_fused"], 0.0)
        self.assertAlmostEqual(rep["per_video"]["v1"]["sa_change_in_video"], -0.5)
