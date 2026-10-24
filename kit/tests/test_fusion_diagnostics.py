"""Checks for the RGB-only label builder and the complementarity count (no video, no torch)."""
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

TOOLS = Path(__file__).parents[1] / "tools"
sys.path.insert(0, str(TOOLS))


def load(name):
    spec = importlib.util.spec_from_file_location(name, TOOLS / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def write_seq(root, split, name, ir, rgb):
    d = Path(root) / split / name
    d.mkdir(parents=True)
    (d / "infrared.json").write_text(json.dumps({"exist": [int(b is not None) for b in ir],
                                                 "gt_rect": [b or [] for b in ir]}))
    (d / "visible.json").write_text(json.dumps({"exist": [int(b is not None) for b in rgb],
                                                "gt_rect": [b or [] for b in rgb]}))
    return d


class RgbOnlyLabels(unittest.TestCase):
    def test_labels_use_rgb_boxes_normalised_by_rgb_size(self):
        mk = load("make_rgb_only")
        with tempfile.TemporaryDirectory() as tmp:
            root, out = Path(tmp) / "raw", Path(tmp) / "yolo"
            write_seq(root, "val", "v1", [[10, 10, 4, 4], None], [[960, 540, 192, 108], None])
            (out / "images/visible/val").mkdir(parents=True)
            (out / "images/visible/val/v1_000000.jpg").write_bytes(b"x")
            (out / "images/visible/val/v1_000001.jpg").write_bytes(b"x")
            (out / "spla_meta_val.json").write_text(json.dumps(
                {"v1_000000": {"seq": "v1", "frame": 0}, "v1_000001": {"seq": "v1", "frame": 1}}))
            mk.main(["--root", str(root), "--out", str(out), "--splits", "val", '--rgb-size', '1920', '1080'])
            pos = (out / "rgbonly/labels/val/v1_000000.txt").read_text().split()
            neg = (out / "rgbonly/labels/val/v1_000001.txt").read_text()
            self.assertEqual(pos[0], "0")
            for got, want in zip(map(float, pos[1:]), [0.55, 0.55, 0.1, 0.1]):
                self.assertAlmostEqual(got, want, places=5)
            self.assertEqual(neg, "")
            img_dir = out / "rgbonly/images/val"
            # a folder symlink would resolve to images/visible/val and pick up the IR-box labels
            self.assertFalse(img_dir.is_symlink())
            self.assertEqual(img_dir.resolve(), img_dir.absolute().resolve())
            self.assertTrue((img_dir / "v1_000000.jpg").exists())
            self.assertIn("images/val", (out / "antiuav_rgbonly.yaml").read_text())

    def test_missing_source_image_is_an_error(self):
        mk = load("make_rgb_only")
        with tempfile.TemporaryDirectory() as tmp:
            root, out = Path(tmp) / "raw", Path(tmp) / "yolo"
            write_seq(root, "val", "v1", [None], [None])
            (out / "images/visible/val").mkdir(parents=True)
            (out / "spla_meta_val.json").write_text(json.dumps({"v1_000000": {"seq": "v1", "frame": 0}}))
            with self.assertRaises((ValueError, FileNotFoundError)):
                mk.main(["--root", str(root), "--out", str(out), "--splits", "val"])


class Complementarity(unittest.TestCase):
    def test_missing_prediction_is_not_silently_skipped(self):
        cp = load('complementarity')
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / 'raw'
            box = [1, 1, 10, 10]
            write_seq(root, 'val', 'present', [box], [box])
            write_seq(root, 'val', 'missing', [box], [box])
            for run in ('ir', 'rgb'):
                d = Path(tmp) / run; d.mkdir()
                (d / 'present_IR.txt').write_text(json.dumps({'res': [box]}))
                (d / 'run_config.json').write_text(json.dumps({'split': 'val', 'input': 'rgb' if run == 'rgb' else 'auto'}))
            with self.assertRaises(FileNotFoundError):
                cp.main(['--root', str(root), '--ir-run', str(Path(tmp) / 'ir'), '--rgb-run', str(Path(tmp) / 'rgb'),
                         '--boot', '10', '--out', str(Path(tmp) / 'out.json')])

    def test_no_both_positive_frames_is_valid_null_json(self):
        cp = load('complementarity')
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / 'raw'
            write_seq(root, 'val', 'v1', [None], [None])
            for run in ('ir', 'rgb'):
                d = Path(tmp) / run; d.mkdir()
                (d / 'v1_IR.txt').write_text(json.dumps({'res': [[]]}))
                (d / 'run_config.json').write_text(json.dumps({'split': 'val', 'input': 'rgb' if run == 'rgb' else 'auto'}))
            out = Path(tmp) / 'out.json'
            cp.main(['--root', str(root), '--ir-run', str(Path(tmp) / 'ir'), '--rgb-run', str(Path(tmp) / 'rgb'),
                     '--boot', '10', '--out', str(out)])
            result = json.loads(out.read_text())
            self.assertIsNone(result['pooled']['oracle_gain'])
            self.assertIsNone(result['oracle_gain_ci95'])
            self.assertEqual(result['oracle_gain_bootstrap_valid'], 0)

    def test_lengths_must_match(self):
        cp = load("complementarity")
        with self.assertRaises(ValueError):
            cp.frame_success([[]], [1, 1], [[1, 1, 1, 1]] * 2, 0.5)
        import numpy as np
        with self.assertRaises(ValueError):
            cp.counts_for(np.array([True]), np.array([True, False]), np.array([True, True]))

    def test_empty_denominators_are_null(self):
        cp = load("complementarity")
        c = {"n": 0, "both": 0, "ir_only": 0, "rgb_only": 0, "neither": 0}
        s = cp.summarise(c)
        self.assertIsNone(s["oracle_gain"])
        self.assertIsNone(s["rescue_share"])

    def test_counts_and_oracle_gain(self):
        cp = load("complementarity")
        box_i, box_r = [100, 100, 10, 10], [500, 300, 30, 30]
        ir_lab = [box_i] * 4 + [None]
        rgb_lab = [box_r] * 4 + [box_r]
        # frame: 0 both hit, 1 IR only, 2 RGB only, 3 neither, 4 IR has no label (excluded)
        ir_pred = [box_i, box_i, [], [0, 0, 5, 5], []]
        rgb_pred = [box_r, [], box_r, [], box_r]
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "raw"
            write_seq(root, "val", "v1", ir_lab, rgb_lab)
            for run, pred in (("ir", ir_pred), ("rgb", rgb_pred)):
                (Path(tmp) / run).mkdir()
                (Path(tmp) / run / "v1_IR.txt").write_text(json.dumps({"res": pred}))
                (Path(tmp) / run / 'run_config.json').write_text(json.dumps(
                    {'split': 'val', 'input': 'rgb' if run == 'rgb' else 'auto'}))
            outp = Path(tmp) / "c.json"
            cp.main(["--root", str(root), "--split", "val", "--ir-run", str(Path(tmp) / "ir"),
                     "--rgb-run", str(Path(tmp) / "rgb"), "--boot", "50", "--out", str(outp)])
            rep = json.loads(outp.read_text())
        v = rep["per_video"]["v1"]
        self.assertEqual((v["n"], v["both"], v["ir_only"], v["rgb_only"], v["neither"]), (4, 1, 1, 1, 1))
        self.assertAlmostEqual(rep["pooled"]["ir_success"], 0.5)
        self.assertAlmostEqual(rep["pooled"]["oracle_gain"], 0.25)
        self.assertAlmostEqual(rep["pooled"]["rescue_share"], 0.5)


class InputAndCoordinates(unittest.TestCase):
    def test_defaults_and_blanking(self):
        import numpy as np
        runner = load('run_sequences')
        rgb = np.arange(12, dtype=np.uint8).reshape(2, 2, 3)
        ir = np.arange(4, dtype=np.uint8).reshape(2, 2)
        np.testing.assert_array_equal(runner.model_input(rgb, ir, 4), np.concatenate([rgb, ir[..., None]], -1))
        np.testing.assert_array_equal(runner.model_input(rgb, ir, 3), np.repeat(ir[..., None], 3, -1))
        x = runner.model_input(rgb, ir, 4, blank='rgb')
        self.assertTrue((x[..., :3] == 114).all())
        np.testing.assert_array_equal(x[..., 3], ir)
        x = runner.model_input(rgb, ir, 4, blank='ir')
        self.assertTrue((x[..., 3] == 114).all())
        np.testing.assert_array_equal(x[..., :3], rgb)
        np.testing.assert_array_equal(runner.model_input(rgb, ir, 3, input_mode='rgb'), rgb)
        with self.assertRaises(ValueError):
            runner.model_input(rgb, ir, 4, input_mode='rgb')

    def test_rgb_output_uses_original_anisotropic_dimensions(self):
        import numpy as np
        runner = load('run_sequences')
        # 640x512 IR is padded by 64 on the 640 canvas; original RGB is 1920x1080.
        lm = np.array([[640, 0, 0], [0, 512, 64], [0, 0, 1]])
        box = [100, 164, 120, 184]
        np.testing.assert_allclose(runner.frame_box(box, lm, 640, (640, 512)), [100, 100, 20, 20])
        np.testing.assert_allclose(runner.frame_box(box, lm, 640, (1920, 1080)), [300, 210.9375, 60, 42.1875])


if __name__ == "__main__":
    unittest.main()
