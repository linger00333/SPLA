"""Check saved RGB pixel boxes against repeated real GPU canvas detections."""
import json
import os
import sys
from runtime import V, R, OLD, ENV

os.environ.update(ENV)
sys.path[:0] = [str(V / 'kit'), str(V / 'framework'), str(V / 'kit/tools')]
import cv2
import numpy as np
import torch
from spla.antiuav_io import find_files
from spla.checkpoints import load_inference_model
from run_sequences import letterbox
from ultralytics.utils.ops import non_max_suppression

directory = V / 'results/runtime_rgb_input'
source = next(directory.glob('*_dets.json'))
seq = source.name.removesuffix('_dets.json')
saved = json.loads(source.read_text())
fs = find_files(R / 'val' / seq)
ci, cr = cv2.VideoCapture(str(fs['ir_video'])), cv2.VideoCapture(str(fs['rgb_video']))
wi, hi = int(ci.get(cv2.CAP_PROP_FRAME_WIDTH)), int(ci.get(cv2.CAP_PROP_FRAME_HEIGHT))
wv, hv = int(cr.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cr.get(cv2.CAP_PROP_FRAME_HEIGHT))
ci.release()
model = load_inference_model(OLD / 'A0_s0/weights/best.pt', torch.device('cuda:1'))
count = 0; maximum = 0
for t, stored in enumerate(saved):
    ok, frame = cr.read(); assert ok
    frame = cv2.resize(frame, (wi, hi), interpolation=cv2.INTER_AREA)
    rgb, lm = letterbox(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), 640)
    x = torch.from_numpy(rgb.transpose(2, 0, 1)[None].astype(np.float32) / 255).to('cuda:1')
    with torch.no_grad():
        det = non_max_suppression(model(x), conf_thres=.001, iou_thres=.6, max_det=10)[0].cpu().numpy()
    assert len(det) == len(stored)
    for canvas, original in zip(det, stored):
        x0, y0, w, h, score = original
        # Independently reconstruct canvas corners from saved original RGB pixels.
        back = np.array([x0 / wv * lm[0, 0] + lm[0, 2], y0 / hv * lm[1, 1] + lm[1, 2],
                         (x0 + w) / wv * lm[0, 0] + lm[0, 2], (y0 + h) / hv * lm[1, 1] + lm[1, 2]])
        np.testing.assert_allclose(back, canvas[:4], rtol=0, atol=1e-4)
        assert abs(score - float(canvas[4])) < 1e-7
        maximum = max(maximum, float(np.abs(back - canvas[:4]).max())); count += 1
cr.release()
assert count > 0
report = {'sequence': seq, 'real_frames': len(saved), 'candidate_boxes_checked': count,
          'IR_dimensions': [wi, hi], 'original_RGB_dimensions': [wv, hv],
          'maximum_canvas_roundtrip_error_px': maximum, 'device': 'physical GPU 1',
          'note': 'Old IR weights used solely to test RGB plumbing and coordinate conversion; not RGB baseline performance'}
(V / 'records/real_rgb_coordinate_audit.json').write_text(json.dumps(report, indent=2))
print(json.dumps(report))
