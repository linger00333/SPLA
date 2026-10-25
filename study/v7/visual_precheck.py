"""Inspect the specified video before registering the fusion scoring protocol."""
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace
from runtime import V, IR, RGB, R, ENV
os.environ.update(ENV)
sys.path[:0] = [str(V / 'kit'), str(V / 'kit/tools')]
import cv2
import numpy as np
from spla.antiuav_io import find_files, read_labels
from decision_fusion import load_pair, train_mean_theta, fuse_video

seq = '20190925_213001_1_1'
ir_res, di, dr = load_pair(IR, RGB, seq)
train = json.loads((V / 'inputs/seq_theta_train.json').read_text())
theta = train_mean_theta(train)
args = SimpleNamespace(map='train_mean', size='mapped', ir_conf=.5, rgb_conf=.5, pair_conf=.5, window=30, online_k=5)
fused, _, log, _ = fuse_video(seq, ir_res, di, dr, theta, args, (640, 512), (1920, 1080))
replaced = [r[1] for r in log if r[4] == 'replace']
assert replaced
selected = sorted(set([0, len(ir_res)-1] + [replaced[int(i)] for i in np.linspace(0, len(replaced)-1, 8)]))
# The chosen frames depend only on predictions; only this explicitly requested
# single-video precheck reads labels before full protocol registration.
fs = find_files(R / 'val' / seq)
ei, bi = read_labels(fs['ir_label']); er, br = read_labels(fs['rgb_label'])
assert len(ei) == len(er) == len(ir_res)
out = V / 'private_visuals'; out.mkdir(exist_ok=True)
caps = [cv2.VideoCapture(str(fs[k])) for k in ('ir_video', 'rgb_video')]
def box(image, b, color, thickness=2):
    if len(b) >= 4:
        x, y, w, h = map(float, b[:4])
        cv2.rectangle(image, (int(x), int(y)), (int(x+w), int(y+h)), color, thickness)
rows = []
for t in selected:
    frames = []
    for cap in caps:
        cap.set(cv2.CAP_PROP_POS_FRAMES, t); ok, f = cap.read(); assert ok
        frames.append(f)
    for m, (e, b, pred) in enumerate(((ei, bi, ir_res), (er, br, [d[0][:4] if d and d[0][4] >= .25 else [] for d in dr]))):
        if e[t]: box(frames[m], b[t], (0, 255, 0))
        box(frames[m], pred[t], (255, 150, 0))
    if t in replaced: box(frames[0], fused[t], (0, 0, 255))
    for m in range(2):
        frames[m] = cv2.resize(frames[m], (640, 512), interpolation=cv2.INTER_AREA)
        title = f'{seq} frame={t} {"IR" if m == 0 else "RGB"} label={int((ei if m == 0 else er)[t])}'
        cv2.putText(frames[m], title, (5, 22), cv2.FONT_HERSHEY_SIMPLEX, .43, (0, 0, 255), 1)
    cv2.imwrite(str(out / f'{seq}_{t:06d}.jpg'), np.hstack(frames))
    rows.append({'frame': t, 'IR_label_present': int(ei[t]), 'RGB_label_present': int(er[t]),
                 'primary_replacement': t in replaced, 'original_IR_output_present': len(ir_res[t]) == 4})
for cap in caps: cap.release()
rep = {'sequence': seq, 'frames': len(ir_res), 'primary_replacements': len(replaced),
       'selection': 'Eight evenly spaced replacement indices plus first/last frame; no score-based selection',
       'samples': rows, 'publication': 'Images remain private; no protocol or thresholds tuned from this video',
       'legend': 'Green: own modality GT; blue: original detector; red on IR: mapped RGB replacement'}
(V / 'records/visual_precheck.json').write_text(json.dumps(rep, indent=2))
print(json.dumps(rep, indent=2))
