"""Explanatory geometry review; post-registration, never used to tune the gate."""
import json
import os
import sys
import numpy as np
from runtime import V, R, ENV
os.environ.update(ENV)
sys.path[:0] = [str(V / 'kit'), str(V / 'kit/tools')]
from spla.antiuav_io import find_files, read_labels, KNOWN_LABEL_ISSUES
from decision_fusion import rgb_box_to_ir, iou_xywh
table = json.loads((V / 'inputs/seq_theta_val.json').read_text())
identity = []; heights = []; widths = []; ious = []; per = {}
for seq, row in sorted(table.items()):
    fs = find_files(R / 'val' / seq)
    ei, bi = read_labels(fs['ir_label']); er, br = read_labels(fs['rgb_label'])
    if len(ei) != len(er): raise ValueError('Unequal label lengths')
    wi = np.asarray(row['ir_wh']); wr = np.asarray(row['rgb_wh'])
    mask = (ei == 1) & (er == 1); ix = np.flatnonzero(mask)
    ci = (bi[mask, :2] + bi[mask, 2:]/2) / wi
    cr = (br[mask, :2] + br[mask, 2:]/2) / wr
    d = np.linalg.norm((ci-cr)*wi, axis=1)
    identity.extend(d.tolist())
    local_iou = []; local_h = []; local_w = []
    for t in ix:
        mapped = rgb_box_to_ir(br[t], row['theta'], wi, wr)
        local_iou.append(iou_xywh(mapped, bi[t]))
        local_h.append(float(bi[t, 3] / mapped[3])); local_w.append(float(bi[t, 2] / mapped[2]))
    heights.extend(local_h); widths.extend(local_w); ious.extend(local_iou)
    per[seq] = {'paired_frames': len(ix), 'identity_center_error_median_IR_px': float(np.median(d)) if len(d) else None,
                'mapped_GT_IoU_median': float(np.median(local_iou)) if local_iou else None,
                'known_label_issue': seq in KNOWN_LABEL_ISSUES}
report = {'scope': 'Post-registration explanatory label-geometry review; not used for thresholds, exclusions or gate. Identity mismatch characterizes the unregistered input; not a causal accuracy contribution.',
          'paired_positive_frames': len(identity),
          'identity_center_error_IR_px': {'median': float(np.median(identity)), 'q25': float(np.percentile(identity, 25)), 'q75': float(np.percentile(identity, 75))},
          'label_derived_sequence_mapping_reference': {'median_GT_IoU': float(np.median(ious)),
             'share_GT_IoU_ge_0_5': float(np.mean(np.asarray(ious) >= .5)),
             'median_IR_height_over_mapped_RGB_height': float(np.median(heights)),
             'median_IR_width_over_mapped_RGB_width': float(np.median(widths))}, 'per_video': per}
(V / 'reports/geometry_review.json').write_text(json.dumps(report, indent=2, allow_nan=False))
print(json.dumps({k: v for k, v in report.items() if k != 'per_video'}, indent=2))
