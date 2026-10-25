"""Select a lag on earlier trajectory data and evaluate it on a later block."""
import json
import os
import sys
import numpy as np
from runtime import V, R, IR, RGB, ENV
os.environ.update(ENV)
sys.path[:0] = [str(V / 'kit'), str(V / 'kit/tools')]
from decision_fusion import _read_json, run_sequences
from sync_audit import centres_from_labels, centres_from_dets
from spla.antiuav_io import find_files, read_labels
from spla.geometry import fit_transform

def evaluate(a, b):
    if len(a) != len(b): raise ValueError('Unequal trajectory lengths')
    n = len(a); idx = np.arange(15, n-15)
    valid = np.isfinite(a[idx]).all(axis=1)
    for k in range(-15, 16): valid &= np.isfinite(b[idx+k]).all(axis=1)
    # A 15-frame gap on each side prevents positive lags from reading the held-out block.
    idx = idx[valid]; cutoff = (2*n)//3
    tr = idx[idx < cutoff-15]; te = idx[idx >= cutoff+15]
    row = {'frames': n, 'train_pairs': len(tr), 'test_pairs': len(te)}
    if min(len(tr), len(te)) < 40: return dict(row, status='insufficient_common_support')
    fits = {}; scores = {}
    def residual(fit, indices, lag):
        raw = b[indices+lag] - (a[indices] @ fit.theta[:, :2].T + fit.theta[:, 2])
        r = np.linalg.solve(fit.theta[:, :2], raw.T).T * [640, 512]
        return float(np.median(np.linalg.norm(r, axis=1)))
    for k in range(-15, 16):
        try:
            fit = fit_transform(a[tr], b[tr+k], 'scale_translation')
            if np.linalg.cond(fit.theta[:, :2]) > 1e8: continue
            scores[k] = residual(fit, tr, k); fits[k] = fit
        except np.linalg.LinAlgError: continue
    if 0 not in fits: return dict(row, status='singular_spatial_fit')
    best = min(scores, key=lambda k: (scores[k], abs(k)))
    rz = residual(fits[0], te, 0); rb = residual(fits[best], te, best)
    return dict(row, status='estimated', selected_training_lag=best, training_residual_zero_px=scores[0],
                training_residual_selected_px=scores[best], heldout_residual_zero_px=rz,
                heldout_residual_selected_px=rb, heldout_candidate=bool(best != 0 and rz-rb > 1 and rz > 1.5*rb))

if __name__ == '__main__':
    report = {'settings': 'Fixed common support; first two thirds select lag and fit transforms, final third evaluates fixed choices; lags -15..15; min 40 pairs per block',
              'scope': 'Diagnostic candidates, not proof of dropped frames or sensor timing', 'sources': {}}
    for source in ('predictions', 'labels'):
        per = {}
        for seq in run_sequences(IR):
            if source == 'predictions':
                a = centres_from_dets(_read_json(IR / (seq + '_dets.json')), [640, 512], .5)
                b = centres_from_dets(_read_json(RGB / (seq + '_dets.json')), [1920, 1080], .5)
            else:
                fs = find_files(R / 'val' / seq)
                ei, bi = read_labels(fs['ir_label']); er, br = read_labels(fs['rgb_label'])
                a = centres_from_labels(ei, bi, [640, 512]); b = centres_from_labels(er, br, [1920, 1080])
            per[seq] = evaluate(a, b)
        report['sources'][source] = {'per_video': per, 'candidates': [k for k, v in per.items() if v.get('heldout_candidate')]}
    (V / 'reports/sync_heldout.json').write_text(json.dumps(report, indent=2, allow_nan=False))
    print({k: len(v['candidates']) for k, v in report['sources'].items()})
