"""Run the single authorized RGB baseline only after step 1 and data gates pass."""
import datetime
import fcntl
import hashlib
import json
import os
import subprocess
import sys
import time
import traceback
from runtime import V, V3, P, R, PY, ENV, inference

lock = (V / 'step2.lock').open('w')
fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
assert not (V / 'step2_status.json').exists(), 'Already launched; do not duplicate training'
state = {'phase': 'waiting_for_step1', 'pid': os.getpid(), 'jobs': {}, 'training_launched': False,
         'authorized_training_runs': 1, 'test_evaluated': False, 'recurring_automation': False}
def save():
    state['updated_at'] = datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat()
    p = V / 'step2_status.tmp'; p.write_text(json.dumps(state, indent=2)); p.replace(V / 'step2_status.json')
def run(name, cmd, cwd=V):
    cmd = [str(x) for x in cmd]
    with (V / (name + '.log')).open('w') as log:
        proc = subprocess.Popen(cmd, cwd=cwd, env=ENV, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        state['jobs'][name] = {'state': 'running', 'pid': proc.pid, 'command': cmd, 'started': time.time()}; save()
        rc = proc.wait()
    state['jobs'][name].update(state='complete' if rc == 0 else 'failed', returncode=rc, finished=time.time()); save()
    assert rc == 0, f'{name} failed; no automatic retry'
save()
try:
    start = time.time()
    while True:
        step1 = json.loads((V / 'step1_status.json').read_text())
        assert step1['phase'] != 'failed', 'Step 1 failed; training not authorized past the gate'
        if step1['phase'] == 'complete':
            break
        assert time.time() - start < 3 * 3600, 'Step 1 exceeded the wait window; inspect before relaunch'
        time.sleep(20)
    data_audit = json.loads((V / 'reports/dataset_audit.json').read_text())
    visual_gate = json.loads((V / 'records/visual_review.json').read_text())
    assert visual_gate['passed'] and visual_gate['samples_inspected'] == 4
    assert data_audit['shared_source_labels_unchanged'] and not data_audit['test_processed']
    assert [data_audit['splits'][s]['images'] for s in ('train', 'val')] == [29708, 12158]
    for s in ('train', 'val'):
        assert data_audit['splits'][s]['all_actual_loader_label_paths_correct']
        assert data_audit['splits'][s]['all_loader_boxes_match_original_rgb_normalization']
    manifest = json.loads((V / 'records/frozen_code.json').read_text())
    for rel, expected in manifest['sha256'].items():
        assert hashlib.sha256((V / rel).read_bytes()).hexdigest() == expected, rel
    state['phase'] = 'rgb_training'; state['training_launched'] = True; save()
    cmd = [PY, V / 'kit/tools/train_spla.py', '--stage', '0', '--data', V / 'data/antiuav_rgbonly.yaml',
           '--size', 'n', '--seed', '0', '--imgsz', '640', '--batch', '16', '--workers', '4',
           '--epochs', '100', '--device', '1', '--project', V / 'runs', '--name', 'A0rgb_s0']
    run('train_A0rgb_s0', cmd)
    import yaml
    old_args = yaml.safe_load((P / 'experiments/v1_spla_zoom_20261023/framework/runs/detect/A0_s0/args.yaml').read_text())
    args = yaml.safe_load((V / 'runs/A0rgb_s0/args.yaml').read_text())
    keys = ['model', 'epochs', 'batch', 'imgsz', 'workers', 'seed', 'deterministic', 'optimizer', 'pretrained',
            'lr0', 'lrf', 'warmup_epochs', 'cos_lr', 'mosaic', 'close_mosaic', 'mixup', 'scale', 'translate',
            'fliplr', 'flipud', 'hsv_h', 'hsv_s', 'hsv_v', 'fraction', 'channels', 'use_simotm', 'amp']
    differences = {k: [old_args[k], args[k]] for k in keys if old_args[k] != args[k]}
    assert not differences, differences
    weight = V / 'runs/A0rgb_s0/weights/best.pt'
    (V / 'records/training_recipe_audit.json').write_text(json.dumps({'all_selected_recipe_fields_match_old_A0': True,
         'compared_fields': keys, 'scratch_architecture_yaml': args['model'], 'best_checkpoint_sha256': hashlib.sha256(weight.read_bytes()).hexdigest(),
         'changed_fields': 'RGB images and own labels, independent output directory; budget and scratch recipe held fixed'}, indent=2))
    state['phase'] = 'rgb_validation'; save()
    run('val_A0rgb_s0', inference('A0rgb_s0', weight, ['--input', 'rgb']))
    sys.path.insert(0, str(V3))
    from audit_predictions import audit
    cov = audit([V / 'results/A0rgb_s0'], V / 'reports/coverage_rgb.json')
    assert all(r['n_sequences'] == 67 and r['frames'] == 61999 and not r['length_mismatches'] for r in cov.values())
    state['phase'] = 'complementarity'; save()
    run('complementarity', [PY, V / 'kit/tools/complementarity.py', '--root', R, '--split', 'val',
                           '--ir-run', V3 / 'results/val/A0_s0', '--rgb-run', V / 'results/A0rgb_s0',
                           '--iou', '0.5', '--boot', '10000', '--out', V / 'reports/complementarity_val.json'])
    state['phase'] = 'complete_report_ready'; save()
except BaseException:
    state['phase'] = 'failed'; state['error'] = traceback.format_exc(); save(); raise
