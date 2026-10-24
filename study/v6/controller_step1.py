"""One-shot full validation ablations, strict coverage audit and original scoring."""
import datetime
import fcntl
import json
import os
import subprocess
import sys
import time
import traceback
from runtime import V, V3, R, PY, OLD, ENV, inference

lock = (V / 'step1.lock').open('w')
fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
assert not (V / 'step1_status.json').exists(), 'Already launched; inspect before restarting'
assert (V / 'records/real_runtime_validation.json').exists()
state = {'phase': 'ablation_inference', 'pid': os.getpid(), 'jobs': {}, 'training_launched': False}
def save():
    state['updated_at'] = datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat()
    p = V / 'step1_status.tmp'
    p.write_text(json.dumps(state, indent=2)); p.replace(V / 'step1_status.json')
def spawn(name, cmd):
    log = (V / (name + '.log')).open('w')
    proc = subprocess.Popen([str(x) for x in cmd], env=ENV, cwd=V, stdout=log,
                            stderr=subprocess.STDOUT, start_new_session=True)
    state['jobs'][name] = {'state': 'running', 'pid': proc.pid, 'command': [str(x) for x in cmd], 'started': time.time()}
    save()
    return name, proc, log
def finish(job):
    name, proc, log = job
    rc = proc.wait(); log.close()
    state['jobs'][name].update(state='complete' if rc == 0 else 'failed', returncode=rc, finished=time.time())
    save(); assert rc == 0, f'{name} failed'
try:
    jobs = [spawn('A1_blank' + m + '_s0', inference('A1_blank' + m + '_s0', OLD / 'A1_s0/weights/best.pt', ['--blank', m]))
            for m in ('rgb', 'ir')]
    for job in jobs:
        finish(job)
    sys.path.insert(0, str(V3))
    from audit_predictions import audit
    dirs = [V / 'results' / ('A1_blank' + m + '_s0') for m in ('rgb', 'ir')]
    coverage = audit(dirs, V / 'reports/coverage_ablation.json')
    assert all(r['n_sequences'] == 67 and r['frames'] == 61999 and not r['length_mismatches'] for r in coverage.values())
    state['phase'] = 'ablation_scoring'; save()
    cmd = [PY, V / 'kit/tools/score_runs.py', '--root', R, '--split', 'val', '--baseline', 'A1',
           '--run', 'A1=' + str(V3 / 'results/val/A1_s0'),
           '--run', 'A1_blankrgb=' + str(dirs[0]), '--run', 'A1_blankir=' + str(dirs[1]),
           '--boot', '10000', '--out', V / 'reports/ablation_scores.json',
           '--per-seq', V / 'reports/ablation_per_sequence.csv']
    finish(spawn('score_ablation', cmd))
    state['phase'] = 'complete'; save()
except BaseException:
    state['phase'] = 'failed'; state['error'] = traceback.format_exc(); save(); raise
