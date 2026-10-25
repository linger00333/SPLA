"""One CPU-only validation pass; no training, test, tuning or automatic retry."""
import datetime
import fcntl
import hashlib
import json
import os
import subprocess
import sys
import time
import traceback
from runtime import V, R, IR, RGB, PY, ENV
lock = (V / 'controller.lock').open('w'); fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
assert not (V / 'status.json').exists(), 'Already launched; inspect instead of duplicating'
state = {'phase': 'freezing_inputs', 'pid': os.getpid(), 'jobs': {}, 'training': False, 'test': False, 'GPU_used': False}
def save():
    state['updated_at'] = datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat()
    p = V / 'status.tmp'; p.write_text(json.dumps(state, indent=2)); p.replace(V / 'status.json')
def run(name, cmd):
    cmd = [str(x) for x in cmd]
    with (V / (name + '.log')).open('w') as log:
        p = subprocess.Popen(cmd, cwd=V, env=ENV, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        state['jobs'][name] = {'state': 'running', 'pid': p.pid, 'command': cmd, 'started': time.time()}; save()
        rc = p.wait()
    state['jobs'][name].update(state='complete' if rc == 0 else 'failed', returncode=rc, finished=time.time()); save()
    assert rc == 0, name + ' failed; no retry'
save()
try:
    run('freeze_and_audit', [PY, V / 'freeze_and_audit.py'])
    sys.path[:0] = [str(V / 'kit'), str(V / 'kit/tools')]
    from subset_bootstrap import load, bootstrap
    old_csv = V.parent / 'v6_fusion_diagnostics_20261024/reports/ablation_per_sequence.csv'
    old_json = json.loads((V.parent / 'v6_fusion_diagnostics_20261024/reports/ablation_scores.json').read_text())
    reproduced = bootstrap(load(old_csv, 'A1_blankrgb', 'A1'), 10000, 0)
    expected = old_json['vs_baseline']['A1_blankrgb-A1']['SA']
    assert reproduced['ci95'] == expected['ci95'] and reproduced['boot_mean'] == expected['mean']
    (V / 'records/v6_SA_bootstrap_reproduction.json').write_text(json.dumps({'exact_first_comparison_reproduction': True, 'result': reproduced}, indent=2))
    state['phase'] = 'fusion_and_sync_audit'; save()
    for name, mode, size in (('P', 'train_mean', 'mapped'), ('S', 'online', 'ir_recent'), ('R', 'seq', 'mapped')):
        cmd = [PY, V / 'kit/tools/decision_fusion.py', '--ir-run', IR, '--rgb-run', RGB,
               '--train-theta', V / 'inputs/seq_theta_train.json', '--map', mode, '--size', size,
               '--ir-conf', '.5', '--rgb-conf', '.5', '--pair-conf', '.5', '--window', '30', '--online-k', '5',
               '--out', V / 'results' / name]
        if mode == 'seq': cmd += ['--seq-theta', V / 'inputs/seq_theta_val.json']
        run('fusion_' + name, cmd)
        summary = json.loads((V / 'results' / name / 'fusion_summary.json').read_text())
        assert summary['videos'] == 67 and summary['pooled']['frames'] == 61999
        supplied = json.loads((V / 'supplied_prereg' / f'fusion_summary_{name}_{mode}.json').read_text())
        def equivalent(a, b):
            if isinstance(a, dict):
                assert a.keys() == b.keys()
                for k in a: equivalent(a[k], b[k])
            elif isinstance(a, float):
                assert b is not None and abs(a-b) <= 1e-12
            else: assert a == b
        equivalent(summary['pooled'], supplied['pooled']); equivalent(summary['per_video'], supplied['per_video'])
    for label, flags in (('predictions', ['--ir-run', IR, '--rgb-run', RGB]), ('labels', ['--root', R])):
        for suffix, extra in (('whole', []), ('windows200', ['--window', '200']), ('common', ['--support', 'common'])):
            run('sync_' + label + '_' + suffix, [PY, V / 'kit/tools/sync_audit.py', *flags, '--split', 'val', *extra,
                '--out', V / 'reports' / ('sync_' + label + '_' + suffix + '.json')])
    run('sync_heldout', [PY, V / 'heldout_sync.py'])
    # All final inputs must remain byte-identical to the registration manifest.
    frozen = json.loads((V / 'records/frozen_manifest.json').read_text())
    for rel, sha in frozen['code_sha256'].items(): assert hashlib.sha256((V / rel).read_bytes()).hexdigest() == sha, rel
    state['phase'] = 'fusion_scoring'; save()
    run('score_fusion', [PY, V / 'kit/tools/score_runs.py', '--root', R, '--split', 'val', '--require-dets',
         '--run', 'A0=' + str(IR), '--run', 'P=' + str(V / 'results/P'), '--run', 'S=' + str(V / 'results/S'),
         '--run', 'R=' + str(V / 'results/R'), '--baseline', 'A0', '--boot', '10000', '--bootstrap-mode', 'per_comparison',
         '--per-seq', V / 'reports/fusion_val_per_seq.csv', '--out', V / 'reports/fusion_val.json'])
    from spla.antiuav_io import KNOWN_LABEL_ISSUES
    raw_names = {p.name for p in (R / 'val').iterdir() if p.is_dir()}
    issue_names = sorted(raw_names & set(KNOWN_LABEL_ISSUES))
    (V / 'records/known_issue_exclusions.json').write_text(json.dumps(issue_names))
    for name in ('P', 'S', 'R'):
        run('breakdown_' + name, [PY, V / 'kit/tools/fusion_breakdown.py', '--root', R, '--split', 'val',
            '--ir-run', IR, '--fused', V / 'results' / name, '--out', V / 'reports' / ('breakdown_' + name + '.json')])
        for label, extra in (('without_213001', ['--exclude', '20190925_213001_1_1']),
                             ('without_known_issues', ['--exclude-file', V / 'records/known_issue_exclusions.json']),
                             ('without_prediction_sync_candidates', ['--exclude-file', V / 'supplied_prereg/sync_flagged_predictions_val.json'])):
            run(name + '_' + label, [PY, V / 'kit/tools/subset_bootstrap.py', '--per-seq', V / 'reports/fusion_val_per_seq.csv',
                 '--a', name, '--b', 'A0', '--label', label, *extra, '--boot', '10000', '--seed', '0',
                 '--out', V / 'reports' / (name + '_' + label + '.json')])
    state['phase'] = 'complete_report_ready'; save()
except BaseException:
    state['phase'] = 'failed'; state['error'] = traceback.format_exc(); save(); raise
