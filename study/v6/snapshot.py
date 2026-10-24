"""Read actual processes, saved sequence coverage and completed epochs."""
import csv
import datetime
import json
import subprocess
from runtime import V

report = {'observed_at': datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat()}
for name in ('step1', 'step2'):
    path = V / f'{name}_status.json'
    if path.exists():
        s = json.loads(path.read_text())
        active = {}
        for key, row in s['jobs'].items():
            actual = subprocess.run(['ps', '-p', str(row['pid']), '-o', 'pid=,etimes=,pcpu=,stat=,args='], stdout=subprocess.PIPE, text=True).stdout.strip()
            active[key] = {'recorded_state': row['state'], 'actual_process': actual or None}
        controller = subprocess.run(['ps', '-p', str(s['pid']), '-o', 'pid=,etimes=,stat='], stdout=subprocess.PIPE, text=True).stdout.strip()
        report[name] = {'phase': s['phase'], 'controller_pid': s['pid'], 'controller_process': controller or None, 'jobs': active}
saved = {}
for name in ('A1_blankrgb_s0', 'A1_blankir_s0', 'A0rgb_s0'):
    files = list((V / 'results' / name).glob('*_IR.txt'))
    saved[name] = {'videos': len(files), 'frames': sum(len(json.loads(p.read_text())['res']) for p in files)}
report['saved_predictions'] = saved
path = V / 'runs/A0rgb_s0/results.csv'
if path.exists():
    with path.open() as f:
        rows = list(csv.DictReader(f))
    if rows:
        report['rgb_training'] = {'completed_epochs': len(rows), 'latest_epoch': {k.strip(): v.strip() for k, v in rows[-1].items()}}
report['gpu_processes'] = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid,gpu_uuid,used_memory', '--format=csv,noheader'], text=True).strip().splitlines()
report['errors'] = {name: (V / f'{name}_controller.log').read_text()[-1500:] for name in ('step1', 'step2') if (V / f'{name}_controller.log').stat().st_size}
print(json.dumps(report, indent=2))
(V / 'records/latest_snapshot.json').write_text(json.dumps(report, indent=2))
