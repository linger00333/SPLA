"""Export completed v7 evidence and derived predictions, excluding raw data."""
from pathlib import Path
import datetime
import gzip
import json
import math
import zipfile
from runtime import V, P, V3, V6, R
OUT = V / 'publication_export'; OUT.mkdir(exist_ok=True)
aliases = [(str(V), 'experiments/v7'), (str(V3), 'experiments/v3'), (str(V6), 'experiments/v6'),
           (str(R), 'data/raw'), (str(P), 'workspace'), (str(Path.home()), 'user-home')]
def portable(text):
    for a, b in aliases: text = text.replace(a, b)
    assert text.isascii(), repr(text)
    return text
def clean(x):
    if isinstance(x, dict): return {portable(str(k)): clean(v) for k, v in x.items()}
    if isinstance(x, list): return [clean(v) for v in x]
    if isinstance(x, str): return portable(x)
    if isinstance(x, float) and not math.isfinite(x): return None
    return x
def export(src, rel):
    dest = OUT / rel; dest.parent.mkdir(parents=True, exist_ok=True)
    if src.suffix == '.json':
        obj = json.loads(src.read_text())
        if src.name == 'visual_precheck.json':
            obj = {k: v for k, v in obj.items() if k != 'samples'}
            obj.update(inspected_frame_pairs=9, selected_frame_indices=[r['frame'] for r in json.loads(src.read_text())['samples']],
                       findings='Early IR samples are strongly blurred; modality label presence differs. Original labels remain unchanged; frame-level label states and images stay private.')
        text = json.dumps(clean(obj), indent=2, allow_nan=False) + '\n'
    else:
        text = portable(src.read_text())
    dest.write_text(text, encoding='ascii', newline='\n')
state = json.loads((V / 'status.json').read_text())
assert state['phase'] == 'complete_report_ready' and all(x['state'] == 'complete' for x in state['jobs'].values())
export(V / 'status.json', Path('artifacts/v7/status_snapshot.json'))
for folder in ('reports', 'records'):
    for src in (V / folder).iterdir():
        if src.suffix in ('.json', '.csv', '.log'): export(src, Path('artifacts/v7') / folder / src.name)
for src in V.glob('*.log'):
    if src.stat().st_size: export(src, Path('artifacts/v7/logs') / src.name)
for name in ('P', 'S', 'R'):
    for src in (V / 'results' / name).iterdir():
        if src.suffix not in ('.json', '.csv', '.txt'): continue
        text = json.dumps(clean(json.loads(src.read_text())), allow_nan=False) if src.suffix == '.json' else portable(src.read_text())
        dst = OUT / 'artifacts/predictions/v7' / name / (src.name + '.gz'); dst.parent.mkdir(parents=True, exist_ok=True)
        with dst.open('wb') as f:
            with gzip.GzipFile(filename='', fileobj=f, mode='wb', mtime=0) as z: z.write(text.encode('ascii'))
snapshot = {'exported_at': datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),
            'phase': state['phase'], 'prediction_sets': ['P', 'S', 'R'], 'no_raw_dataset_content': True}
(OUT / 'artifacts/v7/export_snapshot.json').write_text(json.dumps(snapshot, indent=2) + '\n', encoding='ascii')
with zipfile.ZipFile(V / 'publication_export.zip', 'w', zipfile.ZIP_DEFLATED) as z:
    for p in sorted(OUT.rglob('*')):
        if p.is_file(): z.write(p, p.relative_to(OUT).as_posix())
print('Completed v7 export bytes:', (V / 'publication_export.zip').stat().st_size)
