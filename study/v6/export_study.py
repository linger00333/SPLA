"""Export completed evidence without raw dataset content or machine metadata."""
from pathlib import Path
import datetime
import gzip
import hashlib
import json
import math
import os
import sys
import zipfile
from runtime import V, V3, P, ENV

os.environ.update(ENV)
sys.path[:0] = [str(V / 'kit'), str(V / 'framework')]
OUT = V / 'publication_export'
OUT.mkdir(exist_ok=True)
aliases = [(str(V), 'experiments/v6'), (str(V3), 'experiments/v3'),
           (str(V.parent / 'v1_spla_zoom_20261023'), 'experiments/legacy'),
           (str(P / 'spla_kit_v1/datasets'), 'data/raw'), (str(P / 'antiuav_yolo'), 'data/converted'),
           (str(P), 'workspace'), (str(Path.home()), 'user-home')]
def portable(s):
    for old, new in aliases:
        s = s.replace(old, new)
    assert s.isascii(), repr(s)
    return s
def clean(x):
    if isinstance(x, dict):
        return {portable(str(k)): clean(v) for k, v in x.items()}
    if isinstance(x, list):
        return [clean(v) for v in x]
    if isinstance(x, str):
        return portable(x)
    if isinstance(x, float) and not math.isfinite(x):
        return None
    return x
def export(src, rel):
    dst = OUT / rel; dst.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(clean(json.loads(src.read_text())), indent=2, allow_nan=False) + '\n' if src.suffix == '.json' else portable(src.read_text())
    dst.write_text(text, encoding='ascii', newline='\n')

states = {s: json.loads((V / f'{s}_status.json').read_text()) for s in ('step1', 'step2')}
for name in states:
    export(V / f'{name}_status.json', Path('artifacts/v6') / f'{name}_status_snapshot.json')
for src in (V / 'reports').iterdir():
    if src.suffix in ('.json', '.csv'):
        export(src, Path('artifacts/v6/reports') / src.name)
for name in ('input_manifest.json', 'frozen_code.json', 'real_runtime_validation.json', 'real_rgb_coordinate_audit.json',
             'runtime_source_audit.json', 'visual_review.json', 'training_recipe_audit.json'):
    src = V / 'records' / name
    if src.exists():
        export(src, Path('artifacts/v6/records') / name)
for name in ('A1_blankrgb_s0', 'A1_blankir_s0', 'A0rgb_s0'):
    key = name if name.startswith('A1') else 'val_A0rgb_s0'
    state = states['step1' if name.startswith('A1') else 'step2']
    if state['jobs'].get(key, {}).get('state') != 'complete':
        continue
    for src in (V / 'results' / name).iterdir():
        if src.suffix not in ('.json', '.csv', '.txt'):
            continue
        text = json.dumps(clean(json.loads(src.read_text())), allow_nan=False) if src.suffix == '.json' else portable(src.read_text())
        dst = OUT / 'artifacts/predictions/v6' / name / (src.name + '.gz')
        dst.parent.mkdir(parents=True, exist_ok=True)
        with dst.open('wb') as f:
            with gzip.GzipFile(filename='', fileobj=f, mode='wb', mtime=0) as z:
                z.write(text.encode('ascii'))
train_done = states['step2']['jobs'].get('train_A0rgb_s0', {}).get('state') == 'complete'
if train_done:
    import yaml
    export(V / 'runs/A0rgb_s0/results.csv', Path('artifacts/v6/training/A0rgb_s0/results.csv'))
    args = yaml.safe_load((V / 'runs/A0rgb_s0/args.yaml').read_text())
    dst = OUT / 'artifacts/v6/training/A0rgb_s0/args.json'
    dst.write_text(json.dumps(clean(args), indent=2) + '\n', encoding='ascii')
    import torch
    from safetensors.torch import save_file
    from spla.checkpoints import load_inference_model
    weight = V / 'runs/A0rgb_s0/weights/best.pt'
    model = load_inference_model(weight, torch.device('cpu'))
    model.float().eval(); torch.set_num_threads(2)
    directory = OUT / 'models/v6/A0rgb_s0'; directory.mkdir(parents=True, exist_ok=True)
    config = dict(model.yaml); config.pop('yaml_file', None)
    (directory / 'config.json').write_text(json.dumps(clean(config), indent=2) + '\n', encoding='ascii')
    state = {k: (v.half() if v.is_floating_point() else v).detach().cpu().contiguous().clone() for k, v in model.state_dict().items()}
    sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    save_file(state, str(directory / 'model.safetensors'), metadata={'model_id': 'A0rgb_s0', 'source_checkpoint_sha256': sha(weight),
              'precision': 'FP16 storage; FP32 inference', 'modality': 'RGB', 'training_epochs': '100', 'seed': '0'})
    restored = load_inference_model(directory / 'model.safetensors', torch.device('cpu'))
    x = torch.rand((1, 3, 640, 640), generator=torch.Generator().manual_seed(19))
    with torch.no_grad():
        a, b = model(x)[0], restored(x)[0]
    torch.testing.assert_close(a, b, rtol=0, atol=0)
    row = {'source_checkpoint_sha256': sha(weight), 'tensor_sha256': sha(directory / 'model.safetensors'),
           'config_sha256': sha(directory / 'config.json'), 'channels': 3, 'seed': 0, 'epochs': 100,
           'initialization': 'scratch', 'output_coordinates': 'RGB_pixels', 'export_parity': 'Exact CPU output equality on seeded tensor'}
    (OUT / 'artifacts/v6/records/rgb_model_export.json').write_text(json.dumps(row, indent=2) + '\n', encoding='ascii')
snapshot = {'exported_at': datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),
            'step1_phase': states['step1']['phase'], 'step2_phase': states['step2']['phase'],
            'test_evaluated': False, 'recurring_automation': False, 'training_completed': train_done}
(OUT / 'artifacts/v6/export_snapshot.json').write_text(json.dumps(snapshot, indent=2) + '\n', encoding='ascii')
with zipfile.ZipFile(V / 'publication_export.zip', 'w', zipfile.ZIP_DEFLATED) as z:
    for p in sorted(OUT.rglob('*')):
        if p.is_file():
            z.write(p, p.relative_to(OUT).as_posix())
print(json.dumps(snapshot))
print('Export bytes:', (V / 'publication_export.zip').stat().st_size)
