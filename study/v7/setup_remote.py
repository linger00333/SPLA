"""Provision only the independent CPU v7 audit and visual precheck."""
from pathlib import Path
import hashlib
import json
import shutil
import zipfile
V = Path(__file__).resolve().parent
assert not (V / 'kit').exists()
with zipfile.ZipFile(V / 'source_bundle.zip') as z:
    assert all(not Path(n).is_absolute() and '..' not in Path(n).parts for n in z.namelist())
    z.extractall(V)
for d in ('records', 'reports', 'inputs', 'results', 'private_visuals'): (V / d).mkdir(exist_ok=True)
V3 = V.parent / 'v3_spla_controls_20261024'
(V / 'venv').symlink_to(V3 / 'venv', target_is_directory=True)
for name in ('seq_theta_train.json', 'seq_theta_val.json'):
    shutil.copy2(V3 / 'inputs_old' / name, V / 'inputs' / name)
manifest = {'inputs': {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in (V / 'inputs').iterdir()},
            'scope': 'CPU validation only; no training; old inputs immutable; no test access'}
(V / 'records/setup_manifest.json').write_text(json.dumps(manifest, indent=2))
print('Independent CPU v7 precheck provisioned.')
