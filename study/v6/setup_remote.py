"""Provision an independent experiment without changing any previous experiment."""
from pathlib import Path
import hashlib
import json
import shutil
import subprocess
import zipfile

V = Path(__file__).resolve().parent
V3 = V.parent / 'v3_spla_controls_20261024'
assert not (V / 'kit').exists(), 'Study already provisioned'
with zipfile.ZipFile(V / 'source_bundle.zip') as z:
    assert all(not Path(n).is_absolute() and '..' not in Path(n).parts for n in z.namelist())
    z.extractall(V)
for d in ('records', 'reports', 'inputs', 'results', 'runs', 'private_visuals'):
    (V / d).mkdir(exist_ok=True)
for d in ('framework', 'venv'):
    (V / d).symlink_to(V3 / d, target_is_directory=True)
for name in ('seq_theta_train.json', 'seq_theta_val.json'):
    shutil.copy2(V3 / 'inputs_old' / name, V / 'inputs' / name)
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
weights = {n: V.parents[1] / 'experiments/v1_spla_zoom_20261023/framework/runs/detect' / n / 'weights/best.pt'
           for n in ('A0_s0', 'A1_s0')}
expected = json.loads((V3 / 'records/input_manifest.json').read_text())['old_weights']
assert all(sha(p) == expected[n] for n, p in weights.items())
manifest = {'old_weights': {n: sha(p) for n, p in weights.items()},
            'input_sha256': {p.name: sha(p) for p in (V / 'inputs').iterdir()},
            'code_sha256': {p.relative_to(V).as_posix(): sha(p) for p in (V / 'kit').rglob('*.py')},
            'protocol_sha256': sha(V / 'PROTOCOL.md'),
            'split': 'train/val only', 'physical_gpu': 1, 'recurring_automation': False,
            'authorized_training': 'one independent A0rgb_s0 scratch run, 100 epochs'}
assert sha(V / 'kit/tools/score_runs.py') == sha(V.parent / 'v5_spla_shift_20261024/kit/tools/score_runs.py')
(V / 'records/input_manifest.json').write_text(json.dumps(manifest, indent=2))
from runtime import PY, ENV
with (V / 'records/remote_tests.log').open('w') as log:
    subprocess.run([PY, '-m', 'unittest', 'discover', '-s', str(V / 'kit/tests')], env=ENV,
                   cwd=V, stdout=log, stderr=subprocess.STDOUT, check=True)
print('Isolated v6 ready; checkpoint hashes, unchanged scorer and remote tests passed.')
