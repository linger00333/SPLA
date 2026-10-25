"""Freeze v7 sources and audit every input before any fusion score is read."""
import hashlib
import json
import os
import subprocess
import sys
from runtime import V, R, IR, RGB, PY, ENV
os.environ.update(ENV)
sys.path[:0] = [str(V / 'kit'), str(V / 'kit/tools')]
from decision_fusion import load_pair, run_sequences
from spla.antiuav_io import find_files, read_labels
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
assert not (V / 'records/frozen_manifest.json').exists()
seqs = sorted(p for p in (R / 'val').iterdir() if p.is_dir())
assert len(seqs) == 67 and [p.name for p in seqs] == run_sequences(IR) == run_sequences(RGB)
rows = []; hashes = {}
for seq in seqs:
    fs = find_files(seq)
    for role in ('ir_label', 'rgb_label'):
        raw = json.loads(fs[role].read_text())
        assert len(raw['exist']) == len(raw['gt_rect'])
        hashes['data/raw/val/' + seq.name + '/' + fs[role].name] = sha(fs[role])
    ei, _ = read_labels(fs['ir_label']); er, _ = read_labels(fs['rgb_label'])
    pred, di, dr = load_pair(IR, RGB, seq.name)
    assert len(pred) == len(ei) == len(er) > 0
    rows.append({'sequence': seq.name, 'frames': len(pred)})
    for name, directory in (('A0_s0', IR), ('A0rgb_s0', RGB)):
        for suffix in ('_IR.txt', '_dets.json'):
            p = directory / (seq.name + suffix)
            hashes['source_runs/' + name + '/' + p.name] = sha(p)
assert sum(r['frames'] for r in rows) == 61999
for p in (V / 'inputs').iterdir():
    if p.is_file(): hashes['inputs/' + p.name] = sha(p)
sources = {p.relative_to(V).as_posix(): sha(p) for p in (V / 'kit').rglob('*.py')}
for name in ('PROTOCOL.md', 'runtime.py', 'controller.py', 'freeze_and_audit.py', 'heldout_sync.py'):
    sources[name] = sha(V / name)
manifest = {'code_sha256': sources, 'input_sha256': hashes, 'split': 'val', 'n_sequences': 67, 'frames': 61999,
            'sequences': rows, 'training': False, 'GPU_used': False, 'test_accessed': False,
            'registration_commit': (V / 'registration_commit.txt').read_text().strip()}
(V / 'records/frozen_manifest.json').write_text(json.dumps(manifest, indent=2))
with (V / 'records/remote_tests.log').open('w') as log:
    subprocess.run([PY, '-m', 'unittest', 'discover', '-s', str(V / 'kit/tests')], env=ENV,
                   cwd=V, stdout=log, stderr=subprocess.STDOUT, check=True)
print('Frozen code/input hashes, all 67/61999 source frames and remote tests passed.')
