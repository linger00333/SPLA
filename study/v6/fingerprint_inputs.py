"""Hash the actual validation videos and train/val labels, without test access."""
import hashlib
import json
import sys
from pathlib import Path
from runtime import V, R

sys.path.insert(0, str(V / 'kit'))
from spla.antiuav_io import find_files

def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()

rows = []
for split in ('train', 'val'):
    seqs = sorted(p for p in (R / split).iterdir() if p.is_dir())
    for seq in seqs:
        fs = find_files(seq)
        roles = ('ir_label', 'rgb_label', 'ir_video', 'rgb_video') if split == 'val' else ('ir_label', 'rgb_label')
        for role in roles:
            p = fs[role]; assert p is not None
            rows.append({'input': p.relative_to(R).as_posix(), 'role': role, 'bytes': p.stat().st_size, 'sha256': sha(p)})
        print('Hashed:', split, seq.name, flush=True)
manifests = {}
for split in ('train', 'val'):
    p = V / 'records' / f'rgb_dataset_manifest_{split}.json'
    data = json.loads(p.read_text())
    manifests[split] = {'file': p.name, 'sha256': sha(p), 'images_and_labels': len(data),
                         'content': 'Full per-image SHA256 and generated RGB-label SHA256 retained privately'}
report = {'raw_inputs': rows, 'training_input_manifests': manifests, 'test_accessed': False,
          'scope': 'Full SHA256 for all validation videos and train/val modality labels; training uses the separately hashed sampled JPEGs'}
(V / 'records/raw_input_hashes.json').write_text(json.dumps(report, indent=2))
print('Complete input fingerprint:', len(rows), 'raw files', flush=True)
