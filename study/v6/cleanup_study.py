"""Remove only generated transfers and validated smoke outputs after publication."""
from pathlib import Path
import json
import shutil

V = Path(__file__).resolve().parent
targets = [V / 'source_bundle.zip', V / 'publication_export.zip', V / 'publication_export']
targets += [V / 'results' / ('runtime_' + m) for m in ('default', 'rgb', 'ir', 'rgb_input')]
assert (V / 'records/real_runtime_validation.json').exists()
assert json.loads((V / 'step2_status.json').read_text())['phase'] == 'complete_report_ready'
removed = []
for p in targets:
    assert p.resolve().is_relative_to(V.resolve()) and p.resolve() != V.resolve()
    assert not p.is_symlink()
    if not p.exists():
        continue
    size = sum(x.stat().st_size for x in p.rglob('*') if x.is_file()) if p.is_dir() else p.stat().st_size
    if p.is_dir():
        shutil.rmtree(p)
    else:
        p.unlink()
    removed.append({'path': p.relative_to(V).as_posix(), 'bytes': size})
(V / 'records/cleanup_after_verified_export.json').write_text(json.dumps({'removed': removed,
    'preserved': 'Original patch, formal predictions, weights, reports, failure evidence and all shared runtimes'}, indent=2))
print('Removed confirmed generated duplicates and smoke outputs only:', len(removed))
