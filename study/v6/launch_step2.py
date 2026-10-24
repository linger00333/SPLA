"""Start the authorized gated one-shot baseline queue exactly once."""
import hashlib
import json
import subprocess
from runtime import V, PY, ENV
assert not (V / 'step2_status.json').exists()
files = list((V / 'kit').rglob('*.py')) + [V / n for n in ('controller_step2.py', 'runtime.py', 'PROTOCOL.md')]
(V / 'records/frozen_code.json').write_text(json.dumps({'sha256': {
    p.relative_to(V).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}}, indent=2))
p = subprocess.Popen([PY, str(V / 'controller_step2.py')], cwd=V, env=ENV,
                     stdout=(V / 'step2_controller.log').open('w'), stderr=subprocess.STDOUT, start_new_session=True)
print('One-shot gated RGB baseline controller started:', p.pid)
