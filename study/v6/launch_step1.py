"""Launch exactly one ablation controller; existing state prevents duplicates."""
import subprocess
from runtime import V, PY, ENV
assert not (V / 'step1_status.json').exists()
p = subprocess.Popen([PY, str(V / 'controller_step1.py')], cwd=V, env=ENV,
                     stdout=(V / 'step1_controller.log').open('w'), stderr=subprocess.STDOUT,
                     start_new_session=True)
print('One-shot step-1 controller started:', p.pid)
