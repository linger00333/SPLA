"""Extend the pre-training gate to every shared framework source file."""
import hashlib
import json
import os
import sys
from runtime import V, ENV

os.environ.update(ENV)
sys.path[:0] = [str(V / 'kit'), str(V / 'framework')]
state = json.loads((V / 'step2_status.json').read_text())
assert not state['training_launched'] and state['phase'] == 'waiting_for_step1'
p = V / 'records/frozen_code.json'
manifest = json.loads(p.read_text())
files = sorted((V / 'framework/ultralytics').rglob('*.py'))
for file in files:
    manifest['sha256'][file.relative_to(V).as_posix()] = hashlib.sha256(file.read_bytes()).hexdigest()
manifest['framework_freeze_scope'] = 'Every shared framework Python source; frozen while step 1 runs, before RGB training starts. Existing kit/controller hashes retained.'
manifest['framework_python_files'] = len(files)
p.write_text(json.dumps(manifest, indent=2))
import torch
import numpy
import cv2
import ultralytics
import spla.checkpoints
report = {'python': sys.version, 'torch': torch.__version__, 'numpy': numpy.__version__,
          'opencv': cv2.__version__, 'actual_framework_import': ultralytics.__file__,
          'actual_kit_import': spla.checkpoints.__file__, 'framework_source_files': len(files),
          'verification': 'The already-running step-2 controller will check all these hashes immediately before its only training run'}
(V / 'records/runtime_source_audit.json').write_text(json.dumps(report, indent=2))
print(json.dumps(report))
