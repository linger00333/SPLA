"""Exercise the actual runner on real validation frames and physical GPU 1."""
import json
import os
from pathlib import Path
import runpy
import sys

from runtime import V, V3, OLD, ENV, inference

os.environ.update(ENV)
sys.path[:0] = [str(V / 'kit'), str(V / 'framework')]
import torch
import spla.checkpoints as checkpoints

assert torch.cuda.get_device_name(1).startswith('NVIDIA A100')
original = checkpoints.load_inference_model
records = {}
for mode in ('default', 'rgb', 'ir', 'rgb_input'):
    stats = []
    def load(*args, **kwargs):
        model = original(*args, **kwargs)
        def inspect(module, inputs):
            x = inputs[0]
            assert str(x.device) == 'cuda:1'
            if mode == 'rgb':
                assert torch.allclose(x[:, :3], torch.full_like(x[:, :3], 114 / 255))
            if mode == 'ir':
                assert torch.allclose(x[:, 3:4], torch.full_like(x[:, 3:4], 114 / 255))
            stats.append({'device': str(x.device), 'shape': list(x.shape),
                          'channel_sd': x.std(dim=(0, 2, 3)).tolist()})
        model.register_forward_pre_hook(inspect)
        return model
    checkpoints.load_inference_model = load
    extra = ['--max-seq', '1', '--max-frames', '5']
    if mode in ('rgb', 'ir'):
        extra += ['--blank', mode]
    if mode == 'rgb_input':
        extra += ['--input', 'rgb']
    weight = OLD / ('A0_s0' if mode == 'rgb_input' else 'A1_s0') / 'weights/best.pt'
    cmd = inference('runtime_' + mode, weight, extra)
    sys.argv = cmd[1:]
    runpy.run_path(str(V / 'kit/tools/run_sequences.py'), run_name='__main__')
    directory = V / 'results' / ('runtime_' + mode)
    pf = next(directory.glob('*_IR.txt'))
    pred = json.loads(pf.read_text())['res']
    assert len(pred) == 5 and len(stats) == 5
    if mode == 'default':
        old = V3 / 'results/val/A1_s0' / pf.name
        old_dets = old.with_name(pf.name.replace('_IR.txt', '_dets.json'))
        assert pred == json.loads(old.read_text())['res'][:5], 'Default top boxes changed'
        assert json.loads(next(directory.glob('*_dets.json')).read_text()) == json.loads(old_dets.read_text())[:5], 'Default candidates changed'
    cfg = json.loads((directory / 'run_config.json').read_text())
    assert cfg['output_coordinates'] == ('RGB_pixels' if mode == 'rgb_input' else 'IR_pixels')
    records[mode] = {'command': cmd, 'frames': 5, 'gpu_tensor_checks': stats,
                     'original_A1_prefix_exact_match': mode == 'default',
                     'output_coordinates': cfg['output_coordinates'],
                     'note': 'RGB input smoke uses old IR weights solely to verify the real runner; not an RGB baseline result' if mode == 'rgb_input' else None}
(V / 'records/real_runtime_validation.json').write_text(json.dumps(records, indent=2))
print('All four real-frame/GPU runner conditions passed; default equals original A1 predictions exactly.')
