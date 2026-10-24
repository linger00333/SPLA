"""Inference-only loading of portable, tensor-only model bundles."""
from pathlib import Path
import json


def load_inference_model(weights, device):
    """Use a sibling config.json for Safetensors; retain trusted local PT support."""
    path = Path(weights)
    if path.suffix != '.safetensors':
        from ultralytics.nn.tasks import attempt_load_one_weight
        model, _ = attempt_load_one_weight(str(path), device=device)
        return model.float().eval()

    from safetensors.torch import load_file
    from ultralytics.nn.tasks import DetectionModel
    config_path = path.with_name('config.json')
    config = json.loads(config_path.read_text(encoding='ascii'))
    model = DetectionModel(config, ch=int(config['ch']), nc=int(config['nc']), verbose=False)
    model.load_state_dict(load_file(str(path), device='cpu'), strict=True)
    model.names = {0: 'uav'}
    return model.to(device).float().eval()
