"""Paths and process environment for this isolated A100 study."""
from pathlib import Path
import os

V = Path(__file__).resolve().parent
P = V.parents[1]
V3 = V.parent / 'v3_spla_controls_20261024'
R = P / 'spla_kit_v1/datasets'
PY = str(V / 'venv/bin/python')
OLD = P / 'experiments/v1_spla_zoom_20261023/framework/runs/detect'
ENV = dict(os.environ, TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD='1', OMP_NUM_THREADS='4',
           OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='4', OPENCV_FOR_THREADS_NUM='1',
           PYTHONPATH=str(V / 'kit') + ':' + str(V / 'framework'),
           PYTHONUNBUFFERED='1', YOLO_CONFIG_DIR=str(V / 'yolo_settings'))


def inference(name, weight, extra=()):
    return [PY, str(V / 'kit/tools/run_sequences.py'), '--weights', str(weight),
            '--root', str(R), '--split', 'val', '--out', str(V / 'results' / name),
            '--seq-theta', str(V / 'inputs/seq_theta_val.json'),
            '--train-theta', str(V / 'inputs/seq_theta_train.json'), '--prior', 'zero',
            '--process-sigma', '0.001', '--meas-sigma', '0.008', '--device', 'cuda:1',
            '--conf', '0.25', '--imgsz', '640', *extra]
