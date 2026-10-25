"""Isolated CPU-only v7 paths; old experiments are read-only inputs."""
from pathlib import Path
import os
V = Path(__file__).resolve().parent
P = V.parents[1]
V3 = V.parent / 'v3_spla_controls_20261024'
V6 = V.parent / 'v6_fusion_diagnostics_20261024'
R = P / 'spla_kit_v1/datasets'
IR = V3 / 'results/val/A0_s0'
RGB = V6 / 'results/A0rgb_s0'
PY = str(V / 'venv/bin/python')
ENV = dict(os.environ, PYTHONPATH=str(V / 'kit'), OMP_NUM_THREADS='2',
           OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='2', OPENCV_FOR_THREADS_NUM='1',
           PYTHONUNBUFFERED='1', CUDA_VISIBLE_DEVICES='')
