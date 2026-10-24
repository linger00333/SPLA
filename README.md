# SPLA: Sequence-Prior Local Alignment

Research implementation for unregistered RGB and thermal infrared UAV detection.
The model aligns RGB features to the infrared grid using a sequence transform
and a learned local residual. The deployable online mode estimates its transform
from the detector's previous predictions.

## Current evidence

This repository contains the tested implementation, completed legacy seed-0
models, training curves, completed prediction exports, and reproducibility
records. It also contains a timestamped snapshot of the ongoing control study.
It does not claim that the new three-seed zoom experiment has been completed.

| Legacy validation run | SA | Penalized SA | AP50 |
|---|---:|---:|---:|
| A3, zero prior | 0.813385 | 0.797093 | 0.989535 |
| A4, online prior | 0.815502 | 0.800971 | 0.990973 |

The paired sequence bootstrap for A4 minus A3 is available in
[`artifacts/legacy/val_scores.json`](artifacts/legacy/val_scores.json).
These are validation results from the legacy four-parameter transform recipe,
with one training seed. They do not establish superiority over A1, control for
all extra fine-tuning effects, or measure variation across training seeds.

The newer, completed baseline inference snapshot gives:

| Baseline | SA | Penalized SA | AP50 |
|---|---:|---:|---:|
| A0, infrared only | 0.815417 | 0.799854 | 0.990056 |
| A1, simple RGB-T fusion | 0.814932 | 0.798989 | 0.990287 |

This snapshot has no bootstrap interval. The complete seven-condition control
comparison remains pending as of the exported status. Test and stress evaluation
have not been executed.

## Contents

- `kit/spla`: geometry, online estimator, alignment layers, training integration,
  and portable tensor-only checkpoint loading.
- `kit/tools`: conversion, prior tuning, training, video inference, and scoring.
- `framework`: the experiment's YOLOv11-RGBT runtime, with the SPLA parser patch.
- `models/legacy`: four completed models as Safetensors plus model configuration.
- `artifacts/legacy`: completed legacy results and per-epoch training records.
- `artifacts/predictions`: completed predictions, compressed per file.
- `artifacts/v3`: a timestamped status and control-training snapshot.
- `docs`: evaluation protocol, experiment scope, installation, and model use.
- `scripts/check_release.py`: publication content and manifest verification.

Raw dataset videos, frames, original annotations, machine-specific environments,
temporary diagnostics, and temporary training runs are not included.

## Install and use

See [installation](docs/INSTALL.md) and [evaluation protocol](docs/EVALUATION.md).
The implementation was validated with Python 3.10.20, PyTorch 2.6.0+cu118,
NumPy 1.26.4, OpenCV 4.8.1, and an NVIDIA A100 40 GB. A fresh installation of the
publication snapshot has not been independently validated on another machine.

```bash
python -m pip install -r requirements.txt
python -m pip install --no-deps -e kit
python -m pip install --no-deps -e framework
python kit/tools/patch_ultralytics.py framework
```

All training and inference starts only when the corresponding command is run.
This repository does not install scheduled training or background services.

## Provenance and license

The runtime is derived from [YOLOv11-RGBT](https://github.com/wandahangFY/YOLOv11-RGBT)
and [Ultralytics](https://github.com/ultralytics/ultralytics). Source notices and
the AGPL-3.0 license are retained. See [NOTICE](NOTICE.md) and [LICENSE](LICENSE).
The original supplied kit archive SHA-256 and source checkpoint hashes are
recorded under `artifacts/records` and `models/registry.json`.

Repository text, filenames, configuration values, and model metadata use ASCII.
Machine paths are replaced with portable aliases. Model files contain tensors
and explicit portable metadata, without original training checkpoint objects.
