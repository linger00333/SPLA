# SPLA: Sequence-Prior Local Alignment

Research implementation for unregistered RGB and thermal infrared UAV detection.
The model aligns RGB features to the infrared grid using a sequence transform
and a learned local residual. The deployable online mode estimates its transform
from the detector's previous predictions.

## Current evidence

This repository contains the tested implementation, completed legacy seed-0
models, training curves, completed prediction exports, and reproducibility
records. All seven conditions below were evaluated with the same current video
runner and scored together in one invocation, on all 67 validation sequences.

| Validation condition (legacy seed-0 weights) | SA | Penalized SA | AP50 |
|---|---:|---:|---:|
| A0, infrared only | 0.815417 | 0.799854 | 0.990056 |
| A1, simple RGB-T fusion | 0.814932 | 0.798989 | 0.990287 |
| A3, zero prior | 0.813385 | 0.797093 | 0.989535 |
| A4, online prior | 0.815502 | 0.800971 | 0.990973 |
| A4, fixed dataset prior | 0.815405 | 0.800911 | 0.991511 |
| A4, annotation oracle | 0.815109 | 0.800424 | 0.991283 |
| A4, zero prior | 0.815502 | 0.801348 | 0.990890 |

The unified report includes paired sequence bootstrap intervals relative to A1:
[`artifacts/v3/val_scores_all.json`](artifacts/v3/val_scores_all.json).
These are validation results from the legacy four-parameter transform recipe,
with one training seed. A4 online and A4 zero have nearly identical SA; the
A4-versus-A3 difference alone does not demonstrate a useful online inference
prior. A0/A1 show no consistent RGB-T advantage across these metrics. Sequence
bootstrap intervals do not measure variation across training seeds.

The old A1c control completed 40 epochs at an initial LR of 0.001. Its selected
checkpoint has SA 0.812468, penalized SA 0.796586, and AP50 0.990555; see the
[control report](artifacts/v3/val_scores_A1c.json). Its recipe differs from legacy
A3/A4 and cannot be used as their matched fine-tuning control.

The supplied [fine-tuning patch](patches/finetune_recipe.patch) is applied.
Fine-tuning now defaults to constant LR 0.0001 with no warmup. Both full-data,
10-epoch A1c pilots completed and failed the predeclared retention gate:

| Pilot | Initial mAP50-95 | Final mAP50-95 | Gate |
|---|---:|---:|---|
| LR 0.0001, unfrozen | 0.64640 | 0.63524 | Failed |
| LR 0.0001, freeze 21 | 0.64640 | 0.63691 | Failed |

The matched three-seed study was not launched. Both complete CSVs, runtime
audits and gate reports are in [artifacts/v4](artifacts/v4); see the
[pilot summary](artifacts/v4/reports/pilot_summary.json). Lowering LR alone did
not retain the original score; freezing backbone parameters also failed.
Batch-normalization running statistics were not explicitly frozen, and the
underlying cause remains unresolved. No further recipe search was launched.
The four-parameter `scale_translation` model is retained. Test evaluation has
not been executed. The validation shift diagnostic is complete.

## Contents

The evaluation-only [persistent RGB shift study](docs/SHIFT_DIAGNOSTICS.md)
completed all six conditions and eight paired bootstrap comparisons. At 128 IR
pixels, online A4 has target alignment error 1.75 px versus 12.15 px with zero
prior, but its SA and AP50 advantages have intervals crossing zero. Penalized
SA improves by 0.003124, with an unadjusted 95% sequence interval
[0.000703, 0.005763]. This is a small exploratory robustness signal, not a broad
detection superiority claim. See the [full report](artifacts/v5/REPORT.md).
All 67 videos are retained. No new training was launched. The completed clean
screen also finds two videos where A1 exceeds A0 by more than 0.02 SA.

- `kit/spla`: geometry, online estimator, alignment layers, training integration,
  and portable tensor-only checkpoint loading.
- `kit/tools`: conversion, prior tuning, training, video inference, and scoring.
- `framework`: the experiment's YOLOv11-RGBT runtime, with the SPLA parser patch.
- `models/legacy`: four completed models as Safetensors plus model configuration.
- `artifacts/legacy`: completed legacy results and per-epoch training records.
- `artifacts/predictions`: completed predictions, compressed per file.
- `artifacts/v3`: complete seven-condition scores, coverage and old A1c curves.
- `artifacts/v4`: low-LR pilot evidence and a timestamped execution snapshot.
- `artifacts/v5`: per-video RGB gains, shift geometry and stress-study status.
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
