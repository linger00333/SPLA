# Experiment scope and evidence

## Model controls

| Name | Input/model | Training |
|---|---|---|
| A0 | Infrared-only detector | 100 epochs, seed 0 |
| A1 | Simple RGB-T fusion | 100 epochs, seed 0 |
| A3 | SPLA architecture, zero prior | 40 extra epochs from A1 |
| A4 | SPLA architecture, sequence prior | 40 extra epochs from the same A1 |
| A1c | A1 continuation, no SPLA layers | 40 extra epochs, seed 0; ongoing snapshot |

All recorded runs use the n model scale, imgsz 640, batch 16, and four workers.
Legacy A3/A4 use a four-parameter scale/translation prior, process_sigma=0.001,
and meas_sigma=0.008. Their original training arguments and per-epoch CSV files
are preserved under `artifacts/legacy/training` with portable path aliases.

## Current control study

The new study requests seven whole-video conditions on the original validation
split: A0 zero, A1 zero, A3 zero, and A4 online/dataset/oracle/zero. It also trains
the A1c continuation control. The status export names completed versus running
jobs. Only completed prediction directories are included in this publication.
The partial A1c CSV is explicitly a progress snapshot, not a final score.

Automatic follow-up is disabled. There is no user crontab for this experiment.
The current server controller is a single launched job, not recurring training.
No scheduler, credential, host address, or machine-specific controller is
installed by this repository.

The requested subsequent study has not started: three-parameter zoom conversion,
training-only prior tuning, causal replay generation, A1 seeds 1/2, and three
seed-matched A3/A4 pairs. It uses lr0=0.001 and zero warmup. Its execution is
conditioned on first delivering the existing comparison and tuning reports.
Test and stress evaluation have not been run.

## Necessary fixes already applied

1. Transient layer caches are omitted from model serialization/deepcopy, avoiding
   graph-backed cache failures during EMA/checkpoint creation.
2. The dataset loader receives four image channels; the two dense prior channels
   are appended after loading, preventing six-channel color augmentation errors.
3. Replay metadata uses the pre-observation state for an observed frame and the
   latest post-observation state for intervening frames. It avoids an extra
   observation interval of lag without leaking a current-frame label update.

The original package omitted these fixes; they were ported into the tested
runtime. Nineteen package tests and real batch-16 A1c/SPLA GPU training,
validation, save, and reload checks passed before formal jobs started. Temporary
diagnostic code and temporary model outputs were removed after recording a
concise validation summary. New portable model loading has its own export
verification artifact.

## Interpretation limits

- A4 versus A3 isolates much of the prior effect within the same architecture
  and initialization, but one seed is not a training-seed robustness study.
- A1 versus A0 asks whether this RGB-T recipe benefits from RGB; the current
  snapshot does not show a consistent advantage across SA and AP50.
- A4 versus A4-zero checks dependence on the prior at inference.
- A4 versus oracle diagnoses the remaining gap to annotation-assisted alignment;
  it is not an attainable guarantee for the online estimator.
- A1c helps control extra fine-tuning. SPLA also changes geometry augmentation
  and introduces alignment supervision, so residual differences do not isolate
  one architectural operation alone.
- Validation model selection uses the training dataset prior for A4 and identity
  for A3. Annotation-assisted validation replay/oracle is diagnostic only.
- Known annotation issues remain disclosed; the full video report retains all
  67 validation sequences. No claim of cross-dataset or test generalization is made.
