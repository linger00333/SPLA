# Experiment scope and evidence

## Model controls

| Name | Input/model | Training |
|---|---|---|
| A0 | Infrared-only detector | 100 epochs, seed 0 |
| A1 | Simple RGB-T fusion | 100 epochs, seed 0 |
| A3 | SPLA architecture, zero prior | 40 extra epochs from A1 |
| A4 | SPLA architecture, sequence prior | 40 extra epochs from the same A1 |
| Old A1c | A1 continuation, no SPLA layers | 40 extra epochs, seed 0; complete |

All recorded runs use the n model scale, imgsz 640, batch 16, and four workers.
Legacy A3/A4 use a four-parameter scale/translation prior, process_sigma=0.001,
and meas_sigma=0.008. Their original training arguments and per-epoch CSV files
are preserved under `artifacts/legacy/training` with portable path aliases.

## Current control study

The completed study has seven whole-video conditions on the original validation
split: A0 zero, A1 zero, A3 zero, and A4 online/dataset/oracle/zero. It also trains
the old A1c continuation control. All seven are complete and scored in one
`score_runs.py` call, with A1 as the bootstrap reference. Every condition covers
67 sequences and 61,999 frames without prediction/annotation length mismatches.
The old A1c also has a completed, separate A1 comparison. Its full 40-epoch CSV
and arguments are preserved. Its mAP50-95 ends at 0.62089, below the original
A1 best of 0.64640; this does not by itself identify a unique cause.

Automatic follow-up is disabled. There is no user crontab for this experiment.
The current server controller is a single launched job, not recurring training.
No scheduler, credential, host address, or machine-specific controller is
installed by this repository.

The zoom plan is superseded. Its prior-error median did not improve, so the
four-parameter scale/translation model and process_sigma=0.001,
meas_sigma=0.008 remain fixed. Frozen training/validation replay metadata are
merged without test records.

## Fine-tuning patch and pilot gate

The supplied patch applies cleanly to release `0f4d544`. Its original SHA-256 is
`3a96a711e418c6486bc0f53d03e959b0115aec0d1ab90d163c736f000f7f2e90`.
Fine-tuning defaults to SGD with lr0=0.0001, lrf=1.0, no warmup and no mosaic.
The old A1 final CSV logs LR 0.000199; 0.0001 is the schedule's theoretical
endpoint stated in the supplied patch help. The LR diagnosis remains a
hypothesis to be tested rather than a demonstrated cause.

The full-data 10-epoch A1c pilot starts from the same frozen original A1 best
checkpoint. Before any optimizer step, the same training validator measures
baseline B on the complete converted validation set. Random states are restored
after this measurement. The gate was fixed before training:

- Exactly 10 completed epochs and logged LR 0.0001 throughout.
- Final mAP50-95 and the last-three-epoch mean must each be at least B - 0.003.
- The minimum epoch must be at least B - 0.005.
- The last-three mean must be at least the first-three mean - 0.003.

The published pilot CSV is a progress snapshot until it contains all 10 epochs
and has a gate report. Its pre-training baseline is 0.64640. Early values of
0.64190, 0.63999 and 0.63969 show that lowering LR alone has not retained the
original score; the complete curve is still being collected before the fresh
frozen-backbone retry. No passing result is inferred from these partial rows.

If the unfrozen pilot fails, a fresh pilot uses the original A1 checkpoint with
`--freeze 21`. Both backbones' parameters are frozen; batch-normalization
running statistics are not explicitly frozen. If both pilots fail, formal
training does not start. Runtime failures require diagnosis and do not count as
failed scientific retention tests. A first pre-training callback failure was
fixed before any training batches; its diagnosis remains in the records.

Only a passing pilot releases the one-shot matched study: A1 scratch seeds 1/2
for 100 epochs, then A1c/A3/A4 for 40 epochs each at seeds 0/1/2. Each group uses
the corresponding seed's original A1 initialization and the same passing LR,
warmup and freeze settings. Pilot-finetuned weights are not reused as the
formal initialization. Maximum concurrency is two GPU tasks on the authorized
GPU. The status files are timestamped snapshots; queued work is not a result.
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

- A4 versus A3 changes the training prior within the same architecture and
  initialization, as well as the inference prior; A4-zero helps distinguish
  inference dependence. One seed is not a training-seed robustness study.
- A1 versus A0 asks whether this RGB-T recipe benefits from RGB; the current
  completed comparison does not show a consistent advantage across SA and AP50.
- A4 versus A4-zero checks dependence on the prior at inference.
- A4 versus oracle diagnoses the remaining gap to annotation-assisted alignment;
  it is not an attainable guarantee for the online estimator.
- A matched-recipe A1c helps control extra fine-tuning. SPLA also changes geometry augmentation
  and introduces alignment supervision, so residual differences do not isolate
  one architectural operation alone.
- Validation model selection uses the training dataset prior for A4 and identity
  for A3. Annotation-assisted validation replay/oracle is diagnostic only.
- Known annotation issues remain disclosed; the full video report retains all
  67 validation sequences. No claim of cross-dataset or test generalization is made.
