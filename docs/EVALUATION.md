# Evaluation protocol

## Two separate stages

1. **Prediction generation (`run_sequences.py`)**: read paired videos, prepare
   each frame, run the neural network, perform non-maximum suppression, update
   the online estimator when applicable, and save predictions. This uses the GPU
   in the recorded experiment and the CPU for decoding and other work.
2. **Metric calculation (`score_runs.py`)**: read saved predictions and infrared
   annotations and calculate metrics and sequence bootstrap intervals. This is
   CPU-only; it does not execute the detector.

The recorded validation split has 67 sequences and 61,999 infrared annotation
frames. Seven inference configurations require approximately 434,000 frame
forwards when all videos have matching decoded coverage. Annotation and video
length mismatches are recorded in the coverage artifact rather than hidden.

## Frame processing

The runner processes video pairs in order. RGB is resized to the infrared image
dimensions, preserving normalized coordinates but changing its pixel aspect.
Both modalities are resized and padded to a 640 x 640 canvas. Input order is
RGB plus one infrared channel; A0 instead repeats infrared into three channels.
SPLA models also receive two dense prior-offset channels, in canvas pixels.

The model runs in evaluation mode with gradients disabled. The recorded runner
uses FP32 and batch size 1. NMS keeps at most 10 detections with confidence
at least 0.001 and an NMS IoU threshold of 0.6. All retained detections are saved
for AP. The top detection becomes the sequence prediction only when confidence
is at least 0.25; otherwise the prediction is empty. Boxes are converted back
to infrared-frame xywh coordinates.

This is a detector evaluated sequentially. It is not an initialized single-object
tracker: the runner does not receive the first-frame ground-truth target box.

## Prior modes

| Mode | Input information | Interpretation |
|---|---|---|
| zero | Identity transform | No-prior control |
| dataset | Mean transform from training sequences | Fixed calibration |
| online | Training initialization plus previous confident model predictions | Deployable closed-loop mode |
| oracle | Whole-sequence transform fitted to that sequence's annotations | Diagnostic upper bound only |

For online mode, the estimator predicts the prior **before** the current forward.
After the forward, detections with confidence at least 0.5 supply infrared
centers and the alignment layer's estimated RGB offsets. They are converted
to normalized image coordinates and update the estimator for subsequent frames.
Current-frame annotations never supply an online update.

Annotations are separately used to calculate alignment diagnostics. These
diagnostics do not feed back into online prediction. Oracle mode intentionally
uses annotation-derived transforms and must remain separate in every report.

## Metrics implemented here

For each sequence:

- **SA**: target present -> box IoU (or zero for an empty prediction); target
  absent -> one for an empty prediction and zero for a nonempty prediction.
  Average those frame values, then average sequence SA values equally.
- **Penalized SA**: per-sequence SA minus
  `0.2 * (empty_predictions_on_target_frames / target_frames) ** 0.3`.
  Sequence values are then averaged equally.
- **AP50**: match detections at IoU >= 0.5, with at most one true positive per
  target frame. Pool detections across sequences, rank by confidence, and
  integrate the precision envelope over recall. With multiple training seeds,
  compute pooled AP for each seed and then average those AP values.
- **Alignment/prior errors**: diagnostic pixel errors, summarized first within
  sequences and then across sequences. They are not detection accuracy.

These definitions describe the provided implementation. Benchmark versions can
use different penalty normalizations, aggregation, thresholds, or challenge
protocols; do not substitute this AP50 for a different official evaluation.

## Bootstrap and coverage

The scorer performs 10,000 paired resamples of **sequences**, using the same
resampled sequence list for both compared methods. It reports metric differences,
95% percentile intervals, and the fraction of bootstrap differences <= 0.
Frames within a video are not independent samples. These intervals quantify
sequence sampling uncertainty, not uncertainty across training seeds.

The original scorer can skip missing sequence files and uses the available
prediction/annotation overlap for sequence SA. Always verify complete sequence
coverage before interpreting its result. Published coverage files disclose
length mismatches and known annotation issues. The full validation comparison
retains all 67 sequences; the three known problematic validation sequences are
not silently filtered. Prior tuning excludes known issues and uses a training
dataset prior for their converted metadata, as recorded in the experiment notes.

## Why prediction generation takes time

The completed A0 and A1 passes took approximately 37.3 and 41.8 minutes while
sharing GPU 1 with the A1c training job: roughly 28 and 25 frames/s end to end.
The runner decodes two videos, transfers data, forwards the network, reads
detections back, and records diagnostics for each frame. Even A0 decodes RGB
in this implementation. The seven configurations repeat that work.

This implementation does not batch the independent A0/A1/fixed-prior frames or
pipeline decoding and GPU work. Online A4 has a within-sequence feedback
dependency, although different sequences can in principle be processed in
parallel. CPU/GPU synchronization and repeated AP sorting in the bootstrap add
overhead. No stage profiler has been run, so no single bottleneck is claimed.

Saved predictions can be rescored on a CPU without repeating GPU inference.

Optional `--per-seq reports/val_per_seq.csv` writes seed-mean SA for every
video and named run; unavailable scores are blank. This output is written
before the bootstrap starts and does not change existing aggregate scores or
random draws. See [shift diagnostics](SHIFT_DIAGNOSTICS.md) for the current
per-video analysis and six validation-only persistent RGB shift conditions.
