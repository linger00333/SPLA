# Per-video RGB gains and persistent-shift diagnostics

This evaluation-only study reuses the completed legacy weights and frozen
four-parameter priors. It does not launch training or evaluate the test split.
The protocol was written before the six stress runs started; see
[the protocol](../artifacts/v5/PROTOCOL.md) and timestamped execution records.

## Per-video export

`score_runs.py --per-seq PATH` optionally writes a CSV with one `sequence`
column and one `NAME_SA` column per named run. Each value averages the saved
prediction scores across the supplied training-seed directories. Missing or
non-finite values are blank, not zero. The option does not change aggregate
metrics, bootstrap calculations, or random draws. Three regression checks cover
seed means, missing values/CSV quoting, and identical aggregate/interval output.

Repeat the seven-condition clean command with its original run order and
`--baseline A1`, adding `--per-seq reports/val_per_seq.csv`. This only reads
saved predictions and annotations on the CPU; it does not repeat inference.

The complete [per-video CSV](../artifacts/v5/reports/val_per_seq.csv) gives two
videos with A1 minus A0 SA strictly greater than 0.02:

| Video | A0 SA | A1 SA | A1 minus A0 |
|---|---:|---:|---:|
| 20190926_200510_1_3 | 0.734120 | 0.797626 | +0.063507 |
| 20190925_101846_1_4 | 0.731997 | 0.752279 | +0.020282 |

Four videos favor A0 over A1 by more than 0.02. See the full
[sorted gain table](../artifacts/v5/reports/val_rgb_gain.csv). This is a
descriptive, post-hoc screen on validation data. It is not an independent
benchmark or proof that RGB is universally useful or useless. Both selected
positive-gain videos are outside the disclosed known-annotation-issue list.

## Six inference runs

For each shift magnitude P in 48 and 128 IR pixels, run the existing
`run_sequences.py` with:

| Name | Weights | Prior | Additional arguments |
|---|---|---|---|
| A1_shiftP | Original A1 seed 0 | zero | `--shift-mode persistent --shift-px P` |
| A4_shiftP | Original A4 seed 0 | online | same shift, `--process-sigma 0.001 --meas-sigma 0.008` |
| A4z_shiftP | Same original A4 | zero | same shift |

Use `--split val`, the original validation/train theta tables, imgsz 640,
FP32, batch size 1 and the original confidence/NMS settings. Every run covers
all 67 videos, including known annotation issues. A1 has no SPLA prior input.
The raw-video runner is unchanged from the completed clean evaluation.

Only RGB is shifted, after stretching to IR dimensions and before letterbox.
One deterministic vector is fixed for each video. All models and prior modes
receive the same vector; the two magnitudes share its direction. Zero borders
and clipping can remove RGB target content, so the
[geometry table](../artifacts/v5/reports/shift_geometry.csv) records target
retention, alongside exact vectors and image dimensions.

The actual A4 global head can add up to 96 canvas pixels per axis; local
P3/P4/P5 corrections add 24/48/96. The combined per-axis theoretical bounds are
120/144/192. At 128 IR pixels, only 25 of 67 vectors exceed the P3 bound in an
axis; none do at 48. These perturbations do not guarantee that every per-image
alignment mechanism is beyond its capacity. Bounds do not imply that the
learned model can achieve every allowed displacement.

Compare each shifted run to its own clean condition and A4 online to A4 zero
at each shift. Use 10,000 paired sequence bootstrap resamples and report SA,
penalized SA, AP50 and alignment diagnostics, with coverage checked first.
Keep point differences distinct from bootstrap means. An online/zero gap would
show prior dependence under this perturbation; it would not by itself prove a
general deployment contribution. Similar degradation would not uniquely prove
that the prior was never learned or that one retraining recipe must work.

The six runs are in progress in the exported snapshot. Only completed
prediction sets are exported; queued work is not a measured result. Follow-up
automation remains disabled. No retraining is authorized by this diagnostic.

## Boundaries of the earlier interpretation

The clean A4 online/zero near-equality shows little detection dependence on
the supplied prior under that particular condition. It does not distinguish
ignoring RGB from compensating with per-image alignment. The target-error
comparison with oracle or tuning has different estimators and fitting
objectives; it is diagnostic rather than proof of an optimal prior estimator.
EMA reset, BN running-statistics drift and frame correlation remain candidate
explanations for pilot degradation, not established causes. Two failed pilots
do not establish that every possible fine-tuning recipe must fail.
