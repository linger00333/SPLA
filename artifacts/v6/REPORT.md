# V6 completed fusion diagnostics

Two A1 blank-input validation runs, one independent 100-epoch scratch RGB-only training run, RGB validation inference and fixed-detector complementarity are complete. Every full inference run covers 67 videos and 61,999 frames without length gaps. Test was not evaluated. Only physical GPU 1 was used; no recurring automation or failed-pilot three-seed queue was started.

## A1 blank-input intervention

The original A1 weights, four-parameter prior tables, zero prior, thresholds and IR-coordinate scorer are retained. The unchanged scorer uses 10,000 paired sequence bootstrap draws. Flat 114 inputs are out of distribution; these differences do not identify a strict causal contribution or reliance upper bound.

| Condition | SA | Penalized SA | AP50 |
|---|---:|---:|---:|
| A1 | 0.814932 | 0.798989 | 0.990287 |
| A1_blankrgb | 0.798835 | 0.778071 | 0.985698 |
| A1_blankir | 0.022213 | -0.177787 | 0.000001 |

| Difference from A1 | SA point delta [95% CI] | Penalized SA point delta [95% CI] | AP50 point delta [95% CI] |
|---|---|---|---|
| A1_blankrgb-A1 | -0.016097 [-0.023274, -0.009369] | -0.020918 [-0.029237, -0.013116] | -0.004589 [-0.008677, -0.001867] |
| A1_blankir-A1 | -0.792720 [-0.813837, -0.762408] | -0.976776 [-1.002155, -0.941873] | -0.990286 [-0.993558, -0.986594] |

Undefined values from the unchanged legacy scorer, including AP when no candidate exists, are retained as null rather than converted to zero.

## RGB-only baseline and fixed-detector selection

A0rgb_s0 completed exactly 100 epochs with the old A0 scratch recipe: YOLO11n, seed 0, 640, batch 16 and workers 4. Its selected checkpoint uses validation performance in RGB labels. Training and validation use exactly the old 29,708/12,158 sampled images. The original sampling depends on IR existence, and the visible images are stretched to the IR dimensions. This is not a native-resolution RGB benchmark.

The actual framework loader was checked against every RGB-normalized label; all images were hard-linked, source labels remained unchanged, and four real validation overlays were inspected. A decimal-format audit mismatch was corrected without changing labels; its failure evidence was retained. Source images and overlays are not published.

Complementarity compares old A0 and new A0rgb_s0, each against its own modality labels. Saved RGB boxes use original RGB pixels despite the historical filename suffix. On the intersection of valid positive annotations, the highest-confidence saved box succeeds at IoU >= 0.5. Missing files and unequal lengths are errors. These conditional success rates are not official-style SA and do not directly quantify IR-coordinate improvements.

Both-positive frames: 57982; IR failures: 619. All-frame coverage: 61,999. Each bootstrap draw samples 67 whole sequences with replacement and recomputes pooled ratios; 10,000 draws, seed 0.

| Statistic | Value |
|---|---:|
| ir_success | 0.989324 |
| rgb_success | 0.991446 |
| oracle_success | 0.999776 |
| oracle_gain | 0.010452 [0.005669, 0.017295] |
| rescue_share | 0.978998 [0.943925, 0.996800] |

oracle_gain is RGB-only successes divided by all both-positive frames; rescue_share divides those successes by IR failures. The oracle selects between two fixed outputs. It is not a mathematical upper bound on fusion models, retrained detectors or native RGB resolution.

Known release annotation issues remain unchanged and are listed in the JSON report. The primary analysis retains all videos; the sensitivity summary excludes flagged sequences:

```json
{
  "frames": 56141,
  "ir_success": 0.9890988760442457,
  "rgb_success": 0.9912541636237331,
  "oracle_success": 0.9997684401774104,
  "oracle_gain": 0.01066956413316471,
  "rescue_share": 0.9787581699346405,
  "ir_failures": 612
}
```

## Next step

The diagnostic is inconclusive under the predeclared exploratory bands. Review sequence variation, known label issues and error cases before proposing a single next experiment; no training search is launched.

The 0.5% and 2% bands were frozen before observing these results and are task-specific exploratory heuristics, not accepted standards. Confidence intervals are unadjusted for multiple comparisons and quantify sequence sampling, not training-seed robustness. Blanking and shift tolerance do not prove RGB is ignored; optimization, EMA and normalization explanations remain unverified.

Full numeric evidence: [ablation scores](reports/ablation_scores.json), [per-video SA](reports/ablation_per_sequence.csv), [complementarity](reports/complementarity_val.json), [dataset audit](reports/dataset_audit.json), and [frozen protocol](PROTOCOL.md). All three complete prediction sets and the portable RGB model are included.
