# Completed validation RGB shift diagnostics

All six inference conditions and eight paired sequence bootstrap comparisons completed. No training was launched and test was not evaluated. Each condition covers all 67 validation videos and 61,999 frames, without length mismatches. The original seven-condition scores and bootstrap outputs were reproduced exactly.

## Detection scores

| Shift (IR px) | Model / inference prior | SA | Penalized SA | AP50 |
|---:|---|---:|---:|---:|
| 0 | A1 / zero | 0.814932 | 0.798989 | 0.990287 |
| 0 | A4 / online | 0.815502 | 0.800971 | 0.990973 |
| 0 | A4 / zero | 0.815502 | 0.801348 | 0.990890 |
| 48 | A1 / zero | 0.813671 | 0.795695 | 0.989700 |
| 48 | A4 / online | 0.815280 | 0.800878 | 0.990894 |
| 48 | A4 / zero | 0.815145 | 0.800340 | 0.990714 |
| 128 | A1 / zero | 0.811516 | 0.792722 | 0.989572 |
| 128 | A4 / online | 0.814700 | 0.799134 | 0.990969 |
| 128 | A4 / zero | 0.814257 | 0.796010 | 0.990440 |

## Matched differences

Point differences and paired 95% sequence bootstrap intervals (10,000 resamples). Intervals do not quantify training-seed uncertainty. No multiplicity adjustment was applied; these are diagnostic comparisons.

| Comparison | SA delta [95% CI] | Penalized SA delta [95% CI] | AP50 delta [95% CI] |
|---|---|---|---|
| A1_shift48 minus A1 | -0.001261 [-0.004716, +0.001750] | -0.003293 [-0.008063, +0.000812] | -0.000587 [-0.001783, +0.000430] |
| A4_shift48 minus A4 | -0.000222 [-0.000755, +0.000222] | -0.000093 [-0.000862, +0.000546] | -0.000078 [-0.000415, +0.000233] |
| A4z_shift48 minus A4z | -0.000357 [-0.001032, +0.000197] | -0.001008 [-0.002383, +0.000191] | -0.000176 [-0.000600, +0.000225] |
| A1_shift128 minus A1 | -0.003417 [-0.007521, +0.000370] | -0.006267 [-0.011732, -0.001356] | -0.000715 [-0.001823, +0.000308] |
| A4_shift128 minus A4 | -0.000802 [-0.002612, +0.000417] | -0.001837 [-0.005831, +0.000473] | -0.000003 [-0.000654, +0.000634] |
| A4z_shift128 minus A4z | -0.001246 [-0.003819, +0.000715] | -0.005338 [-0.009967, -0.001681] | -0.000450 [-0.001167, +0.000226] |
| A4_shift48 minus A4z_shift48 | +0.000135 [-0.000499, +0.000734] | +0.000538 [-0.000868, +0.002057] | +0.000181 [-0.000406, +0.000841] |
| A4_shift128 minus A4z_shift128 | +0.000444 [-0.000939, +0.001845] | +0.003124 [+0.000703, +0.005763] | +0.000529 [-0.000117, +0.001263] |

## Interpretation

At 128 IR pixels, A1 SA decreases by only 0.003417 from its clean baseline. The SA interval crosses zero, while its penalized SA decrease is supported by a negative interval. This protocol shows modest aggregate sensitivity, not proof that real RGB misregistration never matters.

For identical A4 weights at 128 pixels, online minus zero SA is +0.000444, CI [-0.000939, +0.001845]; AP50 is +0.000529, CI [-0.000117, +0.001263]. Neither interval excludes zero. Penalized SA improves by +0.003124, CI [+0.000703, +0.005763]. This is a small exploratory detection-robustness signal under the stated sequence-bootstrap protocol, not a general superiority claim.

Median-of-video target alignment error is 1.7456 px online versus 12.1529 px with zero prior; online target-prior error is 3.7838 px. These are alignment diagnostics, not SA. At 48 pixels, online-minus-zero intervals cross zero for all three detection metrics.

The results do not support saying the network can never use its prior: alignment under the larger perturbation responds strongly, with a small improvement in the penalized metric. They also do not establish a large detection gain or justify launching new training automatically.

## Shift retention and selected videos

Shifts have zero borders and clip content. The following fractions use paired target frames; known annotation issues remain disclosed.

| Shift | RGB box fully inside | RGB center inside |
|---:|---:|---:|
| 48 | 0.997137 | 0.999000 |
| 128 | 0.981684 | 0.989359 |

The two videos selected by clean A1-minus-A0 SA > 0.02 are an exploratory validation subset, not an independent test. Their 128-pixel SA values are:

| Video | A1 clean | A1 shifted | A4 online shifted | A4 zero shifted |
|---|---:|---:|---:|---:|
| 20190926_200510_1_3 | 0.797626 | 0.784153 | 0.801772 | 0.800809 |
| 20190925_101846_1_4 | 0.752279 | 0.746555 | 0.745656 | 0.746148 |

The combined global/local correction limits, clipping, one training seed, known labels and validation selection limit conclusions. The requested 48/128-pixel shifts do not exceed every per-image alignment bound. No causal claim about EMA reset, normalization drift or overfitting is inferred from this study.

Full numeric evidence: [scores](reports/val_shift_scores.json), [per-video SA](reports/val_shift_per_seq.csv), [coverage](reports/coverage_shift.json), [geometry](reports/shift_geometry.csv), and [summary](reports/shift_results_summary.json).
