# V7 decision-level fusion and data audit

All registered CPU-only validation jobs are complete. No training, GPU inference or test evaluation was run. P, S and R each cover all 67 videos and 61,999 frames. Original data, labels, old experiments and weights were not changed.

## Main result

The primary gate is not met. P only: macro SA delta >= +0.003 and 95% lower bound > 0. No follow-up training is launched. S is a registered exploratory secondary and R is a label-assisted reference; neither can independently open the primary gate.

| Condition | SA | Penalized SA | AP50 |
|---|---:|---:|---:|
| A0 | 0.815417 | 0.799854 | 0.990056 |
| P | 0.811181 | 0.803467 | 0.979283 |
| S | 0.811576 | 0.803949 | 0.980204 |
| R | 0.812715 | 0.804992 | 0.982621 |

Point differences from A0 and 95% paired whole-sequence intervals, 10,000 draws. Seed 0 restarts independently for each comparison. Only P/SA is primary; other intervals are unadjusted descriptive diagnostics.

| Condition minus A0 | SA delta [CI] | Penalized SA delta [CI] | AP50 delta [CI] |
|---|---|---|---|
| P | -0.004237 [-0.010231, -0.000673] | +0.003613 [+0.000139, +0.007270] | -0.010774 [-0.023997, -0.002578] |
| S | -0.003841 [-0.009835, -0.000335] | +0.004095 [+0.000392, +0.008029] | -0.009852 [-0.022941, -0.002010] |
| R | -0.002702 [-0.006424, -0.000303] | +0.005138 [+0.001966, +0.008756] | -0.007435 [-0.015113, -0.002160] |

The penalized metric and SA can move in opposite directions: the missing-output penalty rewards filling a positive frame even when localization remains poor. AP also has an added-candidate and uncalibrated cross-detector confidence limitation. These secondary changes do not override the declared SA gate.

## Replacement costs and mapping

| Condition | Replacements | IR-positive replacements | IR-negative replacements | Actual negative-frame score change | Absolute macro-SA bound |
|---|---:|---:|---:|---:|---:|
| P | 633 | 373 | 260 | -257 | 0.009864 |
| S | 647 | 378 | 269 | -264 | 0.010158 |
| R | 633 | 373 | 260 | -258 | 0.009875 |

A negative-label replacement costs -1 only if the original output was empty. If it already contained a false positive, the added SA cost is 0. The full per-video score change is reconstructed exactly from replacement contributions for all three conditions. Bounds are macro-video bounds for this fixed rule, not the pooled RGB-coordinate oracle from v6.

The requested 20190925_213001_1_1 precheck inspected nine selected pairs. Early IR samples are visibly blurred, while RGB shows target lights. Some modalities have different official presence labels. This is a visibility/annotation concern, not permission to repair IR labels using RGB. Images and frame-level label states remain private.

| Condition | Replacements in requested video | Its SA change |
|---|---:|---:|
| P | 349 | -0.181033 |
| S | 349 | -0.180855 |
| R | 349 | -0.104762 |

Prediction-only proposal agreement with a confident IR box was replayed within 1e-12 of the supplied material. It is not annotation IoU. S agreement is measured in a regime with confident paired history, whereas many actual replacement frames fall back to the train mean.

| Condition | Confident-IR comparison frames | Proposal agreement IoU >= 0.5 |
|---|---:|---:|
| P | 57531 | 25.9182% |
| S | 57531 | 82.0167% |
| R | 57531 | 58.7370% |

S uses online mapping for 289 actual replacements and train-mean fallback for 358; recent IR size is available for 319 and mapped-size fallback is used for 328. Conditional agreement on confident IR frames therefore overstates transfer quality at the replacement operating point.

## Predeclared sensitivities

| Exclusion | Videos retained | P SA delta [CI] | S SA delta [CI] | R SA delta [CI] |
|---|---:|---|---|---|
| without_213001 | 66 | -0.001558 [-0.002942, -0.000451] | -0.001159 [-0.002490, -0.000091] | -0.001156 [-0.002543, -0.000073] |
| without_known_issues | 64 | -0.004417 [-0.011052, -0.000729] | -0.003942 [-0.010469, -0.000271] | -0.002768 [-0.006869, -0.000285] |
| without_prediction_sync_candidates | 33 | -0.007598 [-0.019415, -0.000644] | -0.006932 [-0.018769, -0.000038] | -0.004584 [-0.011840, +0.000023] |

Exclusions and their lists were fixed before scoring; these subset intervals cannot replace the primary all-video comparison. The synchronization subset uses the original 34 prediction-screen candidates, not a later favorable label-defined subset.

## Temporal audit: candidates, not confirmed acquisition defects

| Trajectory source and support | Candidates | Estimable / all videos |
|---|---:|---:|
| predictions_whole | 34 | 66 / 67 |
| predictions_windows200 | 40 | 66 / 67 |
| predictions_common | 31 | 65 / 67 |
| labels_whole | 26 | 66 / 67 |
| labels_windows200 | 38 | 66 / 67 |
| labels_common | 24 | 65 / 67 |
| predictions_heldout | 22 | 62 / 67 |
| labels_heldout | 17 | 63 / 67 |

The supplied prediction screen (34 candidates) is reproduced. Label confirmation, common-support comparison and held-out evaluation do not support simply reporting half of the dataset as confirmed desynchronized. Held-out selection uses the first two thirds, a 15-frame boundary gap, and evaluates fixed transforms/lags on the final third.

For 20190926_183941_1_5, the whole-label fit selects +4 frames; for 20190925_131530_1_3 it selects +7. The detailed JSON files report zero-lag and selected-lag residuals and window estimates. Constant-velocity motion can be absorbed by translation; low excitation, annotation delay, detector errors and spatial-model mismatch remain rival explanations.

Encoded timing was checked separately in the two examples and 20190926_141816_1_6. The media audit reports presentation timestamps, not synchronized sensor capture clocks. Regular encoded PTS cannot establish or exclude source-level frame loss. No temporal shifts were fed into the fusion comparison, and the training split was not audited here.

## Geometry and A1 interpretation

The additional post-registration explanatory label review covers 57982 both-positive frames. Identity center mismatch has pooled median 45.060 IR px, IQR [33.219, 58.446]. A1 receives resized, unregistered inputs; it is inaccurate to describe its blanking or shift tests as a direct measurement of perfectly aligned RGB utility.

Even a label-derived per-video mapping of RGB GT boxes gives median IR GT IoU 0.546419, with 57.2057% at IoU >= 0.5. Median IR height divided by mapped RGB height is 1.1991, while the width ratio is 0.9960. These are label-geometry diagnostics with additional supervision, not deployable fusion performance.

## Repository fixes and evidence limits

The scorer now refuses missing/truncated/invalid predictions and RGB-coordinate inputs, reads gzip artifacts, and supports order-invariant bootstrap draws. Legacy shared-stream draws remain available. Subset CSV omissions/duplicates and malformed fusion lengths are errors. The previously published v6 first SA interval was reproduced exactly. Old reports are preserved; their separate full-coverage audits protected them from the newly identified silent-input failure mode.

The supplied author is preserved. Only the private external session URL was removed from the applied commit message; original and application hashes are recorded. Full source/input hashes and commands, 32 actual-runtime tests and 12 supplemental local tests (three held-out, four I/O, five reporting-gate checks) are retained. Test logs include an expected invalid-video fixture error; scoring warnings refer to undefined mechanism diagnostics, not missing SA or AP results. No raw labels, images or videos are published.

## Concrete next step (proposed, not executed)

First verify timing/visibility and spatial identifiability on a small fixed set: the +4-frame example, the +7-frame example, the window-drift example and the requested blurred-IR video. Retain original labels, distinguish acquisition timing from annotation or motion effects, and use a calibration/held-out split for any corrected mapping. Only after that audit should a separately versioned train-derived calibration or confidence/box-transfer experiment be specified. This run does not justify automatically launching modality dropout, attention, cross-scale fusion or the old failed-pilot queue.

A failure of this fixed confidence-gated rule does not establish that RGB is uninformative or that all fusion methods are redundant. Single detector seeds, selected validation checkpoints, conditional proposal diagnostics, possible timing mismatch and label visibility differences limit generalization.

Evidence: [registered protocol](PROTOCOL.md), [full scores](reports/fusion_val.json), [per-video scores](reports/fusion_val_per_seq.csv), [replacement breakdowns](reports/breakdown_P.json), [held-out audit](reports/sync_heldout.json), [geometry](reports/geometry_review.json), [media timing](reports/media_timing_audit.json), [frozen sources and inputs](records/frozen_manifest.json).
