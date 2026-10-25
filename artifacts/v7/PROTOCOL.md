# V7 registered decision-level fusion and synchronization audit

Registered before any v7 fusion score, replacement breakdown or subset bootstrap.
This is exploratory validation on the same split used for checkpoint selection.
The supplied draft and external prediction-only summaries are retained as source
material; their synchronization interpretations are not accepted as proven facts.

## Scope and pre-registration exception

CPU only, validation only, no training or test access. Immutable inputs are the
completed A0_s0 and A0rgb_s0 seed-0 runs plus the old four-parameter training
theta table. Full coverage is 67 videos and 61,999 frames per condition.

As explicitly requested, the single-video precheck of 20190925_213001_1_1 was
performed before registration: eight evenly spaced primary replacement indices
plus first/last frame, selected using predictions only. Nine unique image pairs
were produced; labels were read for that video to inspect the requested cases.
No gate, confidence, mapping or exclusion was tuned from this inspection.
Previously published v6 results were already known; this is not a blind study.
The precheck images remain private. Original annotations are never repaired here.

## Fixed conditions

Keep the old IR saved output unless its highest candidate confidence is below
0.5 and the RGB highest confidence is at least 0.5. This includes existing IR
outputs with confidence in [0.25, 0.5), not only empty outputs. Reject proposals
with centers outside the IR frame; clip accepted boxes to its extent. Use the
inverse of the normalized IR-to-RGB transform on all four RGB box corners.

| Condition | Mapping | Size | Validation supervision | Role |
|---|---|---|---|---|
| A0 | Original IR prediction | Original | None at inference | Baseline |
| P | Equal mean of train theta matrices with n_pairs > 0 | Mapped box | No val theta or labels read | Primary |
| S | Train mean plus median correction of at most 5 past confident pairs within 30 frames | Most recent confident IR size within 30 frames; otherwise mapped size | Past predictions only | Registered exploratory secondary |
| R | Existing per-video val theta | Mapped box | Label-derived val theta | Reference only |

For S, pair confidence is 0.5 in both modalities. The current frame is proposed
before it is observed. State resets for every video. Missing history falls back
to train mean and mapped size. P/S must not receive --seq-theta. R is not an
oracle upper bound: imperfect boxes and a fitted spatial transform can still fail.

## Primary gate and statistics

Only P can satisfy the primary proposal gate: macro-video SA point difference
P minus A0 >= +0.003 and its 95% paired sequence-bootstrap lower bound > 0.
S/R cannot independently open it. This resolves the draft's optional secondary
gate before scoring, avoiding selection between two unadjusted primary tests.
The 0.003 band is an exploratory task-specific choice, not an accepted standard.
No new training is launched even if the gate is met; report one concrete proposal.
A null result concerns this fixed rule and mapping, not all fusion or RGB utility.

Use the existing SA, penalized-SA and AP50 formulas with IR labels. Average SA
over all 67 videos, rather than pooling their frames. Use 10,000 paired whole-
sequence resamples, seed 0 independently restarted for every comparison, so
--run ordering cannot change intervals. Record this changed draw policy.
The --bootstrap-mode legacy_shared option retains the old stream for reproducing
published results; old results and their original scorer remain unchanged.
Reproduce the v6 first-comparison SA interval before the new analysis.

SA is primary. Penalized SA and AP50 are descriptive secondaries. AP uses the
unchanged candidate-matching formula; replacement frames add the RGB candidate
to the saved IR candidates, using raw confidences without calibration or a new
NMS pass. This candidate-budget/ranking difference limits AP interpretation.
Undefined metrics stay null; missing files, malformed boxes, unequal frame
counts, duplicate subset rows or wrong coordinates raise errors. Compressed
inputs are supported; conflicting plain/gzip contents are refused.

## Mandatory sensitivity and replacement analyses

For P, S and R, report SA with all videos and three separately fixed exclusions:

1. Exclude 20190925_213001_1_1.
2. Exclude the intersection of validation videos and KNOWN_LABEL_ISSUES.
3. Exclude the unchanged 34 prediction-screen candidates supplied in prereg/
   sync_flagged_predictions_val.json, regardless of subsequent label confirmation.

These are descriptive sensitivity analyses, not new primary tests. Report both
included/excluded sequence lists. All three conditions also receive a replaced-
frame breakdown by original IR label state, original output and fused output.
For IR-negative frames use the actual score difference: replacement costs -1
only when the original IR output was empty, and 0 when it was already a false
positive. Do not multiply all negative replacements by -1. Check the sum of
per-video replacement contributions equals the full macro SA change.
The absolute macro-SA effect is bounded by mean_v(replacements_v / frames_v),
not by the pooled RGB-coordinate complementarity gain from v6.

## Synchronization audit and evidence boundary

Reproduce the supplied prediction-only whole-video screen (scale_translation,
lags -15..15, min_pairs 40, confidence 0.5, gain > 1 IR px and ratio > 1.5).
Then run the same predeclared settings on labels, for whole videos and 200-frame
windows. Also report a common-support audit: identical eligible IR indices at
every candidate lag, avoiding improvements obtained by dropping different frames.
Report eligible counts and windows with insufficient or singular spatial fits.
If a held-out trajectory confirmation is added, fit/select lag on the first two
thirds and evaluate its fixed transform/lag on the last third; keep it diagnostic.

Flags are lag-fit candidates, not proof of temporal mismatch, clock drift or
dropped acquisition frames. Constant-velocity motion can be absorbed by a fixed
translation; static/poorly excited tracks cannot identify lag. Annotation delay,
camera motion, detector errors and spatial-model mismatch remain rival causes.
The visual precheck's on-screen clocks have coarse resolution and are not a
validated synchronization ground truth. Do not shift, refit or retrain inputs
after this audit in the current fusion comparison. Training-split synchronization
is not inferred from val or from rms_norm; no train audit is claimed unless run.

## Registration and provenance

Record this commit, source/input SHA256s and actual commands before scoring.
Preserve the original patch privately with its hash; its private external
session URL was removed only from the applied commit message, retaining author
and source changes. Publish English/ASCII code, protocol, reports and complete
derived predictions under the standing authorization. Exclude original labels,
videos, images/overlays, credentials and machine paths. No recurring automation.
