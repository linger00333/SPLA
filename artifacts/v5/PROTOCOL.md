# Validation RGB shift diagnostics

Authorized scope: no training; all 67 validation videos; existing seed-0 legacy
weights. The seven clean conditions are rescored in one original scorer call
with the optional per-video SA CSV. A1 minus A0 > 0.02 selects videos only for
descriptive inspection; the six stress runs still use every validation video.

Six conditions: A1 zero, A4 online, A4 zero at persistent RGB shifts of 48 and
128 IR pixels. Retain n/640/FP32/batch1, output confidence 0.25, observation
confidence 0.5, NMS confidence 0.001/IoU 0.6/max10, scale_translation prior,
process_sigma=0.001 and meas_sigma=0.008. The runner and frozen prior tables
are byte-identical to the completed clean experiment. No current label enters
the online estimator; labels are used separately for evaluation diagnostics.

Each video has one deterministic random shift direction, identical for all
models and prior modes; the 128-pixel vector is parallel to the 48-pixel vector.
Only RGB is shifted, after stretching it to IR dimensions and before letterbox.
Warping uses zero borders and clips out-of-frame content. Record shifted RGB
target retention alongside scores; the perturbation is not pure registration
error when it removes a target from the RGB image.

The actual model also has a global per-image correction head. At the default
radius 3 it can add up to 96 canvas pixels per axis; local P3/P4/P5 corrections
add 24/48/96. Thus 48 or 128 IR pixels do not necessarily exceed all per-image
correction bounds. Save actual model bounds and per-video vectors rather than
assuming this condition is satisfied.

Report clean and shifted SA, penalized SA, AP50 and alignment diagnostics.
Use 10,000 paired sequence bootstrap resamples for six within-model shift
contrasts and two same-shift A4-online minus A4-zero contrasts. Keep point
deltas distinct from bootstrap means. Sequence intervals do not quantify
training seed uncertainty. Retain all known annotation issues and disclose
coverage; reject incomplete prediction/label lengths before final scoring.

Interpret changes as diagnostics for these weights, data and perturbations.
Small A1 changes do not establish that real misregistration never matters.
An online/zero gap is evidence of prior dependence under this protocol, not
proof of an unrestricted deployment benefit. EMA reset, normalization drift,
overfitting and a need for larger training shifts remain hypotheses. This study
does not authorize retraining or further recipe searches. Test is not evaluated.

Use only GPU 1, at most two inference jobs, and one-shot dispatch. Periodic
automation remains disabled. Preserve original predictions and negative results.
