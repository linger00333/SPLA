# V6 fusion diagnostics: frozen validation protocol

Registered before observing any new ablation or RGB-only model scores.

## Scope and ordering

1. Run the old A1 seed-0 checkpoint on all validation sequences with RGB filled with 114, and with IR filled with 114. The 114 value is the existing letterbox fill. Compare to the unchanged old A1 output using the unchanged score_runs.py (SA, penalized SA, AP50; 10,000 paired sequence bootstrap draws, seed 0). Audit all 67 sequences and 61,999 frames before scoring. Prior is zero; retain the old four-parameter scale_translation tables, thresholds and preprocessing. No test evaluation.
2. Build a separate RGB-only dataset from exactly the old train/val sampled image lists. Use original visible annotations normalized by actual original RGB video dimensions. Reuse stretched RGB images by per-file hard links, with copies as a portable fallback; never use directory or file symlinks for the new images. Never modify shared images, old labels or conversions. Require metadata/image equality and exact annotation indices. Audit the actual Ultralytics loader and inspect real validation box overlays before training.
3. Train exactly one A0rgb_s0 scratch YOLO11n model: seed 0, 640, batch 16, workers 4, 100 epochs, original A0 scratch recipe and validation checkpoint selection, physical GPU 1 only. No pretrained checkpoint, no extra training searches. The original sampling was based on IR existence; retain and disclose this limitation. This evaluates the existing stretched RGB pipeline, not native-resolution RGB potential.
4. Run RGB-only full validation inference. Saved boxes are in original RGB pixels. Do not apply the old IR-label scorer to these predictions. Compare with old A0 IR predictions using each modality's own labels through complementarity.py.

## Fixed-detector complementarity

- Primary set: all frames where valid positive annotations exist in both modalities, with original frame indices. Missing prediction files, unequal label/prediction lengths or wrong coordinate metadata are errors, never exclusions or truncation.
- Success: the saved highest-confidence box has IoU >= 0.5 with the corresponding modality's own xywh box. Output confidence threshold 0.25, NMS candidate threshold 0.001, NMS IoU 0.6 and max_det 10 remain unchanged.
- Count both success, IR only, RGB only and neither. Primary summaries pool counts over the 67 validation sequences. Report the full-frame coverage and both-positive denominator explicitly. This conditional metric excludes absent-target behavior; the blank-input official-style SA remains a separate metric.
- oracle_gain = RGB-only successes / all both-positive frames. rescue_share = RGB-only successes / IR failures on that set. An empty denominator is null, never zero. The oracle selects between the outputs of these two fixed detectors. It is not an upper bound on all fusion methods or a measure of IR-coordinate performance improvement.
- Uncertainty: 10,000 paired bootstrap resamples of entire sequences with replacement, seed 0, recompute pooled count ratios per draw; percentile 95% intervals. Report valid draw counts for undefined ratios. Single training seed; no seed robustness claim and no multiple-comparison correction.
- Original annotations remain unchanged. Disclose known release issues from KNOWN_LABEL_ISSUES. Primary includes all sequences; report a sensitivity summary excluding affected sequences. No manual label repair or favorable subgroup selection.
- Exploratory decision bands set in advance: oracle_gain < 0.005 is small evidence of complementarity between these fixed detectors; >= 0.02 is a potentially useful signal; intermediate values are inconclusive. Inspect uncertainty and known-issue sensitivity before making a recommendation. These are task-specific heuristics, not accepted standards. Even a small gain does not prove RGB has no information or all fusion designs are futile.
- For blanking, an absolute SA change < 0.003 is a descriptive small-change band only. Blanking is an out-of-distribution intervention; its change is not a strict causal contribution bound. Shift tolerance does not prove RGB is ignored.

## Follow-up gate

Only after the complete complementarity analysis, propose a concrete next experiment if the evidence supports it. Do not automatically train modality dropout, attention, cross-scale fusion, A1c/A3/A4, or new seeds. The previous failed-pilot gate remains effective for that separate queue. EMA, batch-normalization drift, optimization and modality laziness remain possible explanations rather than established causes.

## Reproducibility and publication

Keep original patch/hash, source-only extraction and hardened source changes, frozen code/input/weight hashes, actual commands, full predictions, coverage and failure logs. English/ASCII source, reports, configurations, file names, commit messages and portable model metadata for GitHub; exclude raw videos/images/annotations, credentials and machine paths. Full derived predictions are authorized. Run publication checks and compare actual Git blobs with the manifest. No recurring automation.
