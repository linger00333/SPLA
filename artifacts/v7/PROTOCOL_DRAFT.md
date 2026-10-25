# V7 decision-level fusion: protocol DRAFT

Status: **draft, not registered.** Rename to `PROTOCOL.md` and commit it before
running `score_runs.py`, `fusion_breakdown.py` or `subset_bootstrap.py` on any
fusion output. No label has been read for anything in this directory.

## Scope

- No training. Validation split only; test is refused by every new tool.
- Inputs: the completed A0_s0 predictions (`artifacts/predictions/v3/A0_s0`, IR
  pixels) and A0rgb_s0 predictions (`artifacts/predictions/v6/A0rgb_s0`,
  original RGB pixels), and the existing four-parameter tables in
  `artifacts/prior_scale_translation`.
- Scoring: the unchanged `kit/tools/score_runs.py` (IR labels; SA, penalized
  SA, AP50), baseline A0, 10,000 paired sequence draws, seed 0, `--per-seq`.
- CPU only; no GPU job is needed.

## Conditions

All conditions share the gates `--ir-conf 0.5 --rgb-conf 0.5 --pair-conf 0.5
--window 30 --online-k 5`: the IR output is replaced only when its top
confidence is below 0.5 (empty or low confidence) and the RGB top confidence is
at least 0.5. Mapped boxes whose centre leaves the IR frame are discarded.

| Name | `--map` | `--size` | Uses val labels | Role |
|---|---|---|---|---|
| P | `train_mean` | `mapped` | no | primary (handoff step 2) |
| S | `online` | `ir_recent` | no | secondary: causal, label-free correction |
| R | `seq` | `mapped` | **yes** | reference only; never decides |

## Decision rule for the follow-up gate

A condition shows a clear gain when its SA difference from A0 has a 95% sequence
interval with lower bound > 0 **and** point difference >= +0.003 (the v6
small-change band). The gate to propose one learned-fusion experiment opens if
P or S shows a clear gain. Two comparisons, unadjusted; both are reported.
Penalized SA and AP50 are secondary and cannot open the gate. R is descriptive.
If neither P nor S shows a clear gain, the write-up path ("complementary but
redundant at decision level") applies, with the caveats below.

## Pre-declared sensitivity analyses (reported whatever the primary result)

1. Without `20190925_213001_1_1`, which receives 349 of P's 633 replacements.
2. Without the videos in `KNOWN_LABEL_ISSUES`.
3. Without the 34 videos flagged as desynchronised in
   `prereg/sync_flagged_predictions_val.json`.
4. `fusion_breakdown.py` for P, S and R: replaced frames split into IR-positive
   (fused IoU >= 0.5 or not) and IR-negative (each costs 1 SA frame unit).

Analyses 1 to 3 use `subset_bootstrap.py` on the `--per-seq` CSV (SA only,
same draw scheme). They describe dependence, not new primary tests.

## Label-free facts recorded before scoring

Computed from the two prediction sets only (`prereg/`):

| | P | S | R |
|---|---:|---:|---:|
| Replaced frames (of 61,999) | 633 | 647 | 633 |
| Transfer fidelity: share IoU >= 0.5 vs confident IR box | 0.259 | 0.820 | 0.587 |
| Median IoU vs confident IR box | 0.257 | 0.905 | 0.560 |

Transfer fidelity is measured on the 57,531 frames where both detectors are
confident (>= 0.5); there IR is almost always right, so it estimates how well a
mapped RGB box can reproduce an IR box. S's figure is optimistic for replaced
frames: 358 of its 647 replacements have no recent pair and fall back to the
training mean (348 of them in `20190925_213001_1_1`).

`sync_audit.py` on the same predictions flags 34 of 67 videos whose IR and RGB
streams fit better with a frame offset (median residual 4.8 px at lag 0 versus
3.1 px at the best lag; up to 57 px to 5 px). This is a data property, not a
result of any fusion condition.

## Interpretation limits

- The maximum number of frames this rule can change is about 1.1% of frames,
  so the SA effect is bounded near +/-0.01 by construction.
- A null P or S does not show RGB is uninformative: IR failures with a
  confident wrong IR box are not reachable by confidence gating, and temporal
  desynchronisation is not compensated.
- Single detector seeds; validation is also the checkpoint-selection split.

## Commands

```bash
P=artifacts/prior_scale_translation
for spec in "P train_mean mapped" "S online ir_recent" "R seq mapped"; do set -- $spec
  python kit/tools/decision_fusion.py --ir-run results/A0_s0 --rgb-run results/A0rgb_s0 \
    --train-theta $P/seq_theta_train.json --seq-theta $P/seq_theta_val.json \
    --map $2 --size $3 --out results/fusion/$1
done
python kit/tools/score_runs.py --root data/raw --split val --run A0=results/A0_s0 \
  --run P=results/fusion/P --run S=results/fusion/S --run R=results/fusion/R \
  --baseline A0 --boot 10000 --per-seq reports/fusion_val_per_seq.csv --out reports/fusion_val.json
python kit/tools/subset_bootstrap.py --per-seq reports/fusion_val_per_seq.csv --a P --b A0 \
  --exclude 20190925_213001_1_1 --label without_213001_1_1 --out reports/fusion_P_wo213001.json
# repeat for S; KNOWN_LABEL_ISSUES and --exclude-file artifacts/v7/prereg/sync_flagged_predictions_val.json
python kit/tools/fusion_breakdown.py --root data/raw --split val --ir-run results/A0_s0 \
  --fused results/fusion/P --out reports/fusion_P_breakdown.json
```

The `results/A0_s0` and `results/A0rgb_s0` paths may also point at the
compressed copies under `artifacts/predictions`; `score_runs.py` needs the
plain A0 files.

## Frozen code (SHA-256)

Filled in at registration with `sha256sum` of `kit/tools/decision_fusion.py`,
`fusion_breakdown.py`, `subset_bootstrap.py`, `sync_audit.py` and the unchanged
`score_runs.py`.
