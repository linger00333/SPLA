# Fusion reliance and fixed-detector complementarity

The v6 study adds two validation ablations of the existing A1 seed-0 detector,
and one independent 100-epoch RGB-only scratch baseline with the old A0 budget.
See the [frozen protocol](../artifacts/v6/PROTOCOL.md). Live execution and
completed results must be distinguished in the timestamped study snapshots.

The supplied patch contained cache binary notices; those were excluded. The
[source-only input](../patches/fusion_diagnostics_supplied_source.patch) and
[hardened changes](../patches/fusion_diagnostics_hardened.patch) are retained.
Hardening prevents missing files, length truncation and undefined ratios from
silently changing the analysis. The RGB-only builder defaults to train/val,
requires the exact old sampled image list and writes separate RGB labels.
There are 29,708 train images and 12,158 val images; all are hard-linked in the
executed environment. The actual framework loader was checked against every
new label. Four real positive validation samples were visually inspected.
The images and overlays remain private. The sampling is inherited from IR
existence, and RGB remains stretched to the old IR grid: this baseline does
not establish the performance of native-resolution RGB.

The updated runner was tested on real video and GPU input tensors. Default
A1 boxes and all candidate detections exactly match the old A1 first five
frames. RGB-input plumbing was tested with old IR weights only as a runtime
check; those smoke predictions are not RGB baseline results. Unit tests and
runtime checks are recorded in the [implementation review](../artifacts/v6/records/implementation_review.json).

## Commands

Use the existing four-parameter prior tables, the old A1/A0 checkpoints, and
the unchanged thresholds. Paths below are portable placeholders. The runner
uses zero prior for these detectors. Both ablations require all 67 validation
videos and 61,999 frames before the unchanged scorer is invoked.

```bash
python kit/tools/run_sequences.py --weights models/legacy/A1_s0/model.safetensors \
  --root data/raw --split val --seq-theta data/converted/seq_theta_val.json \
  --train-theta data/converted/seq_theta_train.json --prior zero \
  --process-sigma 0.001 --meas-sigma 0.008 --device cuda:1 \
  --blank rgb --out results/A1_blankrgb_s0
# Repeat with --blank ir and a separate output directory.

python kit/tools/make_rgb_only.py --root data/raw --source data/converted \
  --out data/rgb_baseline --splits train val
python kit/tools/train_spla.py --stage 0 --data data/rgb_baseline/antiuav_rgbonly.yaml \
  --size n --seed 0 --imgsz 640 --batch 16 --workers 4 --epochs 100 \
  --device 1 --project runs --name A0rgb_s0
python kit/tools/run_sequences.py --weights runs/A0rgb_s0/weights/best.pt \
  --root data/raw --split val --seq-theta data/converted/seq_theta_val.json \
  --train-theta data/converted/seq_theta_train.json --prior zero \
  --input rgb --device cuda:1 --out results/A0rgb_s0
python kit/tools/complementarity.py --root data/raw --split val \
  --ir-run results/A0_s0 --rgb-run results/A0rgb_s0 --iou 0.5 --boot 10000 \
  --out reports/complementarity_val.json
```

RGB predictions retain the historical `_IR.txt` filename convention, but
their values are **original RGB pixel coordinates**, declared in run_config.
Do not score them against the IR labels using score_runs.py. Complementarity
uses each detector's own labels on the intersection of valid positive frames.
IoU success and fixed-output oracle gains are separate from official-style SA.
The oracle is not an upper bound on arbitrary fusion or IR-coordinate gains.
Blanking is out of distribution and cannot establish a strict causal bound.

The executed one-shot controllers are preserved in [study/v6](../study/v6).
They assume the existing experiment workspace layout, validated v3 runtime,
old checkpoints and raw/converted data; they are provenance scripts, not an
automatic installer for a fresh checkout. They wait for step-1 completion,
the full loader audit and visual gate before the single training run.
They never enable recurring automation or the failed-pilot three-seed queue.

No follow-up fusion training is authorized by the complementarity result.
Any recommendation must be based on the completed diagnostics and their
uncertainty, without treating the exploratory 0.5% and 2% bands as standards.
