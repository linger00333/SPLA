# Portable model bundles

Four completed legacy seed-0 checkpoints are supplied under `models/legacy`.
Each bundle has a tensor-only `model.safetensors` file and an ASCII `config.json`.
The registry records the source checkpoint SHA-256 and exported model hashes.
The bundles do not contain optimizer state or original training checkpoint
objects. Tensor storage is FP16; the loader reconstructs FP32 inference models.

```bash
python kit/tools/run_sequences.py --weights models/legacy/A4_s0/model.safetensors --root /data/Anti-UAV-RGBT --split val --prior online --seq-theta artifacts/prior_scale_translation/seq_theta_val.json --train-theta artifacts/prior_scale_translation/seq_theta_train.json --process-sigma 0.001 --meas-sigma 0.008 --device cuda:0 --out results/val/A4_online_s0
python kit/tools/score_runs.py --root /data/Anti-UAV-RGBT --split val --run A4=results/val/A4_online_s0 --out results/val/A4_scores.json
```

Use `--prior zero` for A0, A1, or A3. For A4 diagnostic ablations, change the
prior to `dataset`, `zero`, or `oracle`, and write each run to a separate directory.
Oracle is annotation-assisted and must not be reported as deployable performance.

The inference loader also supports original trusted `.pt` checkpoints through
the fork's existing loader. Those original pickle checkpoints are not published
here. Never load an untrusted pickle model. Training initialization still uses
the local `.pt` checkpoints produced by training; the public Safetensors bundles
are inference exports rather than training-resumption checkpoints.

The old A1c run is complete and its CSV and predictions are published; its
weights are not part of the four-model portable bundle. Low-LR pilot and
matched three-seed models are not claimed as finished. The zoom plan is
superseded. Check the timestamped status and model registry for each run.
