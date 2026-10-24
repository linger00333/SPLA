# Installation and reproducibility

Use an isolated Python 3.10 environment. Install the appropriate PyTorch and
Torchvision builds for your driver, then install `requirements.txt`, `kit`,
and `framework` in the order shown in the main README. The recorded CUDA build
was PyTorch 2.6.0+cu118. Other builds have not been validated for this snapshot.

```bash
python -m pip install --index-url https://download.pytorch.org/whl/cu118 torch==2.6.0 torchvision==0.21.0
python -m pip install -r requirements.txt
python -m pip install --no-deps -e kit
python -m pip install --no-deps -e framework
python kit/tools/patch_ultralytics.py framework
```

The bundled runtime is required: a stock `pip install ultralytics` does not
provide the same RGB-T loader or parser. Confirm that `ultralytics.__file__`
points to this repository's `framework/ultralytics` directory.

The source experiment found mixed NumPy files in a shared environment. A clean,
private NumPy 1.26.4 installation resolved a PyTorch/NumPy interoperability
failure. Do not copy system environments or overwrite another active environment.

## Data

Obtain the official Anti-UAV RGB-T data separately and retain its official split.
No raw dataset frames, videos, or original annotations are redistributed here.
Never split neighboring frames randomly across training and validation.

```bash
python kit/tools/convert_antiuav.py --root /data/Anti-UAV-RGBT --out /data/antiuav_yolo --model scale_translation --splits train val
```

The original completed legacy models use `scale_translation`, not `zoom`.
The converter's current default is `zoom`; specify the model explicitly when
reproducing the legacy experiment. Conversion extracts every fifth target frame
and every fiftieth negative frame by default; video evaluation uses every decoded
paired frame. The training conversion has 29,708 training and 12,158 validation
images. Metadata for validation and training are merged without test metadata.

Do not use the last two fit tables in `artifacts/prior_scale_translation` as
training labels. They are the frozen prior parameters required to reproduce
the recorded legacy inference setup; generate image metadata from your own
authorized dataset copy.

## Train explicitly

```bash
python kit/tools/train_spla.py --stage 1 --data /data/antiuav_yolo/antiuav_rgbt.yaml --seed 0 --workers 4 --device 0 --project runs/detect --name A1_s0
python kit/tools/train_spla.py --stage 1 --continue-recipe --pretrained runs/detect/A1_s0/weights/best.pt --data /data/antiuav_yolo/antiuav_rgbt.yaml --seed 0 --workers 4 --device 0 --project runs/detect --name A1c_s0
```

The first command trains 100 epochs. The continuation control trains 40 epochs
with SGD, lr0=0.0001, lrf=1.0 (constant LR), warmup=0, and no mosaic.
Stage 2 uses the same fine-tuning
learning-rate defaults, with its own geometry augmentation and alignment loss.
The continuation control does not have an SPLA layer.

For the 10-epoch retention pilot, use the audited wrapper instead of the plain
training entry point, adding `--epochs 10` and a fresh name:

```bash
python kit/tools/run_retention_pilot.py --stage 1 --continue-recipe --pretrained runs/detect/A1_s0/weights/best.pt --data /data/antiuav_yolo/antiuav_rgbt.yaml --epochs 10 --seed 0 --workers 4 --device 0 --project runs/detect --name A1c_lr1e-4_s0
python kit/tools/check_retention_pilot.py --run runs/detect/A1c_lr1e-4_s0 --out reports/A1c_lr1e-4_s0_gate.json
```

The wrapper records the matched pre-training validation and runtime recipe.
If the pilot fails the predeclared gate, restart from the original A1
checkpoint with `--freeze 21`; do not continue the failed pilot checkpoint.
`--freeze 21` freezes both backbones' parameters (layers 0-20); it does not
explicitly freeze batch-normalization running statistics. `--lrf` overrides the
final-to-initial learning-rate ratio. From-scratch defaults remain lr0=0.01,
lrf=0.01 and three warmup epochs. Complete the pilot before starting matched
40-epoch A1c/A3/A4 experiments.

PyTorch 2.6 defaults to restricted loading of pickle checkpoints. For your own
trusted, locally produced `.pt` training checkpoints, set
`TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1` before running continuation/stage-2 training.
Do not enable that option for untrusted checkpoints. Public Safetensors inference
does not require it.

Legacy A3/A4 were trained with the older recipe. Running the current stage-2
entry point is a new experiment and must not be represented as reproducing
the legacy weights exactly. See [experiment scope](EXPERIMENTS.md).
