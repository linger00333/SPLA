#!/usr/bin/env python3
"""One entry point for every training run in the D1 plan.

  stage 1 (A1)  two-stream mid-fusion, standard trainer, mosaic on:
      python tools/train_spla.py --stage 1 --data /data/antiuav_yolo/antiuav_rgbt.yaml --seed 0 --name A1_s0
  stage 1 (A0)  IR-only YOLO11 on the IR images:
      python tools/train_spla.py --stage 0 --data /data/antiuav_yolo/antiuav_ir.yaml --seed 0 --name A0_s0
  stage 2 (A4)  SPLA with the sequence prior; (A3) the same network with a zero prior:
      python tools/train_spla.py --stage 2 --prior-mode sequence --pretrained runs/detect/A1_s0/weights/best.pt \
          --data /data/antiuav_yolo/antiuav_rgbt.yaml --meta /data/antiuav_yolo/spla_meta.json \
          --train-theta /data/antiuav_yolo/seq_theta_train.json --seed 0 --name A4_s0

NOT EXECUTED where it was written (no PyTorch there).
"""
from __future__ import annotations

import argparse
import re
import shutil
import sys
from pathlib import Path

KIT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(KIT))


def staged_config(base: str, size: str) -> str:
    """Copy a yaml to a digit-free folder and return its scale-specific name.

    Ultralytics derives the scale (n/s/m/l/x) from the file name with a regex
    applied to the WHOLE path, so a folder such as 'run2s' would be misread.
    """
    safe = Path.home() / ".cache" / "spla_cfg"
    if re.search(r"\d[nslmx]", str(safe)):
        raise SystemExit(f"{safe} contains a digit followed by n/s/m/l/x; set HOME to a plainer path")
    safe.mkdir(parents=True, exist_ok=True)
    src = KIT / "configs" / base
    shutil.copy(src, safe / base)                                  # yolo11-RGBT-SPLA.yaml
    return str(safe / base.replace("yolo11-", f"yolo11{size}-"))  # yolo11n-RGBT-SPLA.yaml


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", type=int, choices=[0, 1, 2], required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--size", default="n", choices=list("nsmlx"))
    ap.add_argument("--project", default=None)
    ap.add_argument("--name", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=None, help="default: 100 (stages 0, 1) / 40 (stage 2)")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--device", default="0")
    ap.add_argument("--fraction", type=float, default=1.0, help="train on a share of the images (smoke tests)")
    ap.add_argument("--pretrained", default=None, help="stage 2: stage-1 best.pt")
    ap.add_argument("--meta", default=None, help="stage 2: spla_meta.json")
    ap.add_argument("--train-theta", default=None, help="stage 2: seq_theta_train.json")
    ap.add_argument("--prior-mode", default="sequence", choices=["sequence", "zero"])
    ap.add_argument("--val-prior", default="dataset", choices=["dataset", "replay", "noisy_oracle", "oracle", "zero"])
    # fine-tuning recipe (stage 2, and the stage-1 continuation control A1c)
    ap.add_argument("--lr0", type=float, default=None, help="default 0.01 from scratch, 0.001 when fine-tuning")
    ap.add_argument("--warmup-epochs", type=float, default=None, help="default 3 from scratch, 0 when fine-tuning")
    ap.add_argument("--cos-lr", action="store_true")
    ap.add_argument("--continue-recipe", action="store_true",
                    help="stage 1 only: continue --pretrained weights with the stage-2 recipe (no mosaic, mild "
                         "scale/shift/flip, same epochs and LR). This is the A1c control, which separates the effect "
                         "of the fine-tuning recipe from the effect of SPLA")
    args = ap.parse_args()
    finetune = args.stage == 2 or args.continue_recipe
    lr0 = args.lr0 if args.lr0 is not None else (0.001 if finetune else 0.01)
    warm = args.warmup_epochs if args.warmup_epochs is not None else (0.0 if finetune else 3.0)

    common = dict(data=args.data, imgsz=args.imgsz, batch=args.batch, workers=args.workers, device=args.device,
                  seed=args.seed, deterministic=True, project=args.project, name=args.name, exist_ok=False, fraction=args.fraction,
                  epochs=args.epochs or (40 if finetune else 100), plots=False,
                  lr0=lr0, warmup_epochs=warm, cos_lr=args.cos_lr,
                  optimizer="SGD" if finetune else "auto")   # 'auto' would pick its own lr and ignore lr0

    if args.stage == 0:
        from ultralytics.models.yolo.detect import DetectionTrainer
        DetectionTrainer(overrides=dict(common, model=f"yolo11{args.size}.yaml", use_simotm="RGB", channels=3)).train()
        return

    if args.stage == 1:
        from ultralytics.models.yolo.detect import DetectionTrainer
        cfg = staged_config("yolo11-RGBT-SPLA-base.yaml", args.size)
        extra = {}
        if args.continue_recipe:
            if not args.pretrained:
                raise SystemExit("--continue-recipe needs --pretrained (the A1 best.pt)")
            # mimic spla_prepare's GPU augmentation (flip 0.5, scale +-0.2, shift +-0.1), no mosaic
            extra = dict(pretrained=args.pretrained, mosaic=0.0, close_mosaic=0, mixup=0.0, copy_paste=0.0,
                         scale=0.2, translate=0.1, fliplr=0.5, degrees=0.0, shear=0.0, perspective=0.0)
        DetectionTrainer(overrides=dict(common, model=cfg, use_simotm="RGBT", channels=4,
                                        pairs_rgb_ir=["visible", "infrared"], **extra)).train()
        return

    for k in ("pretrained", "meta", "train_theta"):
        if not getattr(args, k):
            raise SystemExit(f"stage 2 needs --{k.replace('_', '-')}")
    from spla.ultralytics_hooks import DEFAULT_SPLA_CFG, SPLATrainer
    SPLATrainer.spla_cfg = dict(DEFAULT_SPLA_CFG, meta=args.meta, train_theta=args.train_theta,
                                prior_mode=args.prior_mode, val_prior=args.val_prior)
    cfg = staged_config("yolo11-RGBT-SPLA.yaml", args.size)
    SPLATrainer(overrides=dict(
        common, model=cfg, pretrained=args.pretrained, use_simotm="RGBT",
        channels=6,               # only the validator's warm-up uses it; the loader reads 4 channels
        pairs_rgb_ir=["visible", "infrared"],
        mosaic=0.0, mixup=0.0, copy_paste=0.0, degrees=0.0, translate=0.0, scale=0.0, shear=0.0,
        perspective=0.0, fliplr=0.0, flipud=0.0, multi_scale=False, close_mosaic=0)).train()


if __name__ == "__main__":
    main()
