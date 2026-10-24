"""Glue between SPLA and the YOLOv11-RGBT code base (an Ultralytics fork).

NOT EXECUTED IN THE ENVIRONMENT WHERE THIS FILE WAS WRITTEN. Read it as a
worked plan with real class and method names from YOLOv11-RGBT (commit
9cc2e208, 15 Dec 2025), and run it step by step as described in the README.

What it adds
------------
SPLADataset      YOLODataset that attaches per-image metadata (sequence id,
                 sequence transform theta, IR and RGB target centres).
spla_prepare     GPU augmentation that keeps the geometry exact (joint flip,
                 scale and shift; RGB-only misalignment jitter), then builds
                 the prior field and appends it as input channels 4-5, and
                 builds alignment targets.
SPLADetectionModel  DetectionModel whose criterion adds the alignment loss.
SPLATrainer / SPLAValidator  trainer and validator that use the above.

Training recipe (two stages)
----------------------------
Stage 1: train the plain two-stream model (yolo11-RGBT-midfusion.yaml) with
         the standard trainer and its usual augmentation, mosaic included.
Stage 2: build yolo11-RGBT-SPLA.yaml, load the stage-1 weights (all layers
         with matching names and shapes), and train with SPLATrainer.
         Ultralytics' geometric augmentation and mosaic MUST be off in stage
         2 (mosaic=0, degrees=0, translate=0, scale=0, shear=0,
         perspective=0, fliplr=0); spla_prepare does the geometric
         augmentation itself so that the transform stays exact.
"""
from __future__ import annotations

import json
import math
import random
import zlib
from copy import copy
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from ultralytics.data.dataset import YOLODataset
from ultralytics.models.yolo.detect import DetectionTrainer, DetectionValidator
from ultralytics.nn.tasks import DetectionModel
from ultralytics.utils import RANK, colorstr
from ultralytics.utils.loss import v8DetectionLoss
from ultralytics.utils.torch_utils import de_parallel

from .geometry import canvas_matrix, compose_augmentation, params_to_theta
from .modules import GlobalOffsetHead, SPLA, SPLAAlignLoss, prior_field_from_theta

DEFAULT_SPLA_CFG = dict(
    meta="spla_meta.json",       # written by tools/convert_antiuav.py
    prior_mode="sequence",       # "sequence" (A4, SPLA) or "zero" (A3, per-image control)
    prior_noise_px=2.0,          # std of the prior's translation error, canvas px
    prior_scale_noise=0.01,      # std of the prior's scale error
    prior_drop=0.15,             # share of images trained with no prior ("not converged yet")
    prior_outlier=0.05,          # share of images with a grossly wrong prior
    prior_outlier_px=20.0,
    jitter_px=8.0,               # RGB-only synthetic misalignment, canvas px (uniform)
    jitter_scale=0.03,           # RGB-only scale jitter
    flip=0.5, joint_scale=0.2, joint_translate=0.1,
    align_weight=0.5,
    # Prior used by the validator (and so for picking best.pt). Options:
    #   "dataset"      training-set mean transform; uses no validation labels (default)
    #   "replay"       online prior replayed on val labels (tune_prior.py --write-meta):
    #                  what a perfect detector would give the filter; a bound
    #   "noisy_oracle" val-label fit + fixed noise; mimics a converged online prior
    #   "oracle"       val-label fit; an UPPER BOUND, never for model selection
    #   "zero"         identity
    # The honest number is the online prior in tools/run_sequences.py.
    val_prior="dataset",
    train_theta="seq_theta_train.json",   # written by tools/convert_antiuav.py
)


# --------------------------------------------------------------------------
# Dataset
# --------------------------------------------------------------------------
class SPLADataset(YOLODataset):
    """YOLODataset plus per-image SPLA metadata read from a JSON sidecar.

    The sidecar maps an image stem to
      {"seq": str, "frame": int, "img_wh": [W, H],
       "theta_seq": 2x3 IR->RGB transform (normalised coords; the oracle
                    per-sequence fit from the training audit),
       "ir_ctr": [[x, y], ...], "rgb_ctr": [[x, y], ...]}   (normalised)
    """

    def __init__(self, *args, meta_path: str = "spla_meta.json", **kwargs):
        with open(meta_path) as f:
            self.spla_meta = json.load(f)
        super().__init__(*args, **kwargs)

    def update_labels_info(self, label):
        label = super().update_labels_info(label)
        m = self.spla_meta.get(Path(label["im_file"]).stem)
        if m is None:
            raise KeyError(f"no SPLA metadata for {label['im_file']}")
        label["spla_theta"] = np.asarray(m["theta_seq"], np.float32)
        # causal replayed online prior (tools/tune_prior.py --write-meta); falls back to the frozen fit
        label["spla_theta_prior"] = np.asarray(m.get("theta_prior", m["theta_seq"]), np.float32)
        label["spla_ir_ctr"] = np.asarray(m["ir_ctr"], np.float32).reshape(-1, 2)
        label["spla_rgb_ctr"] = np.asarray(m["rgb_ctr"], np.float32).reshape(-1, 2)
        label["spla_wh"] = np.asarray(m["img_wh"], np.float32)
        label["spla_seq"] = m["seq"]
        return label


# --------------------------------------------------------------------------
# Batch preparation: exact geometric augmentation + prior field + targets
# --------------------------------------------------------------------------
def _affine_warp(img: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
    """Warp images (B,C,H,W) so that canvas pixel u moves to m u (m: (B,2,3), px)."""
    b, _, h, w = img.shape
    m = m.float()                                            # linalg.inv has no fp16 kernel
    m3 = torch.cat([m, torch.tensor([0, 0, 1.0], device=m.device).expand(b, 1, 3)], 1)
    inv = torch.linalg.inv(m3)                               # output px -> input px
    # convert to the normalised convention of affine_grid (align_corners=False):
    # pixel edges at 0 and w, pixel centres at i + 0.5, matching cell_centres()
    s = torch.tensor([[2.0 / w, 0, -1], [0, 2.0 / h, -1], [0, 0, 1]], device=m.device)
    theta = s @ inv @ torch.linalg.inv(s)
    grid = F.affine_grid(theta[:, :2], [b, img.shape[1], h, w], align_corners=False)
    out = F.grid_sample(img.float(), grid, mode="bilinear", padding_mode="zeros", align_corners=False)
    return out.to(img.dtype)


def _apply(m: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """Apply a 2x3 or 3x3 affine map to points (N, 2)."""
    m = np.asarray(m)[:2]
    return np.asarray(pts).reshape(-1, 2) @ m[:, :2].T + m[:, 2]


def _perturb(th: np.ndarray, rng, noise_px: float, scale_noise: float, wc: int, hc: int) -> np.ndarray:
    """Canvas transform plus a scale error about the canvas centre and a translation error."""
    k = 1 + rng.normal(0, scale_noise)
    e = np.array([[k, 0, (1 - k) * wc / 2], [0, k, (1 - k) * hc / 2], [0, 0, 1]])
    out = (e @ np.vstack([th, [0, 0, 1]]))[:2]
    out[:, 2] += rng.normal(0, noise_px, 2)
    return out


def spla_prepare(batch: dict, imgsz: int, cfg: dict, train: bool = True) -> dict:
    """Augment geometry exactly, append the prior field, and build alignment targets.

    Expects batch["img"] (B, 4, S, S) float in [0, 1] on device, normalised
    xywh batch["bboxes"] on the canvas, batch["batch_idx"], and the SPLA
    metadata lists from SPLADataset. Returns the batch with a 6-channel image
    and batch["spla_targets"].
    """
    img = batch["img"]
    dev = img.device
    b, _, hc, wc = img.shape
    joint, rgbj, priors = [], [], []
    ir_pts, off_pts, idx, prior_at = [], [], [], []
    boxes = batch["bboxes"].clone()
    for i in range(b):
        w0, h0 = batch["spla_wh"][i]
        lm = canvas_matrix(int(w0), int(h0), imgsz)
        th_true = (lm @ np.vstack([batch["spla_theta"][i], [0, 0, 1]]) @ np.linalg.inv(lm))[:2]
        g = np.array([[1.0, 0, 0], [0, 1.0, 0]])
        r = np.array([[1.0, 0, 0], [0, 1.0, 0]])
        if train:
            s = 1 + random.uniform(-cfg["joint_scale"], cfg["joint_scale"])
            tx = random.uniform(-cfg["joint_translate"], cfg["joint_translate"]) * wc
            ty = random.uniform(-cfg["joint_translate"], cfg["joint_translate"]) * hc
            cx, cy = wc / 2, hc / 2
            g = np.array([[s, 0, cx - s * cx + tx], [0, s, cy - s * cy + ty]])
            if random.random() < cfg["flip"]:
                g = np.array([[-1.0, 0, wc], [0, 1.0, 0]]) @ np.vstack([g, [0, 0, 1]])
            js = 1 + random.uniform(-cfg["jitter_scale"], cfg["jitter_scale"])
            jx, jy = (random.uniform(-cfg["jitter_px"], cfg["jitter_px"]) for _ in range(2))
            r = np.array([[js, 0, cx - js * cx + jx], [0, js, cy - js * cy + jy]])
        th = compose_augmentation(th_true, g, r)               # true transform after augmentation
        # the prior handed to the network
        ident = np.array([[1.0, 0, 0], [0, 1.0, 0]])
        if train:
            # training: the replayed online prior (what inference will hold at this frame),
            # moved with the same augmentation, plus detector-like noise (labels are allowed here)
            th_rp = (lm @ np.vstack([batch["spla_theta_prior"][i], [0, 0, 1]]) @ np.linalg.inv(lm))[:2]
            th_prior = compose_augmentation(th_rp, g, r)
            if cfg["prior_mode"] == "zero" or random.random() < cfg["prior_drop"]:
                th_prior = ident
            else:
                noise_px = cfg["prior_outlier_px"] if random.random() < cfg["prior_outlier"] else cfg["prior_noise_px"]
                # np.random (not a fresh Generator) so Ultralytics' seed makes runs repeatable
                th_prior = _perturb(th_prior, np.random, noise_px, cfg["prior_scale_noise"], wc, hc)
        else:
            # validation: by default NO ground truth reaches the prior (see SPLAValidator)
            mode = "zero" if cfg["prior_mode"] == "zero" else cfg.get("val_prior", "dataset")
            if mode == "zero":
                th_prior = ident
            elif mode == "dataset":
                th_ds = (lm @ np.vstack([cfg["_dataset_theta"], [0, 0, 1]]) @ np.linalg.inv(lm))[:2]
                th_prior = compose_augmentation(th_ds, g, r)
            elif mode == "replay":
                th_rp = (lm @ np.vstack([batch["spla_theta_prior"][i], [0, 0, 1]]) @ np.linalg.inv(lm))[:2]
                th_prior = compose_augmentation(th_rp, g, r)
            elif mode == "noisy_oracle":
                # deterministic per sequence: a stand-in for a converged online estimate
                rng = np.random.default_rng(zlib.crc32(str(batch["spla_seq"][i]).encode()))
                th_prior = _perturb(th, rng, cfg["prior_noise_px"], cfg["prior_scale_noise"], wc, hc)
            elif mode == "oracle":
                th_prior = th.copy()                           # upper bound only: uses val labels
            else:
                raise ValueError(f"unknown val_prior {mode!r}")
        joint.append(g)
        rgbj.append(r)
        priors.append(th_prior)
        # targets: IR centre and true per-frame offset on the augmented canvas
        ir_c = _apply(g, _apply(lm, batch["spla_ir_ctr"][i]))
        rgb_c = _apply(r, _apply(g, _apply(lm, batch["spla_rgb_ctr"][i])))
        inside = (ir_c[:, 0] > 0) & (ir_c[:, 0] < wc) & (ir_c[:, 1] > 0) & (ir_c[:, 1] < hc)
        ir_c, rgb_c = ir_c[inside], rgb_c[inside]
        ir_pts.append(ir_c)
        off_pts.append(rgb_c - ir_c)
        prior_at.append(_apply(th_prior, ir_c) - ir_c)
        idx.append(np.full(len(ir_c), i))
        # move this image's boxes with the joint transform (axis-aligned: scale, shift, flip)
        sel = batch["batch_idx"] == i
        if sel.any():
            bx = boxes[sel].cpu().numpy() * np.array([wc, hc, wc, hc])
            c = _apply(g, bx[:, :2])
            sc = abs(g[0, 0])
            bx[:, :2], bx[:, 2:] = c, bx[:, 2:] * sc
            boxes[sel] = torch.as_tensor(bx / np.array([wc, hc, wc, hc]), dtype=boxes.dtype, device=boxes.device)
    if train:
        gj = torch.as_tensor(np.stack(joint), dtype=torch.float32, device=dev)
        rj = torch.as_tensor(np.stack(rgbj), dtype=torch.float32, device=dev)
        img = _affine_warp(img, gj)                               # both modalities
        img = torch.cat([_affine_warp(img[:, :3], rj), img[:, 3:4]], 1)  # RGB only
    thp = torch.as_tensor(np.stack(priors), dtype=torch.float32, device=dev)
    prior = prior_field_from_theta(thp, hc, wc, stride=1).to(img.dtype)  # (B, 2, S, S) canvas px
    batch["img"] = torch.cat([img, prior], 1)
    # drop boxes whose centre left the canvas
    keep = (boxes[:, 0] > 0) & (boxes[:, 0] < 1) & (boxes[:, 1] > 0) & (boxes[:, 1] < 1)
    for k in ("bboxes", "cls", "batch_idx"):
        batch[k] = (boxes if k == "bboxes" else batch[k])[keep]
    batch["spla_targets"] = {
        "ir_ctr_px": torch.as_tensor(np.concatenate(ir_pts), dtype=img.dtype, device=dev),
        "off_px": torch.as_tensor(np.concatenate(off_pts), dtype=img.dtype, device=dev),
        "prior_px_at_target": torch.as_tensor(np.concatenate(prior_at), dtype=img.dtype, device=dev),
        "img_idx": torch.as_tensor(np.concatenate(idx), device=dev),
    }
    return batch


# --------------------------------------------------------------------------
# Model and loss
# --------------------------------------------------------------------------
class SPLACriterion:
    def __init__(self, model, align_weight=0.5):
        self.det = v8DetectionLoss(model)
        self.align = SPLAAlignLoss()
        self.w = align_weight
        self.spla = [m for m in model.modules() if isinstance(m, SPLA)]
        self.glob = [m for m in model.modules() if isinstance(m, GlobalOffsetHead)]

    def __call__(self, preds, batch):
        loss, items = self.det(preds, batch)
        bs = batch["img"].shape[0]
        la = self.align(self.spla, self.glob, batch["spla_targets"]) if "spla_targets" in batch else \
            torch.zeros((), device=items.device)
        # v8DetectionLoss returns either a scalar or a (3,) vector times the batch
        # size depending on the Ultralytics version; summing covers both.
        return loss.sum() + self.w * la * bs, torch.cat([items, la.detach().view(1)])


class SPLADetectionModel(DetectionModel):
    align_weight = 0.5

    def init_criterion(self):
        return SPLACriterion(self, self.align_weight)


_DATASET_THETA_CACHE = {}


def load_dataset_theta(path: str) -> np.ndarray:
    """Training-set mean transform (normalised 2x3) from seq_theta_train.json."""
    if path not in _DATASET_THETA_CACHE:
        info = json.loads(Path(path).read_text())
        rows = [v for v in info.values() if v.get("n_pairs", 0) > 0]
        if not rows:
            raise ValueError(f"{path}: no sequence with RGB-IR pairs")
        # the mean of the stored 2x3 matrices equals theta(mean params) for every
        # model in spla.geometry (all are linear in their parameters), and needs
        # no model-specific state such as the zoom aspect factor
        _DATASET_THETA_CACHE[path] = np.mean([np.asarray(v["theta"], float) for v in rows], 0)
    return _DATASET_THETA_CACHE[path]


class SPLAValidator(DetectionValidator):
    spla_cfg = DEFAULT_SPLA_CFG

    def preprocess(self, batch):
        batch = super().preprocess(batch)
        cfg = dict(self.spla_cfg)
        if cfg["prior_mode"] != "zero" and cfg.get("val_prior", "dataset") == "dataset":
            cfg["_dataset_theta"] = load_dataset_theta(cfg["train_theta"])
        # The validator sees frames out of video context, so it cannot run the
        # online prior; the online number comes from tools/run_sequences.py.
        return spla_prepare(batch, self.args.imgsz, cfg, train=False)


class SPLATrainer(DetectionTrainer):
    spla_cfg = DEFAULT_SPLA_CFG

    def build_dataset(self, img_path, mode="train", batch=None):
        gs = max(int(de_parallel(self.model).stride.max() if self.model else 0), 32)
        a = copy(self.args)
        a.channels = 4  # loader reads RGB+IR; spla_prepare appends prior
        return SPLADataset(
            meta_path=self.spla_cfg["meta"], img_path=img_path, imgsz=a.imgsz, batch_size=batch,
            augment=mode == "train", hyp=a, rect=False, cache=a.cache or None,
            single_cls=a.single_cls or False, stride=gs, pad=0.0, prefix=colorstr(f"{mode}: "),
            task=a.task, classes=a.classes, data=self.data,
            fraction=a.fraction if mode == "train" else 1.0,
            use_simotm=a.use_simotm, pairs_rgb_ir=a.pairs_rgb_ir)

    def preprocess_batch(self, batch):
        assert not self.args.multi_scale, "multi_scale breaks the exact canvas geometry"
        batch = super().preprocess_batch(batch)
        for k in ("bboxes", "cls", "batch_idx"):
            batch[k] = batch[k].to(self.device)
        return spla_prepare(batch, self.args.imgsz, self.spla_cfg, train=True)

    def get_model(self, cfg=None, weights=None, verbose=True):
        SPLADetectionModel.align_weight = self.spla_cfg["align_weight"]
        model = SPLADetectionModel(cfg, nc=self.data["nc"], verbose=verbose and RANK == -1)
        if weights:
            model.load(weights)          # loads every layer whose name and shape match
        return model

    def get_validator(self):
        self.loss_names = "box_loss", "cls_loss", "dfl_loss", "align_loss"
        SPLAValidator.spla_cfg = self.spla_cfg
        return SPLAValidator(self.test_loader, save_dir=self.save_dir, args=copy(self.args),
                             _callbacks=self.callbacks)
