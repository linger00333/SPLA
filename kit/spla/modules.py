"""PyTorch modules for sequence-prior local alignment (SPLA).

NOT EXECUTED IN THE ENVIRONMENT WHERE THIS FILE WAS WRITTEN (no PyTorch was
available). Run tests/test_modules_torch.py first; it checks the warping
convention, the gradient path and the recovery of a known shift.

Data flow inside the detector (one pyramid level, stride s):

    prior field P (B,2,H,W), canvas px  --avg-pool-->  P_l (B,2,h,w)
    RGB feature F_rgb  --warp by P_l-->  coarse-aligned RGB
    local correlation(IR, coarse RGB) + features --> residual delta (cells),
        bounded by tanh to +-radius, plus a per-cell log scale (uncertainty)
    total offset T = P_l + s * delta  -->  warp F_rgb by T  -->  aligned RGB

Offsets are always "RGB position minus IR position" in canvas pixels, so
sampling the RGB map at (IR cell centre + T) brings RGB content onto the IR
grid. The same convention is used by spla.geometry.offset_field.

The prior reaches the network as two extra input channels (4 and 5) holding
the dense offset field at canvas resolution, so no change to the Ultralytics
forward signature is needed; a SilenceChannel [4, 6] layer extracts it.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------
# Geometry helpers (torch)
# --------------------------------------------------------------------------
def cell_centres(h: int, w: int, stride: int, device, dtype) -> torch.Tensor:
    """(2, h, w) canvas-pixel centres of the cells of a stride-s feature map."""
    ys = (torch.arange(h, device=device, dtype=dtype) + 0.5) * stride
    xs = (torch.arange(w, device=device, dtype=dtype) + 0.5) * stride
    gy, gx = torch.meshgrid(ys, xs, indexing="ij")
    return torch.stack([gx, gy], 0)


def prior_field_from_theta(theta_canvas: torch.Tensor, h: int, w: int, stride: int = 1) -> torch.Tensor:
    """Dense offset field (B, 2, h, w) in canvas px from IR->RGB affine maps (B, 2, 3)."""
    g = cell_centres(h, w, stride, theta_canvas.device, theta_canvas.dtype).reshape(2, -1)
    mapped = theta_canvas[:, :, :2] @ g + theta_canvas[:, :, 2:]
    return (mapped - g).reshape(-1, 2, h, w)


def warp_by_offset(feat: torch.Tensor, offset_px: torch.Tensor, stride: int) -> torch.Tensor:
    """Sample `feat` at (cell centre + offset). offset_px: (B, 2, h, w) or (B, 2, 1, 1)."""
    b, _, h, w = feat.shape
    if offset_px.shape[-2:] != (h, w):
        offset_px = offset_px.expand(b, 2, h, w) if offset_px.shape[-1] == 1 else \
            F.adaptive_avg_pool2d(offset_px, (h, w))
    pos = cell_centres(h, w, stride, feat.device, feat.dtype).unsqueeze(0) + offset_px
    gx = 2.0 * pos[:, 0] / (stride * w) - 1.0
    gy = 2.0 * pos[:, 1] / (stride * h) - 1.0
    grid = torch.stack([gx, gy], -1)
    return F.grid_sample(feat, grid, mode="bilinear", padding_mode="zeros", align_corners=False)


def local_correlation(a: torch.Tensor, b: torch.Tensor, radius: int) -> torch.Tensor:
    """Cost volume (B, (2r+1)^2, h, w): mean over channels of a(x) * b(x + d)."""
    _, _, h, w = a.shape
    bp = F.pad(b, (radius, radius, radius, radius))
    vols = []
    for dy in range(2 * radius + 1):
        for dx in range(2 * radius + 1):
            vols.append((a * bp[:, :, dy:dy + h, dx:dx + w]).mean(1, keepdim=True))
    return torch.cat(vols, 1)


def _conv(c1, c2, k=3):
    return nn.Sequential(nn.Conv2d(c1, c2, k, padding=k // 2, bias=False), nn.BatchNorm2d(c2), nn.SiLU())


# --------------------------------------------------------------------------
# Modules
# --------------------------------------------------------------------------
class Pick(nn.Module):
    """Placeholder that returns its first input unchanged.

    Used in the stage-1 yaml (yolo11-RGBT-SPLA-base.yaml) in place of the
    SPLA layers, so that stage-1 and stage-2 models have identical layer
    indices and stage-1 weights load into stage 2 by name.
    """

    def forward(self, x):
        return x[0] if isinstance(x, (list, tuple)) else x



class _TransientCacheModule(nn.Module):
    def __getstate__(self):
        state = super().__getstate__()
        state["cache"] = {}
        return state


class GlobalOffsetHead(_TransientCacheModule):
    """Per-image global translation on top of the prior, from P5 features.

    Inputs:  [f_rgb_p5, f_ir_p5, prior_full]   prior_full: (B, 2, H, W) canvas px
    Output:  prior_full + global translation   (B, 2, H, W)

    Both the SPLA model (A4) and the per-image control (A3) contain this head,
    so the two have identical parameters: A3 feeds a zero prior, A4 the
    sequence prior. With a zero prior the head IS a per-image aligner.
    """

    def __init__(self, c_rgb: int, c_ir: int, stride: int = 32, radius: int = 3, c_mid: int = 64):
        super().__init__()
        self.stride, self.radius = stride, radius
        self.p_rgb = nn.Conv2d(c_rgb, c_mid, 1)
        self.p_ir = nn.Conv2d(c_ir, c_mid, 1)
        n_corr = (2 * radius + 1) ** 2
        self.head = nn.Sequential(_conv(n_corr + 2 * c_mid, 128), _conv(128, 64), nn.Conv2d(64, 3, 1))
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)
        self.cache = {}

    def forward(self, x):
        f_rgb, f_ir, prior = x
        b, _, h, w = f_ir.shape
        p_l = F.adaptive_avg_pool2d(prior, (h, w))
        rgb_w = warp_by_offset(f_rgb, p_l, self.stride)
        a, c = F.normalize(self.p_ir(f_ir), dim=1), F.normalize(self.p_rgb(rgb_w), dim=1)
        out = self.head(torch.cat([local_correlation(a, c, self.radius), a, c], 1))
        shift = self.radius * torch.tanh(out[:, :2])                  # cells
        attn = torch.softmax(out[:, 2:3].flatten(2), -1).view(b, 1, h, w)
        g = (shift * attn).sum((2, 3), keepdim=True) * self.stride    # (B, 2, 1, 1) px
        self.cache = {"global_px": g}
        return prior + g


class SPLA(_TransientCacheModule):
    """Sequence-prior local alignment for one pyramid level.

    Inputs:  [f_rgb, f_ir, prior_full]  prior_full: (B, 2, H, W) offsets in canvas px
    Output:  RGB features aligned to the IR grid, same shape as f_rgb.
    After each forward, ``self.cache`` holds:
      total_px (B,2,h,w)  prior + residual, canvas px  (used for loss and for
                          reading offset observations at detections)
      delta    (B,2,h,w)  residual in cells
      log_b    (B,1,h,w)  log Laplace scale of the total offset (uncertainty)
    """

    def __init__(self, c_rgb: int, c_ir: int, stride: int, radius: int = 3, c_mid: int = 64):
        super().__init__()
        self.stride, self.radius = stride, radius
        self.p_rgb = nn.Conv2d(c_rgb, c_mid, 1)
        self.p_ir = nn.Conv2d(c_ir, c_mid, 1)
        n_corr = (2 * radius + 1) ** 2
        self.head = nn.Sequential(_conv(n_corr + 2 * c_mid, 128), _conv(128, 64), nn.Conv2d(64, 3, 1))
        nn.init.zeros_(self.head[-1].weight)       # start as "trust the prior"
        nn.init.zeros_(self.head[-1].bias)
        self.cache = {}

    def forward(self, x):
        f_rgb, f_ir, prior = x
        _, _, h, w = f_ir.shape
        p_l = F.adaptive_avg_pool2d(prior, (h, w)) if prior.shape[-1] != 1 else prior.expand(-1, -1, h, w)
        rgb_w = warp_by_offset(f_rgb, p_l, self.stride)
        a, c = F.normalize(self.p_ir(f_ir), dim=1), F.normalize(self.p_rgb(rgb_w), dim=1)
        out = self.head(torch.cat([local_correlation(a, c, self.radius), a, c], 1))
        delta = self.radius * torch.tanh(out[:, :2])
        log_b = out[:, 2:3].clamp(-6.0, 4.0)
        total = p_l + delta * self.stride
        self.cache = {"total_px": total, "delta": delta, "log_b": log_b}
        return warp_by_offset(f_rgb, total, self.stride)


# --------------------------------------------------------------------------
# Alignment loss
# --------------------------------------------------------------------------
def sample_at(field: torch.Tensor, pts_px: torch.Tensor, img_idx: torch.Tensor, stride: int) -> torch.Tensor:
    """Bilinear read of a (B, C, h, w) field at canvas points (N, 2) of images img_idx (N,)."""
    _, c, h, w = field.shape
    gx = 2.0 * pts_px[:, 0] / (stride * w) - 1.0
    gy = 2.0 * pts_px[:, 1] / (stride * h) - 1.0
    grid = torch.stack([gx, gy], -1).view(-1, 1, 1, 2)
    sel = field[img_idx.long()]
    return F.grid_sample(sel, grid, mode="bilinear", align_corners=False).view(-1, c)


class SPLAAlignLoss(nn.Module):
    """Laplace negative log-likelihood of the true RGB-IR offset at each target.

    targets: dict with
      ir_ctr_px  (N, 2) IR box centres on the canvas
      off_px     (N, 2) true offsets (RGB centre - IR centre) on the canvas
      img_idx    (N,)   image index in the batch
    """

    def __init__(self, global_weight: float = 0.5):
        super().__init__()
        self.global_weight = global_weight

    def forward(self, spla_layers, global_heads, targets) -> torch.Tensor:
        if targets["off_px"].numel() == 0:
            dev = next(iter(spla_layers)).cache["total_px"].device if spla_layers else "cpu"
            return torch.zeros((), device=dev)
        loss = 0.0
        for m in spla_layers:
            s = m.stride
            pred = sample_at(m.cache["total_px"], targets["ir_ctr_px"], targets["img_idx"], s)
            log_b = sample_at(m.cache["log_b"], targets["ir_ctr_px"], targets["img_idx"], s)
            scale = torch.exp(log_b) * s                       # px
            err = (pred - targets["off_px"]).abs().sum(1, keepdim=True)
            loss = loss + (err / scale + 2.0 * torch.log(scale)).mean()
        for g in global_heads:
            gp = g.cache["global_px"].view(-1, 2)[targets["img_idx"].long()]
            # the global head is supervised only on what the prior leaves unexplained
            loss = loss + self.global_weight * F.smooth_l1_loss(
                gp, targets["off_px"] - targets.get("prior_px_at_target", torch.zeros_like(gp)), beta=2.0)
        return loss
