#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
import numpy as np
from matplotlib import cm
import cv2 

def mse(img1, img2):
    return (((img1 - img2)) ** 2).view(img1.shape[0], -1).mean(1, keepdim=True)

def psnr(img1, img2):
    mse = (((img1 - img2)) ** 2).view(img1.shape[0], -1).mean(1, keepdim=True)
    return 20 * torch.log10(1.0 / torch.sqrt(mse))

def apply_colormap(image, cmap="viridis"):
    colormap = cm.get_cmap(cmap)
    colormap = torch.tensor(colormap.colors).to(image.device)  # type: ignore
    image_long = (image * 255).long()
    image_long_min = torch.min(image_long)
    image_long_max = torch.max(image_long)
    assert image_long_min >= 0, f"the min value is {image_long_min}"
    assert image_long_max <= 255, f"the max value is {image_long_max}"
    return colormap[image_long[..., 0]]

def apply_depth_colormap(depth, cmap="turbo", min=None, max=None):
    near_plane = float(torch.min(depth)) if min is None else min
    far_plane = float(torch.max(depth)) if max is None else max

    depth = (depth - near_plane) / (far_plane - near_plane + 1e-10)
    depth = torch.clip(depth, 0, 1)

    colored_image = apply_colormap(depth, cmap=cmap)
    return colored_image

def apply_colormap_viridis(rendered_tensor):
    B, C, _, _ = rendered_tensor.shape
    assert C == 3, "Expected input with 3 channels (grayscale RGB)"
    
    vmin = 0.0
    vmax = rendered_tensor.max().item()

    colorized = []
    for b in range(B):
        grayscale_img = rendered_tensor[b, 0].cpu().numpy()  # shape: [H, W]
        img_norm = np.clip((grayscale_img - vmin) / (vmax - vmin + 1e-8), 0.0, 1.0)

        color_img = cm.viridis(img_norm)[..., :3]  # Discard alpha channel
        color_tensor = torch.from_numpy(color_img).permute(2, 0, 1).float()

        colorized.append(color_tensor)

    return torch.stack(colorized, dim=0)  # [B, 3, H, W]

def erode(img_in, erode_size=4):
    img_out = np.copy(img_in)
    kernel = np.ones((erode_size, erode_size), np.uint8)
    img_out = cv2.erode(img_out, kernel, iterations=1)

    return img_out


def _viridis_lut(n: int = 256, device=None, dtype=torch.float32):
    """
    Build a viridis LUT (n x 3) on the right device/dtype.
    Uses matplotlib if available; otherwise falls back to a simple approximation.
    """
    try:
        import numpy as np
        import matplotlib.cm as cm
        lut_np = cm.get_cmap("viridis", n)(np.linspace(0.0, 1.0, n))[:, :3]
        lut = torch.tensor(lut_np, device=device, dtype=dtype)
    except Exception:
        # Fallback: not exact viridis, but keeps code working without matplotlib.
        t = torch.linspace(0, 1, n, device=device, dtype=dtype)
        # crude 3-channel ramp (better than crashing)
        lut = torch.stack([0.18 + 0.82*t, 0.0 + t, 0.20 + 0.80*(1-t)], dim=-1).clamp(0, 1)
    return lut  # (n,3)


def apply_colormap_viridis_train(x: torch.Tensor) -> torch.Tensor:
    """
    Differentiable, LUT-free, piece-wise linear approximation of viridis.
    Assumes x in [0,1]. Accepts (H,W), (1,H,W), (B,1,H,W).
    Returns (B,3,H,W).
    """
    x = x.float()
    if x.ndim == 2:        # (H,W)
        x = x.unsqueeze(0).unsqueeze(0)
    elif x.ndim == 3:      # (C,H,W) or (1,H,W)
        if x.shape[0] == 3:
            return x.unsqueeze(0)        # already RGB -> (1,3,H,W)
        x = x.unsqueeze(0)               # -> (1,1,H,W)
    elif x.ndim == 4:      # (B,C,H,W)
        if x.shape[1] == 3:
            return x                     # already RGB
        assert x.shape[1] == 1, "Expected grayscale (B,1,H,W) or RGB (B,3,H,W)."
    else:
        raise ValueError(f"Unsupported tensor shape {tuple(x.shape)}")

    B, _, H, W = x.shape
    t = x.clamp(0.0, 1.0).squeeze(1)            # (B,H,W)

    # Viridis anchors (approx) at t = [0.0, 0.25, 0.50, 0.75, 0.90, 1.0]
    # Values are from matplotlib viridis samples, rounded a bit for compactness.
    pos = torch.tensor([0.00, 0.25, 0.50, 0.75, 0.90, 1.00], device=t.device, dtype=t.dtype)  # (K,)
    cols = torch.tensor([
        [0.267, 0.005, 0.329],  # t=0.00
        [0.283, 0.141, 0.458],  # t=0.25
        [0.254, 0.265, 0.530],  # t=0.50
        [0.207, 0.372, 0.553],  # t=0.75
        [0.275, 0.633, 0.437],  # t=0.90
        [0.993, 0.906, 0.144],  # t=1.00
    ], device=t.device, dtype=t.dtype)  # (K,3)

    # Find segment for each pixel: pos[i] <= t < pos[i+1]
    idx = torch.bucketize(t, pos) - 1
    idx = idx.clamp(0, pos.numel() - 2)          # (B,H,W)

    p0 = pos[idx]                                 # (B,H,W)
    p1 = pos[idx + 1]
    w  = (t - p0) / (p1 - p0 + 1e-8)              # (B,H,W)

    c0 = cols[idx]                                # (B,H,W,3)
    c1 = cols[idx + 1]
    rgb = (1 - w.unsqueeze(-1)) * c0 + w.unsqueeze(-1) * c1
    rgb = rgb.permute(0, 3, 1, 2).contiguous()    # (B,3,H,W)
    return rgb


def window_from_tensor(x: torch.Tensor, q_lo: float = 0.01, q_hi: float = 0.99):
    """
    Compute a single (vmin, vmax) from a grayscale tensor x (any of HxW, 1xHxW, Bx1xHxW),
    typically the GT, to reuse for both pred and GT.
    """
    if x.ndim == 2:
        v = x
    elif x.ndim == 3:
        v = x if x.shape[0] == 1 else x[0:1]  # assume grayscale
    elif x.ndim == 4:
        v = x[:, :1]  # use all batch pixels
    else:
        raise ValueError(f"Unsupported shape for window: {tuple(x.shape)}")

    lo = torch.quantile(v, q_lo).detach()
    hi = torch.quantile(v, q_hi).detach()
    return lo, hi