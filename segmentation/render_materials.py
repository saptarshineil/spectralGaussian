from __future__ import annotations
import os
import json
import math
from dataclasses import dataclass
from typing import Dict, Tuple, Optional, Sequence
from argparse import ArgumentParser

import numpy as np
import torch
import torchvision
from PIL import Image
from sklearn.cluster import MeanShift, estimate_bandwidth
from sklearn.metrics import silhouette_score, davies_bouldin_score

import sys
sys.path.append('../')

from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import render_mu
from scene import Scene, GaussianModel
from utils.general_utils import safe_state
from utils.spectralCT_utils import get_B_E_at_energy


# ----------------------------- Datatypes ----------------------------- #
@dataclass
class MaterialClusteringResult:
    labels: np.ndarray              # (N,) cluster id per Gaussian
    centers: np.ndarray             # (K, D) cluster centers in feature space
    metrics: Dict[str, float]       # silhouette, dbi
    label_to_name: Dict[int, str]   # e.g., {0: "Soft Tissue", 1: "Bone"}
    label_to_color: Dict[int, Tuple[int, int, int]]  # RGB per label (0..255)


# ----------------------- Step 1: feature extraction ------------------ #
def extract_material_features(gaussians: GaussianModel,
                              pe_is_channel0: bool = True,
                              eps: float = 1e-8) -> Dict[str, np.ndarray]:
    """
    Returns per-Gaussian:
        rho:         (N,)
        pe_coeff:    (N,)
        comp_coeff:  (N,)
        ratio:       (N,)  = pe/comp
        feats2d:     (N,2) = [rho, ratio]
    Works on CUDA tensors but returns *numpy* arrays on CPU (for sklearn).
    """
    with torch.no_grad():
        rho: torch.Tensor = gaussians.get_rhos
        frac: torch.Tensor = gaussians.get_fracs

        if frac.shape[1] < 2:
            raise ValueError(f"Expected at least 2 basis channels (PE, Compton); got {frac.shape[1]}.")

        pe_ch = 0 if pe_is_channel0 else 1
        comp_ch = 1 - pe_ch

        pe_coeff  = (rho[:, 0] * frac[:, pe_ch]).clamp_min(0.0)
        comp_coeff= (rho[:, 0] * frac[:, comp_ch]).clamp_min(eps)
        ratio     = (pe_coeff / (comp_coeff + eps)).clamp_max(1e6)

        feats2d = torch.stack([rho[:, 0], ratio], dim=1)

        out = dict(
            rho      = rho[:, 0].detach().cpu().numpy(),
            pe_coeff = pe_coeff.detach().cpu().numpy(),
            comp_coeff = comp_coeff.detach().cpu().numpy(),
            ratio    = ratio.detach().cpu().numpy(),
            feats2d  = feats2d.detach().cpu().numpy(),
        )
        return out


def _scale_features_minmax(X: np.ndarray,
                           clip_quantiles: Tuple[float, float] = (0.01, 0.99)) -> Tuple[np.ndarray, Dict[str, float]]:
    """min–max scale both columns robustly (ignore extreme tails)."""
    X = np.asarray(X, np.float32)
    Xs = X.copy()
    stats = {}
    for j in range(X.shape[1]):
        lo, hi = np.quantile(X[:, j], clip_quantiles)
        denom = (hi - lo) if (hi > lo) else 1.0
        Xs[:, j] = np.clip((X[:, j] - lo) / denom, 0.0, 1.0)
        stats[f"col{j}_lo"] = float(lo)
        stats[f"col{j}_hi"] = float(hi)
    return Xs, stats


# -------------------- Step 2–3: clustering & labels ------------------ #
def cluster_materials_mean_shift(X_2d: np.ndarray,
                                 bandwidth: Optional[float] = None,
                                 try_quantiles: Sequence[float] = (0.15, 0.2, 0.25)) -> Tuple[np.ndarray, np.ndarray]:
    """
    Mean-Shift in 2D (rho vs ratio). If bandwidth is None, estimate it;
    we try a few quantiles and pick the one yielding 2–3 clusters, else fallback.
    Returns (labels, centers).
    """

    if bandwidth is None:
        best = None
        for q in try_quantiles:
            try:
                bw = estimate_bandwidth(X_2d, quantile=q, n_samples=min(10000, len(X_2d)))
                if not (bw > 0):
                    continue
                ms = MeanShift(bandwidth=bw, bin_seeding=True, cluster_all=True)
                ms.fit(X_2d)
                K = len(ms.cluster_centers_)
                # prefer 2, accept 3
                score = 0 if K == 2 else (1 if K == 3 else 10 + abs(K-2))
                cand = (score, K, ms)
                if best is None or cand < best:
                    best = cand
            except Exception:
                continue
        if best is None:
            ms = MeanShift(bin_seeding=True, cluster_all=True).fit(X_2d)
            return ms.labels_.astype(np.int32), ms.cluster_centers_
        ms = best[2]
        return ms.labels_.astype(np.int32), ms.cluster_centers_
    else:
        ms = MeanShift(bandwidth=bandwidth, bin_seeding=True, cluster_all=True).fit(X_2d)
        return ms.labels_.astype(np.int32), ms.cluster_centers_


def label_clusters_bone_vs_tissue(X_2d: np.ndarray,
                                  labels: np.ndarray) -> Dict[int, str]:
    """
    Label each cluster id as "Bone (PE-dominated)" or "Soft Tissue (Compton-dominated)"
    based on average ratio (column 1).
    """
    uniq = np.unique(labels)
    mean_ratio = {k: float(np.mean(X_2d[labels == k, 1])) for k in uniq}
    if len(uniq) == 1:
        return {int(uniq[0]): "Soft Tissue (Compton-dominated)"}
    bone_label = max(mean_ratio, key=mean_ratio.get)
    mapping = {}
    for k in uniq:
        mapping[int(k)] = "Bone (PE-dominated)" if k == bone_label else "Soft Tissue (Compton-dominated)"
    return mapping


def default_cluster_colors(label_to_name: Dict[int, str]) -> Dict[int, Tuple[int, int, int]]:
    """
    Deterministic, high-contrast defaults:
      Bone  -> blue  (0, 110, 255)
      Tissue-> red   (230, 60, 60)
      Others-> green variants
    """
    colors = {}
    used_else = 0
    for k, name in label_to_name.items():
        if "Bone" in name:
            colors[k] = (0, 110, 255)
        elif "Soft Tissue" in name:
            colors[k] = (230, 60, 60)
        else:
            colors[k] = (60, 180 - 40 * used_else, 80 + 60 * used_else)
            used_else += 1
    return colors


def compute_internal_metrics(X_2d: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
    out = {}
    try:
        if len(np.unique(labels)) >= 2:
            out["silhouette"] = float(silhouette_score(X_2d, labels, metric="euclidean"))
            out["davies_bouldin"] = float(davies_bouldin_score(X_2d, labels))
        else:
            out["silhouette"] = float("nan")
            out["davies_bouldin"] = float("nan")
    except Exception:
        out["silhouette"] = float("nan")
        out["davies_bouldin"] = float("nan")
    return out


# ------------------- Step 4: RGB mapping utilities ------------------- #
def _normalize01(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    xp = x.detach().float()
    v = xp.view(-1)
    lo = torch.quantile(v, 0.01)
    hi = torch.quantile(v, 0.99)
    return (xp - lo).clamp(min=0) / (hi - lo + eps)


def compose_blend(bone_img: torch.Tensor,
                  tissue_img: torch.Tensor,
                  bone_rgb: Tuple[int, int, int],
                  tissue_rgb: Tuple[int, int, int]) -> torch.Tensor:
    """
    Input: bone_img, tissue_img: (1,H,W) tensors (linear μ or path-integral units)
    Output: seg_rgb in [0,1], shape (3,H,W)
    """
    b = _normalize01(bone_img)
    t = _normalize01(tissue_img)
    br = torch.tensor(bone_rgb, dtype=torch.float32, device=b.device) / 255.0
    tr = torch.tensor(tissue_rgb, dtype=torch.float32, device=t.device) / 255.0
    seg = b * br.view(3, 1, 1) + t * tr.view(3, 1, 1)
    seg = seg.clamp(0.0, 1.0)
    return seg


def compose_hard(bone_img: torch.Tensor,
                 tissue_img: torch.Tensor,
                 bone_rgb: Tuple[int, int, int],
                 tissue_rgb: Tuple[int, int, int]) -> torch.Tensor:
    """
    Winner-take-all per pixel. Returns (3,H,W) in [0,1].
    """
    mask = (bone_img > tissue_img)  # (1,H,W) bool

    bone_col = torch.tensor(bone_rgb, dtype=torch.float32, device=bone_img.device).view(3, 1, 1) / 255.0
    tiss_col = torch.tensor(tissue_rgb, dtype=torch.float32, device=bone_img.device).view(3, 1, 1) / 255.0

    # broadcast to (3,H,W) and select per-pixel color
    seg = torch.where(mask.expand(3, -1, -1), bone_col, tiss_col)
    return seg


# --------- Step 5: render per-cluster and build overlays ------------- #
@torch.no_grad()
def _render_cluster_mu(view, gaussians: GaussianModel, pipe: PipelineParams,
                       keep_mask: torch.Tensor,
                       energy_keV: float = 50.0) -> torch.Tensor:
    """
    Render μ image for a subset of Gaussians defined by boolean keep_mask (N,).
    We realize the subset by temporarily zeroing ρ for Gaussians not kept.
    Returns μ_E as (1,H,W) float32 on CUDA.
    """
    device = gaussians._rhos.device
    assert keep_mask.dtype == torch.bool and keep_mask.ndim == 1
    assert keep_mask.numel() == gaussians._rhos.shape[0]

    # backup
    raw_backup = gaussians._rhos.detach().clone()

    try:
        # set non-selected rhos very small -> softplus(raw) ~ 0
        tiny_raw = torch.full_like(gaussians._rhos, -20.0)  # softplus(-20) ~ 2e-9
        gaussians._rhos[~keep_mask] = tiny_raw[~keep_mask]

        pkg = render_mu(view, gaussians, pipe)  # dict with "bcoeffs": (C,H,W)
        bcoeff_img: torch.Tensor = pkg["bcoeffs"]  # accumulated basis coeffs
        torch.cuda.synchronize()

        # print(bcoeff_img.shape)

        # convert to μ(E) using analytic bases at requested energy
        E_grid = torch.tensor([energy_keV], device=device, dtype=torch.float32)
        energy_keV = torch.tensor([energy_keV], device=device, dtype=torch.float32)
        bfuncs = get_B_E_at_energy(E_grid, None, energy_keV).squeeze(0)
        # print(bfuncs.shape)
        mu = (bcoeff_img * bfuncs[:, None, None]).sum(dim=0, keepdim=True)
        return mu
    finally:
        # restore
        gaussians._rhos[:] = raw_backup


def _ensure_dir(p):
    os.makedirs(p, exist_ok=True)
    return p


@torch.no_grad()
def render_material_views(model_path: str,
                          split_name: str,
                          iteration: int,
                          views,
                          gaussians: GaussianModel,
                          pipe: PipelineParams,
                          cluster_result: MaterialClusteringResult,
                          energy_keV: float = 50.0,
                          blend: bool = True):
    """
    For each view:
      - render μ from Bone-only and Tissue-only sets,
      - compose RGB seg image (blend or hard),
      - save PNGs under model_path/materials.
    """
    out_dir = _ensure_dir(os.path.join(model_path, "materials",
                                       f"iter_{int(iteration)}",
                                       split_name))
    out_bone = _ensure_dir(os.path.join(out_dir, "bone"))
    out_tiss = _ensure_dir(os.path.join(out_dir, "tissue"))
    out_rgb  = _ensure_dir(os.path.join(out_dir, "rgb"))

    bone_labels = [k for k, v in cluster_result.label_to_name.items() if "Bone" in v]
    tissue_labels = [k for k, v in cluster_result.label_to_name.items() if "Soft Tissue" in v]
    if not bone_labels or not tissue_labels:
        uniq = np.unique(cluster_result.labels)
        means = {k: float(np.mean(cluster_result.centers[k, 1]))
                 if cluster_result.centers.shape[1] > 1 else float(k)
                 for k in uniq}
        bone_label = max(means, key=means.get)
        bone_labels = [bone_label]
        tissue_labels = [k for k in uniq if k != bone_label]

    bone_mask = torch.from_numpy(np.isin(cluster_result.labels, bone_labels)).to(gaussians._rhos.device)
    tiss_mask = torch.from_numpy(np.isin(cluster_result.labels, tissue_labels)).to(gaussians._rhos.device)

    bone_rgb = cluster_result.label_to_color[bone_labels[0]]
    tiss_rgb = cluster_result.label_to_color[tissue_labels[0]]

    for idx, view in enumerate(views):
        mu_bone = _render_cluster_mu(view, gaussians, pipe, bone_mask, energy_keV)
        mu_tiss = _render_cluster_mu(view, gaussians, pipe, tiss_mask, energy_keV)

        seg = compose_blend(mu_bone, mu_tiss, bone_rgb, tiss_rgb) if blend \
              else compose_hard(mu_bone, mu_tiss, bone_rgb, tiss_rgb)

        torchvision.utils.save_image(mu_bone, os.path.join(out_bone, f"{idx:05d}.png"))
        torchvision.utils.save_image(mu_tiss, os.path.join(out_tiss, f"{idx:05d}.png"))
        torchvision.utils.save_image(seg,      os.path.join(out_rgb,  f"{idx:05d}.png"))


@torch.no_grad()
def run_material_clustering(gaussians: GaussianModel,
                            bandwidth: Optional[float] = None,
                            pe_is_channel0: bool = True) -> MaterialClusteringResult:
    feats = extract_material_features(gaussians, pe_is_channel0=pe_is_channel0)
    Xs, _ = _scale_features_minmax(feats["feats2d"])

    labels, centers = cluster_materials_mean_shift(Xs, bandwidth=bandwidth)
    label_to_name = label_clusters_bone_vs_tissue(Xs, labels)
    label_to_color = default_cluster_colors(label_to_name)
    metrics = compute_internal_metrics(Xs, labels)

    return MaterialClusteringResult(
        labels=labels.astype(np.int32),
        centers=centers.astype(np.float32),
        metrics=metrics,
        label_to_name=label_to_name,
        label_to_color=label_to_color
    )


@torch.no_grad()
def render_materials(dataset: ModelParams,
                     iteration: int,
                     pipeline: PipelineParams,
                     *,
                     skip_train: bool = False,
                     skip_test: bool = False,
                     n_spectra: int = 2,
                     n_bcoeffs: int = 2,
                     bandwidth: Optional[float] = None,
                     pe_is_channel0: bool = True,
                     energy_keV: float = 50.0,
                     blend: bool = True) -> None:
    """
    End-to-end entry point: load scene & gaussians, cluster, save artifacts, render overlays.
    Artifacts:
      model_path/materials/iter_<it>/{train,test}/...
      model_path/materials/iter_<it>/cluster_info.json
      model_path/materials/iter_<it>/labels.npy
    """
    scene = Scene(dataset, n_spectra, n_bcoeffs, load_iteration=iteration, shuffle=False)

    scanner_cfg = scene.scanner_cfg
    volume_to_world = max(scanner_cfg["sVoxel"])
    scale_bound = None
    if dataset.scale_min > 0 and dataset.scale_max > 0:
        scale_bound = np.array([dataset.scale_min,
                                dataset.scale_max]) * volume_to_world
    gaussians = GaussianModel(n_spectra, n_bcoeffs, scale_bound)
    scene.init_gaussians(gaussians)

    # cluster
    result = run_material_clustering(gaussians, bandwidth=bandwidth, pe_is_channel0=pe_is_channel0)

    # persist labels + info
    out_root = _ensure_dir(os.path.join(dataset.model_path, "materials", f"iter_{int(scene.loaded_iter)}"))
    np.save(os.path.join(out_root, "labels.npy"), result.labels)
    info = {
        "iteration": int(scene.loaded_iter),
        "centers": result.centers.tolist(),
        "metrics": result.metrics,
        "label_to_name": result.label_to_name,
        "label_to_color": {int(k): list(map(int, v)) for k, v in result.label_to_color.items()},
        "n_gaussians": int(gaussians.get_xyz.shape[0]),
        "assumptions": {
            "basis_order": "[PE, Compton]" if pe_is_channel0 else "[Compton, PE]",
            "energy_keV": float(energy_keV),
            "blend": bool(blend),
        }
    }
    with open(os.path.join(out_root, "cluster_info.json"), "w") as f:
        json.dump(info, f, indent=2)

    # render splits
    if not skip_train:
        render_material_views(dataset.model_path, "train", scene.loaded_iter,
                              scene.getTrainCameras(), gaussians, pipeline, result,
                              energy_keV=energy_keV, blend=blend)
    if (not skip_test) and len(scene.getTestCameras()) > 0:
        render_material_views(dataset.model_path, "test", scene.loaded_iter,
                              scene.getTestCameras(), gaussians, pipeline, result,
                              energy_keV=energy_keV, blend=blend)


# -------------------------- CLI convenience -------------------------- #

if __name__ == "__main__":
    parser = ArgumentParser(description="Material rendering (Bone vs Soft Tissue) for 3DGS")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--n_spectra", type=int, default=2)
    parser.add_argument("--n_bcoeffs", type=int, default=2)

    # extras
    parser.add_argument("--bandwidth", type=float, default=None, help="Mean-shift bandwidth; None=auto")
    parser.add_argument("--pe_is_channel0", action="store_true", default=True)
    parser.add_argument("--energy_keV", type=float, default=50.0)
    parser.add_argument("--hard", action="store_true", help="Use hard per-pixel labels instead of blending")

    args = get_combined_args(parser)
    if not args.quiet:
        print("Material rendering", args.model_path)

    safe_state(args.quiet)

    render_materials(
        model.extract(args),
        args.iteration,
        pipeline.extract(args),
        skip_train=args.skip_train,
        skip_test=args.skip_test,
        n_spectra=args.n_spectra,
        n_bcoeffs=args.n_bcoeffs,
        bandwidth=None,
        pe_is_channel0=args.pe_is_channel0,
        energy_keV=args.energy_keV,
        blend=not args.hard,
    )
