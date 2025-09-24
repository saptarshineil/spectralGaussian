# from __future__ import annotations
# import os
# import re
# import json
# from dataclasses import dataclass
# from typing import Dict, Tuple, Optional, Sequence, List

# import numpy as np
# import torch
# import torchvision
# from sklearn.cluster import MeanShift, estimate_bandwidth
# from sklearn.metrics import silhouette_score, davies_bouldin_score
# from argparse import ArgumentParser

# import sys
# sys.path.append('../')

# from arguments import ModelParams, PipelineParams, get_combined_args
# from gaussian_renderer import render_bcoeffs
# from scene import Scene, GaussianModel
# from utils.general_utils import safe_state
# from utils.spectralCT_utils import (
#     get_B_E_at_energy, BFuncKedge,
# )

# # ----------------------------- Datatypes ----------------------------- #
# @dataclass
# class MaterialClusteringResult:
#     labels: np.ndarray
#     centers: np.ndarray
#     metrics: Dict[str, float]
#     label_to_name: Dict[int, str]
#     label_to_color: Dict[int, Tuple[int, int, int]]

# # ------------------------ Small utils ------------------------ #
# def _ensure_dir(p: str) -> str:
#     os.makedirs(p, exist_ok=True)
#     return p

# # ------------------- K-edge loader (iteration-specific) --------------- #
# def _load_kedge_checkpoint(model_path: str, iteration: int) -> Tuple[Optional[BFuncKedge], Optional[torch.Tensor]]:
#     """
#     Load kedge_{iteration:06d}.pt if present and return:
#       (bfunc_Kedge_module, E_grid (cuda float))
#     If not found, return (None, None).
#     """
#     ckpt_file = os.path.join(model_path, f"kedge_{int(iteration):06d}.pt")
#     if not os.path.isfile(ckpt_file):
#         # fallback: latest kedge_*.pt
#         cand = [f for f in os.listdir(model_path) if re.match(r"kedge_\d+\.pt$", f)]
#         if cand:
#             ckpt_file = os.path.join(model_path, sorted(cand)[-1])
#         else:
#             print("[kedge] no kedge_*.pt found; proceeding without learned K-edge.")
#             return None, None

#     pkg = torch.load(ckpt_file, map_location="cuda")
#     E_grid = pkg.get("E_grid", None)
#     sd = pkg.get("state_dict", None)
#     if E_grid is None or sd is None:
#         print(f"[kedge] invalid checkpoint payload at {ckpt_file}; ignoring.")
#         return None, None

#     E_grid = E_grid.detach().float().cuda()
#     bfunc_K = BFuncKedge(E_grid).cuda()
#     bfunc_K.load_state_dict(sd, strict=True)
#     bfunc_K.eval()
#     print(f"[kedge] loaded learned K-edge basis from {os.path.basename(ckpt_file)} (K={E_grid.numel()})")
#     return bfunc_K, E_grid

# # ----------------------- Step 1: feature extraction ------------------ #
# def extract_material_features(
#     gaussians: GaussianModel,
#     *,
#     pe_is_channel0: bool = True,
#     eps: float = 1e-8,
# ) -> Dict[str, np.ndarray]:
#     """
#     Returns per-Gaussian:
#         rho, pe_coeff, comp_coeff, kedge_coeff, ratio, log_ratio, feats2d=[rho, log1p(ratio)]
#     Works for 2 or 3 basis channels (kedge_coeff=0 if absent).
#     """
#     with torch.no_grad():
#         rho: torch.Tensor = gaussians.get_rhos         # (N,1) positive
#         frac: torch.Tensor = gaussians.get_fracs       # (N,C)

#         if frac.shape[1] < 2:
#             raise ValueError(f"Expected >=2 basis channels (PE, Compton[, K-edge]); got {frac.shape[1]}.")

#         if pe_is_channel0:
#             pe_ch, comp_ch = 0, 1
#         else:
#             pe_ch, comp_ch = 1, 0
#         k_ch = 2 if frac.shape[1] >= 3 else None

#         pe_coeff    = (rho[:, 0] * frac[:, pe_ch]).clamp_min(0.0)
#         comp_coeff  = (rho[:, 0] * frac[:, comp_ch]).clamp_min(eps)
#         kedge_coeff = (rho[:, 0] * frac[:, k_ch]).clamp_min(0.0) if k_ch is not None else torch.zeros_like(pe_coeff)

#         ratio     = (pe_coeff / (comp_coeff + eps)).clamp_max(1e6)
#         log_ratio = torch.log1p(ratio)
#         feats2d   = torch.stack([rho[:, 0], log_ratio], dim=1)

#         return dict(
#             rho        = rho[:, 0].detach().cpu().numpy(),
#             pe_coeff   = pe_coeff.detach().cpu().numpy(),
#             comp_coeff = comp_coeff.detach().cpu().numpy(),
#             kedge_coeff= kedge_coeff.detach().cpu().numpy(),
#             ratio      = ratio.detach().cpu().numpy(),
#             log_ratio  = log_ratio.detach().cpu().numpy(),
#             feats2d    = feats2d.detach().cpu().numpy(),
#         )

# def _scale_features_minmax(X: np.ndarray, clip_quantiles=(0.01, 0.99)):
#     X = np.asarray(X, np.float32)
#     Xs = X.copy()
#     stats = {}
#     for j in range(X.shape[1]):
#         lo, hi = np.quantile(X[:, j], clip_quantiles)
#         denom = (hi - lo) if (hi > lo) else 1.0
#         Xs[:, j] = np.clip((X[:, j] - lo) / denom, 0.0, 1.0)
#         stats[f"col{j}_lo"] = float(lo); stats[f"col{j}_hi"] = float(hi)
#     return Xs, stats

# # -------------------- Step 2–3: clustering & labels ------------------ #
# def cluster_materials_mean_shift(
#     X_2d: np.ndarray,
#     *,
#     bandwidth: Optional[float] = None,
#     try_quantiles: Sequence[float] = (0.15, 0.2, 0.25),
#     prefer_k3: bool = True,
# ):
#     def score_k(K: int) -> int:
#         return 0 if (K == 3 if prefer_k3 else K == 2) else (1 if (K == 2 if prefer_k3 else K == 3) else 10 + abs(K - (3 if prefer_k3 else 2)))
#     if bandwidth is None:
#         best = None
#         for q in try_quantiles:
#             try:
#                 bw = estimate_bandwidth(X_2d, quantile=q, n_samples=min(10000, len(X_2d)))
#                 if not (bw > 0): continue
#                 ms = MeanShift(bandwidth=bw, bin_seeding=True, cluster_all=True).fit(X_2d)
#                 K = len(ms.cluster_centers_)
#                 cand = (score_k(K), K, ms)
#                 if best is None or cand < best:
#                     best = cand
#             except Exception:
#                 continue
#         if best is None:
#             ms = MeanShift(bin_seeding=True, cluster_all=True).fit(X_2d)
#             return ms.labels_.astype(np.int32), ms.cluster_centers_
#         ms = best[2]
#         return ms.labels_.astype(np.int32), ms.cluster_centers_
#     else:
#         ms = MeanShift(bandwidth=bandwidth, bin_seeding=True, cluster_all=True).fit(X_2d)
#         return ms.labels_.astype(np.int32), ms.cluster_centers_

# def label_clusters_metal_bone_tissue(X_2d: np.ndarray, labels: np.ndarray) -> Dict[int, str]:
#     uniq = np.unique(labels)
#     stats = {int(k): {"mean_rho": float(np.mean(X_2d[labels == k, 0])),
#                       "mean_logR": float(np.mean(X_2d[labels == k, 1]))}
#              for k in uniq}
#     order = sorted(uniq, key=lambda k: (stats[int(k)]["mean_logR"], stats[int(k)]["mean_rho"]), reverse=True)
#     mapping: Dict[int, str] = {}
#     if len(order) >= 3:
#         mapping[int(order[0])] = "Metal (PE-dominated, high Z/ρ)"
#         mapping[int(order[1])] = "Bone (PE-dominated)"
#         mapping[int(order[2])] = "Soft Tissue (Compton-dominated)"
#         for k in order[3:]:
#             mapping[int(k)] = "Soft Tissue (Compton-dominated)"
#     elif len(order) == 2:
#         mapping[int(order[0])] = "Bone/Metal (PE-dominated)"
#         mapping[int(order[1])] = "Soft Tissue (Compton-dominated)"
#     else:
#         mapping[int(order[0])] = "Soft Tissue (Compton-dominated)"
#     return mapping

# def default_cluster_colors(label_to_name: Dict[int, str]) -> Dict[int, Tuple[int,int,int]]:
#     colors: Dict[int, Tuple[int,int,int]] = {}
#     for k, name in label_to_name.items():
#         # Treat "Bone/Metal" as Bone (blue) to keep K=2 visuals consistent.
#         if name.startswith("Bone/Metal"):
#             colors[k] = (0, 110, 255)      # blue
#         elif "Metal" in name:
#             colors[k] = (255, 215, 0)      # yellow
#         elif "Bone" in name:
#             colors[k] = (0, 110, 255)      # blue
#         elif "Soft Tissue" in name:
#             colors[k] = (230, 60, 60)      # red
#         else:
#             colors[k] = (60, 180, 80)      # fallback green
#     return colors

# def compute_internal_metrics(X_2d: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
#     out: Dict[str, float] = {}
#     try:
#         if len(np.unique(labels)) >= 2:
#             out["silhouette"] = float(silhouette_score(X_2d, labels, metric="euclidean"))
#             out["davies_bouldin"] = float(davies_bouldin_score(X_2d, labels))
#         else:
#             out["silhouette"] = float("nan"); out["davies_bouldin"] = float("nan")
#     except Exception:
#         out["silhouette"] = float("nan"); out["davies_bouldin"] = float("nan")
#     return out

# # ------------------- Step 4: RGB mapping utilities ------------------- #
# def _normalize01(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
#     xp = x.detach().float()
#     v = xp.view(-1)
#     lo = torch.quantile(v, 0.01)
#     hi = torch.quantile(v, 0.99)
#     return (xp - lo).clamp(min=0) / (hi - lo + eps)

# def compose_blend_multi(
#     imgs: Sequence[torch.Tensor],
#     rgbs: Sequence[Tuple[int,int,int]],
#     mode: str = "independent"   # "independent" (match 2-class code) or "proportional"
# ) -> torch.Tensor:
#     """
#     Blend K images into RGB using either:
#       - independent: min–max normalize each μ_k to [0,1], then sum μ_k * color_k
#       - proportional: min–max normalize, then normalise weights per pixel to sum=1
#     Returns (3,H,W) in [0,1].
#     """
#     assert len(imgs) == len(rgbs) and len(imgs) >= 1
#     dev = imgs[0].device
#     cols = [torch.tensor(c, dtype=torch.float32, device=dev).view(3,1,1)/255.0 for c in rgbs]
#     norms = [_normalize01(t) for t in imgs]  # (1,H,W) each

#     if mode == "proportional":
#         S = torch.stack(norms, dim=0)                 # (K,1,H,W)
#         w = S / (S.sum(dim=0, keepdim=True) + 1e-8)   # (K,1,H,W) weights sum to 1
#         seg = sum(w[k] * cols[k] for k in range(len(cols)))
#     else:
#         seg = sum(norms[k] * cols[k] for k in range(len(cols)))
#         seg = seg.clamp(0.0, 1.0)
#     return seg

# def compose_hard_multi(
#     imgs: Sequence[torch.Tensor],
#     rgbs: Sequence[Tuple[int,int,int]]
# ) -> torch.Tensor:
#     """
#     Hard labels via argmax over μ_k. Returns (3,H,W) in [0,1].
#     """
#     assert len(imgs) == len(rgbs) and len(imgs) >= 1
#     S = torch.stack(imgs, dim=0)                       # (K,1,H,W)
#     labels = torch.argmax(S, dim=0).squeeze(0)         # (H,W)
#     palette = torch.tensor(rgbs, dtype=torch.float32, device=S.device) / 255.0
#     seg = palette[labels]                               # (H,W,3)
#     return seg.permute(2,0,1).contiguous()

# # --------- Step 5: render per-class μ(E) using learned K-edge ------- #
# def _render_cluster_mu(
#     view,
#     gaussians: "GaussianModel",
#     pipe: "PipelineParams",
#     keep_mask: torch.Tensor,
#     energy_keV: float,
#     bfunc_Kedge: Optional["BFuncKedge"],
#     E_grid: Optional[torch.Tensor],
# ) -> torch.Tensor:
#     """
#     Render μ(E) for a subset of Gaussians (mask True = keep).
#     Uses get_B_E_at_energy on the training E_grid if a K-edge is present.
#     Returns (1, H, W) float32 on CUDA.
#     """
#     device = gaussians._rhos.device

#     assert keep_mask.dtype == torch.bool and keep_mask.ndim == 1
#     assert keep_mask.numel() == gaussians._rhos.shape[0]

#     # Clamp requested energy to grid range (only relevant if K-edge present).
#     if (bfunc_Kedge is not None) and (E_grid is not None):
#         ek = float(
#             torch.clamp(
#                 torch.tensor(energy_keV, device=device),
#                 min=E_grid.min(),
#                 max=E_grid.max(),
#             ).item()
#         )
#     else:
#         ek = float(energy_keV)

#     raw_backup = gaussians._rhos.detach().clone()
#     try:
#         # Soft-remove Gaussians not in the mask by pushing rho to ~0 after softplus.
#         tiny_raw = torch.full_like(gaussians._rhos, -20.0)  # softplus(-20) ≈ 2e-9
#         gaussians._rhos[~keep_mask] = tiny_raw[~keep_mask]

#         pkg = render_bcoeffs(view, gaussians, pipe)
#         bcoeff_img: torch.Tensor = pkg["bcoeffs"]  # (C, H, W)

#         # Build basis vector at ek, but NORMALIZE at a pivot away from the overlay energy
#         E  = torch.tensor([ek], device=device, dtype=torch.float32)
#         # eg = E_grid if E_grid is not None else torch.tensor([20.0, 140.0], device=device)
#         # E_ref_render = float(eg.max().item())  # e.g., 140 keV
#         # bfuncs = get_B_E_at_energy(
#         #     eg,
#         #     bfunc_Kedge if bfunc_Kedge is not None else None,
#         #     E,
#         #     normalize=True,
#         #     E_ref=E_ref_render,              # pivot ≠ overlay energy
#         # ).squeeze(0)  # (C,)
#         eg = E_grid if E_grid is not None else torch.tensor([20.0, 140.0], device=device)
#         E_ref_render = 70.0   # match the training pivot
#         bfuncs = get_B_E_at_energy(
#             eg,
#             bfunc_Kedge if bfunc_Kedge is not None else None,
#             E,
#             normalize=True,
#             E_ref=E_ref_render,
#         ).squeeze(0)

#         # If model has 3 coeffs but bfuncs has only 2, pad with 1.0.
#         if bcoeff_img.shape[0] == 3 and bfuncs.numel() == 2:
#             bfuncs = torch.cat(
#                 [bfuncs, torch.ones(1, device=device, dtype=bfuncs.dtype)], dim=0
#             )

#         mu = (bcoeff_img * bfuncs[:, None, None]).sum(dim=0, keepdim=True)
#         return mu
#     finally:
#         gaussians._rhos[:] = raw_backup

# # ---------------- Entry points: clustering + rendering ---------------- #
# @torch.no_grad()
# def run_material_clustering(gaussians: GaussianModel, *, bandwidth: Optional[float], pe_is_channel0: bool, prefer_k3: bool) -> MaterialClusteringResult:
#     feats = extract_material_features(gaussians, pe_is_channel0=pe_is_channel0)
#     Xs, _ = _scale_features_minmax(feats["feats2d"])
#     labels, centers = cluster_materials_mean_shift(Xs, bandwidth=bandwidth, prefer_k3=prefer_k3)
#     label_to_name  = label_clusters_metal_bone_tissue(Xs, labels)
#     label_to_color = default_cluster_colors(label_to_name)
#     metrics = compute_internal_metrics(Xs, labels)
#     return MaterialClusteringResult(
#         labels=labels.astype(np.int32),
#         centers=centers.astype(np.float32),
#         metrics=metrics,
#         label_to_name=label_to_name,
#         label_to_color=label_to_color
#     )

# @torch.no_grad()
# def render_material_views(
#     model_path: str,
#     split_name: str,
#     iteration: int,
#     views,
#     gaussians: GaussianModel,
#     pipe: PipelineParams,
#     cluster_result: MaterialClusteringResult,
#     *,
#     energy_keV: float,
#     bfunc_Kedge: Optional[BFuncKedge],
#     E_grid: Optional[torch.Tensor],
#     blend: bool,
#     blend_mode: str = "independent",   # or "proportional"
# ):
#     """
#     For each view:
#       - render μ for Metal/Bone/Tissue subsets (if present),
#       - compose RGB seg image (blend or hard),
#       - save PNGs under model_path/materials.
#     """
#     out_dir = _ensure_dir(os.path.join(model_path, "materials", f"iter_{int(iteration)}", split_name))
#     out_metal = _ensure_dir(os.path.join(out_dir, "metal"))
#     out_bone  = _ensure_dir(os.path.join(out_dir, "bone"))
#     out_tiss  = _ensure_dir(os.path.join(out_dir, "tissue"))
#     out_rgb   = _ensure_dir(os.path.join(out_dir, "rgb"))

#     # ----- class label ids (avoid substring collisions) -----
#     def ids_starting(prefix: str) -> List[int]:
#         return [k for k, v in cluster_result.label_to_name.items() if v.startswith(prefix)]

#     metal_labels  = ids_starting("Metal (")
#     bone_labels   = ids_starting("Bone (")
#     tissue_labels = ids_starting("Soft Tissue")

#     # If only 2 clusters with "Bone/Metal", treat it as Bone and keep Metal empty.
#     if not metal_labels and not bone_labels:
#         bone_labels = [k for k, v in cluster_result.label_to_name.items()
#                        if v.startswith("Bone/Metal")]

#     dev = gaussians._rhos.device
#     bone_rgb  = (cluster_result.label_to_color.get(bone_labels[0],  (0,110,255))
#                  if bone_labels else (0,110,255))
#     tiss_rgb  = (cluster_result.label_to_color.get(tissue_labels[0],(230,60,60))
#                  if tissue_labels else (230,60,60))
#     metal_rgb = (cluster_result.label_to_color.get(metal_labels[0], (255,215,0))
#                  if metal_labels else (255,215,0))

#     bone_mask  = torch.from_numpy(np.isin(cluster_result.labels, bone_labels)).to(dev)
#     tiss_mask  = torch.from_numpy(np.isin(cluster_result.labels, tissue_labels)).to(dev)
#     metal_mask = torch.from_numpy(np.isin(cluster_result.labels, metal_labels)).to(dev)

#     for idx, view in enumerate(views):
#         # Pre-render to get (H,W)
#         pkg0 = render_bcoeffs(view, gaussians, pipe)
#         H, W = pkg0["bcoeffs"].shape[-2:]

#         mu_metal = _render_cluster_mu(view, gaussians, pipe, metal_mask, energy_keV, bfunc_Kedge, E_grid) if metal_labels else torch.zeros(1, H, W, device=dev)
#         mu_bone  = _render_cluster_mu(view, gaussians, pipe, bone_mask,  energy_keV, bfunc_Kedge, E_grid) if bone_labels  else torch.zeros(1, H, W, device=dev)
#         mu_tiss  = _render_cluster_mu(view, gaussians, pipe, tiss_mask,  energy_keV, bfunc_Kedge, E_grid) if tissue_labels else torch.zeros(1, H, W, device=dev)

#         if blend:
#             seg = compose_blend_multi([mu_metal, mu_bone, mu_tiss],
#                                       [metal_rgb, bone_rgb, tiss_rgb],
#                                       mode=blend_mode)
#         else:
#             seg = compose_hard_multi([mu_metal, mu_bone, mu_tiss],
#                                      [metal_rgb, bone_rgb, tiss_rgb])

#         torchvision.utils.save_image(mu_metal, os.path.join(out_metal, f"{idx:05d}.png"))
#         torchvision.utils.save_image(mu_bone,  os.path.join(out_bone,  f"{idx:05d}.png"))
#         torchvision.utils.save_image(mu_tiss,  os.path.join(out_tiss,  f"{idx:05d}.png"))
#         torchvision.utils.save_image(seg,      os.path.join(out_rgb,   f"{idx:05d}.png"))

# # -------------------------- CLI convenience -------------------------- #
# @torch.no_grad()
# def render_materials(
#     dataset: ModelParams,
#     iteration: int,
#     pipeline: PipelineParams,
#     *,
#     skip_train: bool,
#     skip_test: bool,
#     n_spectra: int,
#     n_bcoeffs: int,
#     bandwidth: Optional[float],
#     pe_is_channel0: bool,
#     prefer_k3: bool,
#     energy_keV: float,
#     blend: bool,
#     blend_mode: str,
# ) -> None:
#     scene = Scene(dataset, n_spectra, n_bcoeffs, load_iteration=iteration, shuffle=False)

#     scanner_cfg = scene.scanner_cfg
#     volume_to_world = max(scanner_cfg["sVoxel"])
#     scale_bound = None
#     if dataset.scale_min > 0 and dataset.scale_max > 0:
#         scale_bound = np.array([dataset.scale_min, dataset.scale_max]) * volume_to_world
#     gaussians = GaussianModel(n_spectra, n_bcoeffs, scale_bound)
#     scene.init_gaussians(gaussians)

#     # Load learned K-edge for this iteration (if any)
#     bfunc_Kedge, E_grid = (None, None)
#     if n_bcoeffs >= 3:
#         bfunc_Kedge, E_grid = _load_kedge_checkpoint(dataset.model_path, scene.loaded_iter)

#     # cluster
#     result = run_material_clustering(gaussians, bandwidth=bandwidth, pe_is_channel0=pe_is_channel0, prefer_k3=prefer_k3)

#     # persist labels + info
#     out_root = _ensure_dir(os.path.join(dataset.model_path, "materials", f"iter_{int(scene.loaded_iter)}"))
#     np.save(os.path.join(out_root, "labels.npy"), result.labels)
#     info = {
#         "iteration": int(scene.loaded_iter),
#         "centers": result.centers.tolist(),
#         "metrics": result.metrics,
#         "label_to_name": result.label_to_name,
#         "label_to_color": {int(k): list(map(int, v)) for k, v in result.label_to_color.items()},
#         "n_gaussians": int(gaussians.get_xyz.shape[0]),
#         "assumptions": {
#             "basis_order": "[PE, Compton, (K-edge)]" if pe_is_channel0 else "[Compton, PE, (K-edge)]",
#             "feature_space": "[rho, log1p(pe/comp)] (min–max scaled)",
#             "energy_keV": float(energy_keV),
#             "prefer_k3": bool(prefer_k3),
#             "kedge_loaded": bool(bfunc_Kedge is not None),
#             "blend": bool(blend),
#             "blend_mode": str(blend_mode),
#         }
#     }
#     with open(os.path.join(out_root, "cluster_info.json"), "w") as f:
#         json.dump(info, f, indent=2)

#     # render splits
#     if not skip_train:
#         render_material_views(dataset.model_path, "train", scene.loaded_iter,
#                               scene.getTrainCameras(), gaussians, pipeline, result,
#                               energy_keV=energy_keV, bfunc_Kedge=bfunc_Kedge, E_grid=E_grid,
#                               blend=blend, blend_mode=blend_mode)
#     if (not skip_test) and len(scene.getTestCameras()) > 0:
#         render_material_views(dataset.model_path, "test", scene.loaded_iter,
#                               scene.getTestCameras(), gaussians, pipeline, result,
#                               energy_keV=energy_keV, bfunc_Kedge=bfunc_Kedge, E_grid=E_grid,
#                               blend=blend, blend_mode=blend_mode)

# if __name__ == "__main__":
#     parser = ArgumentParser(description="Material rendering (Metal/Bone/Soft) for 3DGS")
#     model = ModelParams(parser, sentinel=True)
#     pipeline = PipelineParams(parser)
#     parser.add_argument("--iteration", default=-1, type=int)
#     parser.add_argument("--skip_train", action="store_true")
#     parser.add_argument("--skip_test", action="store_true")
#     parser.add_argument("--quiet", action="store_true")
#     parser.add_argument("--n_spectra", type=int, default=2)
#     parser.add_argument("--n_bcoeffs", type=int, default=3)

#     # clustering controls
#     parser.add_argument("--bandwidth", type=float, default=None, help="Mean-shift bandwidth; None=auto")
#     parser.add_argument("--prefer_k3", action="store_true", default=True, help="Prefer 3 clusters when choosing bandwidth")
#     parser.add_argument("--pe_is_channel0", action="store_true", default=True, help="If false, assume [Compton, PE, (K)] order")

#     # rendering
#     parser.add_argument("--energy_keV", type=float, default=70.0, help="Monochromatic overlay energy (keV)")
#     parser.add_argument("--hard", action="store_true", help="Use hard per-pixel labels instead of blending")
#     parser.add_argument("--blend_mode", type=str, choices=["independent", "proportional"], default="independent",
#                         help="Blend strategy for multi-class overlays")

#     args = get_combined_args(parser)
#     if not args.quiet:
#         print("Material rendering", args.model_path)

#     safe_state(args.quiet)

#     render_materials(
#         model.extract(args),
#         args.iteration,
#         pipeline.extract(args),
#         skip_train=args.skip_train,
#         skip_test=args.skip_test,
#         n_spectra=args.n_spectra,
#         n_bcoeffs=args.n_bcoeffs,
#         bandwidth=None,           # use CLI
#         pe_is_channel0=args.pe_is_channel0,
#         prefer_k3=args.prefer_k3,
#         energy_keV=args.energy_keV,
#         blend=not args.hard,
#         blend_mode=args.blend_mode,
#     )


from __future__ import annotations
import os
import re
import json
from dataclasses import dataclass
from typing import Dict, Tuple, Optional, Sequence, List

import numpy as np
import torch
import torchvision
from sklearn.cluster import MeanShift, estimate_bandwidth
from sklearn.metrics import silhouette_score, davies_bouldin_score
from argparse import ArgumentParser

import sys
sys.path.append('../')

from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import render_bcoeffs
from scene import Scene, GaussianModel
from utils.general_utils import safe_state
from utils.spectralCT_utils import (
    get_B_E_at_energy, BFuncKedge,
)

# ----------------------------- Datatypes ----------------------------- #
@dataclass
class MaterialClusteringResult:
    labels: np.ndarray
    centers: np.ndarray
    metrics: Dict[str, float]
    label_to_name: Dict[int, str]
    label_to_color: Dict[int, Tuple[int, int, int]]

# ------------------------ Small utils ------------------------ #
def _ensure_dir(p: str) -> str:
    os.makedirs(p, exist_ok=True)
    return p

# ------------------- K-edge loader (iteration-specific) --------------- #
def _load_kedge_checkpoint(model_path: str, iteration: int) -> Tuple[Optional[BFuncKedge], Optional[torch.Tensor]]:
    """
    Load kedge_{iteration:06d}.pt if present and return:
      (bfunc_Kedge_module, E_grid (cuda float))
    If not found, return (None, None).
    """
    ckpt_file = os.path.join(model_path, f"kedge_{int(iteration):06d}.pt")
    if not os.path.isfile(ckpt_file):
        # fallback: latest kedge_*.pt
        cand = [f for f in os.listdir(model_path) if re.match(r"kedge_\d+\.pt$", f)]
        if cand:
            ckpt_file = os.path.join(model_path, sorted(cand)[-1])
        else:
            print("[kedge] no kedge_*.pt found; proceeding without learned K-edge.")
            return None, None

    pkg = torch.load(ckpt_file, map_location="cuda")
    E_grid = pkg.get("E_grid", None)
    sd = pkg.get("state_dict", None)
    if E_grid is None or sd is None:
        print(f"[kedge] invalid checkpoint payload at {ckpt_file}; ignoring.")
        return None, None

    E_grid = E_grid.detach().float().cuda()
    bfunc_K = BFuncKedge(E_grid).cuda()
    bfunc_K.load_state_dict(sd, strict=True)
    bfunc_K.eval()
    print(f"[kedge] loaded learned K-edge basis from {os.path.basename(ckpt_file)} (K={E_grid.numel()})")
    return bfunc_K, E_grid

# ----------------------- Step 1: feature extraction ------------------ #
def extract_material_features(
    gaussians: GaussianModel,
    *,
    pe_is_channel0: bool = True,
    eps: float = 1e-8,
) -> Dict[str, np.ndarray]:
    """
    Returns per-Gaussian:
        rho, pe_coeff, comp_coeff, kedge_coeff, ratio, log_ratio, feats2d=[rho, log1p(ratio)]
    Works for 2 or 3 basis channels (kedge_coeff=0 if absent).
    """
    with torch.no_grad():
        rho: torch.Tensor = gaussians.get_rhos         # (N,1) positive
        frac: torch.Tensor = gaussians.get_fracs       # (N,C)

        if frac.shape[1] < 2:
            raise ValueError(f"Expected >=2 basis channels (PE, Compton[, K-edge]); got {frac.shape[1]}.")

        if pe_is_channel0:
            pe_ch, comp_ch = 0, 1
        else:
            pe_ch, comp_ch = 1, 0
        k_ch = 2 if frac.shape[1] >= 3 else None

        pe_coeff    = (rho[:, 0] * frac[:, pe_ch]).clamp_min(0.0)
        comp_coeff  = (rho[:, 0] * frac[:, comp_ch]).clamp_min(eps)
        kedge_coeff = (rho[:, 0] * frac[:, k_ch]).clamp_min(0.0) if k_ch is not None else torch.zeros_like(pe_coeff)

        ratio     = (pe_coeff / (comp_coeff + eps)).clamp_max(1e6)
        log_ratio = torch.log1p(ratio)
        feats2d   = torch.stack([rho[:, 0], log_ratio], dim=1)

        return dict(
            rho        = rho[:, 0].detach().cpu().numpy(),
            pe_coeff   = pe_coeff.detach().cpu().numpy(),
            comp_coeff = comp_coeff.detach().cpu().numpy(),
            kedge_coeff= kedge_coeff.detach().cpu().numpy(),
            ratio      = ratio.detach().cpu().numpy(),
            log_ratio  = log_ratio.detach().cpu().numpy(),
            feats2d    = feats2d.detach().cpu().numpy(),
        )

def _scale_features_minmax(X: np.ndarray, clip_quantiles=(0.01, 0.99)):
    X = np.asarray(X, np.float32)
    Xs = X.copy()
    stats = {}
    for j in range(X.shape[1]):
        lo, hi = np.quantile(X[:, j], clip_quantiles)
        denom = (hi - lo) if (hi > lo) else 1.0
        Xs[:, j] = np.clip((X[:, j] - lo) / denom, 0.0, 1.0)
        stats[f"col{j}_lo"] = float(lo); stats[f"col{j}_hi"] = float(hi)
    return Xs, stats

# -------------------- Step 2–3: clustering & labels ------------------ #
def cluster_materials_mean_shift(
    X_2d: np.ndarray,
    *,
    bandwidth: Optional[float] = None,
    try_quantiles: Sequence[float] = (0.15, 0.2, 0.25),
    prefer_k3: bool = True,
):
    def score_k(K: int) -> int:
        return 0 if (K == 3 if prefer_k3 else K == 2) else (1 if (K == 2 if prefer_k3 else K == 3) else 10 + abs(K - (3 if prefer_k3 else 2)))
    if bandwidth is None:
        best = None
        for q in try_quantiles:
            try:
                bw = estimate_bandwidth(X_2d, quantile=q, n_samples=min(10000, len(X_2d)))
                if not (bw > 0): continue
                ms = MeanShift(bandwidth=bw, bin_seeding=True, cluster_all=True).fit(X_2d)
                K = len(ms.cluster_centers_)
                cand = (score_k(K), K, ms)
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

def label_clusters_metal_bone_tissue(X_2d: np.ndarray, labels: np.ndarray) -> Dict[int, str]:
    uniq = np.unique(labels)
    stats = {int(k): {"mean_rho": float(np.mean(X_2d[labels == k, 0])),
                      "mean_logR": float(np.mean(X_2d[labels == k, 1]))}
             for k in uniq}
    order = sorted(uniq, key=lambda k: (stats[int(k)]["mean_logR"], stats[int(k)]["mean_rho"]), reverse=True)
    mapping: Dict[int, str] = {}
    if len(order) >= 3:
        mapping[int(order[0])] = "Metal (PE-dominated, high Z/ρ)"
        mapping[int(order[1])] = "Bone (PE-dominated)"
        mapping[int(order[2])] = "Soft Tissue (Compton-dominated)"
        for k in order[3:]:
            mapping[int(k)] = "Soft Tissue (Compton-dominated)"
    elif len(order) == 2:
        mapping[int(order[0])] = "Bone/Metal (PE-dominated)"
        mapping[int(order[1])] = "Soft Tissue (Compton-dominated)"
    else:
        mapping[int(order[0])] = "Soft Tissue (Compton-dominated)"
    return mapping

def default_cluster_colors(label_to_name: Dict[int, str]) -> Dict[int, Tuple[int,int,int]]:
    colors: Dict[int, Tuple[int,int,int]] = {}
    for k, name in label_to_name.items():
        # Treat "Bone/Metal" as Bone (blue) to keep K=2 visuals consistent.
        if name.startswith("Bone/Metal"):
            colors[k] = (0, 110, 255)      # blue
        elif "Metal" in name:
            colors[k] = (255, 215, 0)      # yellow
        elif "Bone" in name:
            colors[k] = (0, 110, 255)      # blue
        elif "Soft Tissue" in name:
            colors[k] = (230, 60, 60)      # red
        else:
            colors[k] = (60, 180, 80)      # fallback green
    return colors

def compute_internal_metrics(X_2d: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
    out: Dict[str, float] = {}
    try:
        if len(np.unique(labels)) >= 2:
            out["silhouette"] = float(silhouette_score(X_2d, labels, metric="euclidean"))
            out["davies_bouldin"] = float(davies_bouldin_score(X_2d, labels))
        else:
            out["silhouette"] = float("nan"); out["davies_bouldin"] = float("nan")
    except Exception:
        out["silhouette"] = float("nan"); out["davies_bouldin"] = float("nan")
    return out

# ------------------- Step 4: RGB mapping utilities ------------------- #
def _normalize01(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    xp = x.detach().float()
    v = xp.view(-1)
    lo = torch.quantile(v, 0.01)
    hi = torch.quantile(v, 0.99)
    return (xp - lo).clamp(min=0) / (hi - lo + eps)

def compose_blend_multi(
    imgs: Sequence[torch.Tensor],
    rgbs: Sequence[Tuple[int,int,int]],
    mode: str = "independent"   # "independent" (match 2-class code) or "proportional"
) -> torch.Tensor:
    """
    Blend K images into RGB using either:
      - independent: min–max normalize each μ_k to [0,1], then sum μ_k * color_k
      - proportional: min–max normalize, then normalise weights per pixel to sum=1
    Returns (3,H,W) in [0,1].
    """
    assert len(imgs) == len(rgbs) and len(imgs) >= 1
    dev = imgs[0].device
    cols = [torch.tensor(c, dtype=torch.float32, device=dev).view(3,1,1)/255.0 for c in rgbs]
    norms = [_normalize01(t) for t in imgs]  # (1,H,W) each

    if mode == "proportional":
        S = torch.stack(norms, dim=0)                 # (K,1,H,W)
        w = S / (S.sum(dim=0, keepdim=True) + 1e-8)   # (K,1,H,W) weights sum to 1
        seg = sum(w[k] * cols[k] for k in range(len(cols)))
    else:
        seg = sum(norms[k] * cols[k] for k in range(len(cols)))
        seg = seg.clamp(0.0, 1.0)
    return seg

def compose_hard_multi(
    imgs: Sequence[torch.Tensor],
    rgbs: Sequence[Tuple[int,int,int]]
) -> torch.Tensor:
    """
    Hard labels via argmax over μ_k. Returns (3,H,W) in [0,1].
    """
    assert len(imgs) == len(rgbs) and len(imgs) >= 1
    S = torch.stack(imgs, dim=0)                       # (K,1,H,W)
    labels = torch.argmax(S, dim=0).squeeze(0)         # (H,W)
    palette = torch.tensor(rgbs, dtype=torch.float32, device=S.device) / 255.0
    seg = palette[labels]                               # (H,W,3)
    return seg.permute(2,0,1).contiguous()

# --------- Step 5: render per-class μ(E) using learned K-edge ------- #
def _render_cluster_mu(
    view,
    gaussians: "GaussianModel",
    pipe: "PipelineParams",
    keep_mask: torch.Tensor,
    energy_keV: float,
    bfunc_Kedge: Optional["BFuncKedge"],
    E_grid: Optional[torch.Tensor],
) -> torch.Tensor:
    """
    Render μ(E) for a subset of Gaussians (mask True = keep).
    Uses get_B_E_at_energy on the training E_grid if a K-edge is present.
    Returns (1, H, W) float32 on CUDA.
    """
    device = gaussians._rhos.device

    assert keep_mask.dtype == torch.bool and keep_mask.ndim == 1
    assert keep_mask.numel() == gaussians._rhos.shape[0]

    # Clamp requested energy to grid range (only relevant if K-edge present).
    if (bfunc_Kedge is not None) and (E_grid is not None):
        ek = float(
            torch.clamp(
                torch.tensor(energy_keV, device=device),
                min=E_grid.min(),
                max=E_grid.max(),
            ).item()
        )
    else:
        ek = float(energy_keV)

    raw_backup = gaussians._rhos.detach().clone()
    try:
        # Soft-remove Gaussians not in the mask by pushing rho to ~0 after softplus.
        tiny_raw = torch.full_like(gaussians._rhos, -20.0)  # softplus(-20) ≈ 2e-9
        gaussians._rhos[~keep_mask] = tiny_raw[~keep_mask]

        pkg = render_bcoeffs(view, gaussians, pipe)
        bcoeff_img: torch.Tensor = pkg["bcoeffs"]  # (C, H, W)

        # Build basis vector at ek, but NORMALIZE at a pivot away from the overlay energy
        E  = torch.tensor([ek], device=device, dtype=torch.float32)
        # eg = E_grid if E_grid is not None else torch.tensor([20.0, 140.0], device=device)
        # E_ref_render = float(eg.max().item())  # e.g., 140 keV
        # bfuncs = get_B_E_at_energy(
        #     eg,
        #     bfunc_Kedge if bfunc_Kedge is not None else None,
        #     E,
        #     normalize=True,
        #     E_ref=E_ref_render,              # pivot ≠ overlay energy
        # ).squeeze(0)  # (C,)
        eg = E_grid if E_grid is not None else torch.tensor([20.0, 140.0], device=device)
        E_ref_render = 70.0   # match the training pivot
        bfuncs = get_B_E_at_energy(
            eg,
            bfunc_Kedge if bfunc_Kedge is not None else None,
            E,
            normalize=True,
            E_ref=E_ref_render,
        ).squeeze(0)

        # If model has 3 coeffs but bfuncs has only 2, pad with 1.0.
        if bcoeff_img.shape[0] == 3 and bfuncs.numel() == 2:
            bfuncs = torch.cat(
                [bfuncs, torch.ones(1, device=device, dtype=bfuncs.dtype)], dim=0
            )

        mu = (bcoeff_img * bfuncs[:, None, None]).sum(dim=0, keepdim=True)
        return mu
    finally:
        gaussians._rhos[:] = raw_backup

# ---------------- Entry points: clustering + rendering ---------------- #
@torch.no_grad()
def run_material_clustering(gaussians: GaussianModel, *, bandwidth: Optional[float], pe_is_channel0: bool, prefer_k3: bool) -> MaterialClusteringResult:
    feats = extract_material_features(gaussians, pe_is_channel0=pe_is_channel0)
    Xs, _ = _scale_features_minmax(feats["feats2d"])
    labels, centers = cluster_materials_mean_shift(Xs, bandwidth=bandwidth, prefer_k3=prefer_k3)
    label_to_name  = label_clusters_metal_bone_tissue(Xs, labels)
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
def render_material_views(
    model_path: str,
    split_name: str,
    iteration: int,
    views,
    gaussians: GaussianModel,
    pipe: PipelineParams,
    cluster_result: MaterialClusteringResult,
    *,
    energy_keV: float,
    bfunc_Kedge: Optional[BFuncKedge],
    E_grid: Optional[torch.Tensor],
    blend: bool,
    blend_mode: str = "independent",   # or "proportional"
):
    """
    For each view:
      - render μ for Metal/Bone/Tissue subsets (if present),
      - compose RGB seg image (blend or hard),
      - save PNGs under model_path/materials.
    """
    out_dir = _ensure_dir(os.path.join(model_path, "materials", f"iter_{int(iteration)}", split_name))
    out_metal = _ensure_dir(os.path.join(out_dir, "metal"))
    out_bone  = _ensure_dir(os.path.join(out_dir, "bone"))
    out_tiss  = _ensure_dir(os.path.join(out_dir, "tissue"))
    out_rgb   = _ensure_dir(os.path.join(out_dir, "rgb"))

    # ----- class label ids (avoid substring collisions) -----
    def ids_starting(prefix: str) -> List[int]:
        return [k for k, v in cluster_result.label_to_name.items() if v.startswith(prefix)]

    metal_labels  = ids_starting("Metal (")
    bone_labels   = ids_starting("Bone (")
    tissue_labels = ids_starting("Soft Tissue")

    # If only 2 clusters with "Bone/Metal", treat it as Bone and keep Metal empty.
    if not metal_labels and not bone_labels:
        bone_labels = [k for k, v in cluster_result.label_to_name.items()
                       if v.startswith("Bone/Metal")]

    dev = gaussians._rhos.device
    bone_rgb  = (cluster_result.label_to_color.get(bone_labels[0],  (0,110,255))
                 if bone_labels else (0,110,255))
    tiss_rgb  = (cluster_result.label_to_color.get(tissue_labels[0],(230,60,60))
                 if tissue_labels else (230,60,60))
    metal_rgb = (cluster_result.label_to_color.get(metal_labels[0], (255,215,0))
                 if metal_labels else (255,215,0))

    bone_mask  = torch.from_numpy(np.isin(cluster_result.labels, bone_labels)).to(dev)
    tiss_mask  = torch.from_numpy(np.isin(cluster_result.labels, tissue_labels)).to(dev)
    metal_mask = torch.from_numpy(np.isin(cluster_result.labels, metal_labels)).to(dev)

    for idx, view in enumerate(views):
        # Pre-render to get (H,W)
        pkg0 = render_bcoeffs(view, gaussians, pipe)
        H, W = pkg0["bcoeffs"].shape[-2:]

        mu_metal = _render_cluster_mu(view, gaussians, pipe, metal_mask, energy_keV, bfunc_Kedge, E_grid) if metal_labels else torch.zeros(1, H, W, device=dev)
        mu_bone  = _render_cluster_mu(view, gaussians, pipe, bone_mask,  energy_keV, bfunc_Kedge, E_grid) if bone_labels  else torch.zeros(1, H, W, device=dev)
        mu_tiss  = _render_cluster_mu(view, gaussians, pipe, tiss_mask,  energy_keV, bfunc_Kedge, E_grid) if tissue_labels else torch.zeros(1, H, W, device=dev)

        if blend:
            seg = compose_blend_multi([mu_metal, mu_bone, mu_tiss],
                                      [metal_rgb, bone_rgb, tiss_rgb],
                                      mode=blend_mode)
        else:
            seg = compose_hard_multi([mu_metal, mu_bone, mu_tiss],
                                     [metal_rgb, bone_rgb, tiss_rgb])

        torchvision.utils.save_image(mu_metal, os.path.join(out_metal, f"{idx:05d}.png"))
        torchvision.utils.save_image(mu_bone,  os.path.join(out_bone,  f"{idx:05d}.png"))
        torchvision.utils.save_image(mu_tiss,  os.path.join(out_tiss,  f"{idx:05d}.png"))
        torchvision.utils.save_image(seg,      os.path.join(out_rgb,   f"{idx:05d}.png"))

# -------------------------- CLI convenience -------------------------- #
@torch.no_grad()
def render_materials(
    dataset: ModelParams,
    iteration: int,
    pipeline: PipelineParams,
    *,
    skip_train: bool,
    skip_test: bool,
    n_spectra: int,
    n_bcoeffs: int,
    bandwidth: Optional[float],
    pe_is_channel0: bool,
    prefer_k3: bool,
    energy_keV: float,
    blend: bool,
    blend_mode: str,
) -> None:
    scene = Scene(dataset, n_spectra, n_bcoeffs, load_iteration=iteration, shuffle=False)

    scanner_cfg = scene.scanner_cfg
    volume_to_world = max(scanner_cfg["sVoxel"])
    scale_bound = None
    if dataset.scale_min > 0 and dataset.scale_max > 0:
        scale_bound = np.array([dataset.scale_min, dataset.scale_max]) * volume_to_world
    gaussians = GaussianModel(n_spectra, n_bcoeffs, scale_bound)
    scene.init_gaussians(gaussians)

    # Load learned K-edge for this iteration (if any)
    bfunc_Kedge, E_grid = (None, None)
    if n_bcoeffs >= 3:
        bfunc_Kedge, E_grid = _load_kedge_checkpoint(dataset.model_path, scene.loaded_iter)

    # cluster
    result = run_material_clustering(gaussians, bandwidth=bandwidth, pe_is_channel0=pe_is_channel0, prefer_k3=prefer_k3)

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
            "basis_order": "[PE, Compton, (K-edge)]" if pe_is_channel0 else "[Compton, PE, (K-edge)]",
            "feature_space": "[rho, log1p(pe/comp)] (min–max scaled)",
            "energy_keV": float(energy_keV),
            "prefer_k3": bool(prefer_k3),
            "kedge_loaded": bool(bfunc_Kedge is not None),
            "blend": bool(blend),
            "blend_mode": str(blend_mode),
        }
    }
    with open(os.path.join(out_root, "cluster_info.json"), "w") as f:
        json.dump(info, f, indent=2)

    # render splits
    if not skip_train:
        render_material_views(dataset.model_path, "train", scene.loaded_iter,
                              scene.getTrainCameras(), gaussians, pipeline, result,
                              energy_keV=energy_keV, bfunc_Kedge=bfunc_Kedge, E_grid=E_grid,
                              blend=blend, blend_mode=blend_mode)
    if (not skip_test) and len(scene.getTestCameras()) > 0:
        render_material_views(dataset.model_path, "test", scene.loaded_iter,
                              scene.getTestCameras(), gaussians, pipeline, result,
                              energy_keV=energy_keV, bfunc_Kedge=bfunc_Kedge, E_grid=E_grid,
                              blend=blend, blend_mode=blend_mode)

if __name__ == "__main__":
    parser = ArgumentParser(description="Material rendering (Metal/Bone/Soft) for 3DGS")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--n_spectra", type=int, default=2)
    parser.add_argument("--n_bcoeffs", type=int, default=3)

    # clustering controls
    parser.add_argument("--bandwidth", type=float, default=None, help="Mean-shift bandwidth; None=auto")
    parser.add_argument("--prefer_k3", action="store_true", default=True, help="Prefer 3 clusters when choosing bandwidth")
    parser.add_argument("--pe_is_channel0", action="store_true", default=True, help="If false, assume [Compton, PE, (K)] order")

    # rendering
    parser.add_argument("--energy_keV", type=float, default=70.0, help="Monochromatic overlay energy (keV)")
    parser.add_argument("--hard", action="store_true", help="Use hard per-pixel labels instead of blending")
    parser.add_argument("--blend_mode", type=str, choices=["independent", "proportional"], default="independent",
                        help="Blend strategy for multi-class overlays")

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
        bandwidth=None,           # use CLI
        pe_is_channel0=args.pe_is_channel0,
        prefer_k3=args.prefer_k3,
        energy_keV=args.energy_keV,
        blend=not args.hard,
        blend_mode=args.blend_mode,
    )
