# Copyright (C) 2023, Gaussian-Grouping
# Gaussian-Grouping research group, https://github.com/lkeab/gaussian-grouping
# All rights reserved.
#
# ------------------------------------------------------------------------
# Modified from codes in Gaussian-Splatting 
# GRAPHDECO research group, https://team.inria.fr/graphdeco

import torch
from scene import Scene
import os
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render_mu, render_r2, render_lighting
import torchvision
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel
import numpy as np
from PIL import Image
import colorsys
import cv2
from sklearn.decomposition import PCA
from scene.NVDIFFREC.util import save_image_raw

import matplotlib.pyplot as plt
# from utils.spectralCT_utils import bfunc_Compton, bfunc_PE, BFuncKedge
from utils.spectralCT_utils import (
    BFuncKedge, get_B_E, energy_quadrature, get_scaled_spectrum, get_B_E_at_energy, renormalize_col_at_pivot
)
from utils.general_utils import interp1d


def orthogonalize_k_to_pe_compton(B_raw: torch.Tensor, W: torch.Tensor) -> torch.Tensor:
    if B_raw.shape[1] < 3:
        return B_raw

    B12 = B_raw[:, :2]         # (K, 2)
    b3  = B_raw[:, 2:3]        # (K, 1)

    # M = B12^T W B12, rhs = B12^T W b3
    BW  = B12 * W[:, None]
    M   = BW.T @ B12 + 1e-6 * torch.eye(2, device=B_raw.device, dtype=B_raw.dtype)
    rhs = BW.T @ b3

    # Solve M c = rhs, project out: b3_orth = b3 - B12 c
    c   = torch.cholesky_solve(rhs, torch.linalg.cholesky(M))
    b3o = b3 - B12 @ c

    # optional: renormalize at pivot (handled by get_B_E via _col_scales_at_pivot)
    Bout = torch.cat([B12, b3o], dim=1)
    return Bout


def feature_to_rgb(features):
    # Input features shape: (16, H, W)
    
    # Reshape features for PCA
    H, W = features.shape[1], features.shape[2]
    features_reshaped = features.view(features.shape[0], -1).T

    # Apply PCA and get the first 3 components
    pca = PCA(n_components=3)
    pca_result = pca.fit_transform(features_reshaped.cpu().numpy())

    # Reshape back to (H, W, 3)
    pca_result = pca_result.reshape(H, W, 3)

    # Normalize to [0, 255]
    pca_normalized = 255 * (pca_result - pca_result.min()) / (pca_result.max() - pca_result.min())

    rgb_array = pca_normalized.astype('uint8')

    return rgb_array

def id2rgb(id, max_num_obj=256):
    if not 0 <= id <= max_num_obj:
        raise ValueError("ID should be in range(0, max_num_obj)")

    # Convert the ID into a hue value
    golden_ratio = 1.6180339887
    h = ((id * golden_ratio) % 1)           # Ensure value is between 0 and 1
    s = 0.5 + (id % 2) * 0.5       # Alternate between 0.5 and 1.0
    l = 0.5

    
    # Use colorsys to convert HSL to RGB
    rgb = np.zeros((3, ), dtype=np.uint8)
    if id==0:   #invalid region
        return rgb
    r, g, b = colorsys.hls_to_rgb(h, l, s)
    rgb[0], rgb[1], rgb[2] = int(r*255), int(g*255), int(b*255)

    return rgb

def visualize_obj(objects):
    rgb_mask = np.zeros((*objects.shape[-2:], 3), dtype=np.uint8)
    all_obj_ids = np.unique(objects)
    for id in all_obj_ids:
        colored_mask = id2rgb(id)
        rgb_mask[objects == id] = colored_mask
    return rgb_mask


def render_lightings(model_path, name, iteration, gaussians, sample_num, n_spectra):
    for spectral_id in list(range(0, n_spectra + 1)):
        lighting_path = os.path.join(model_path, name, "ours_{}_{}".format(iteration,spectral_id))
        makedirs(lighting_path, exist_ok=True)    
        # sampled_indicies = torch.randperm(gaussians.get_xyz.shape[0])[:sample_num]
        sampled_indicies = torch.arange(gaussians.get_xyz.shape[0], dtype=torch.long)[:sample_num]
        for sampled_index in tqdm(sampled_indicies, desc="Rendering lighting progress"):
            lighting = render_lighting(gaussians, sampled_index=sampled_index, specral_id=spectral_id)
            torchvision.utils.save_image(lighting, os.path.join(lighting_path, '{0:05d}'.format(sampled_index) + ".png"))
            save_image_raw(os.path.join(lighting_path, '{0:05d}'.format(sampled_index) + ".hdr"), lighting.permute(1,2,0).detach().cpu().numpy())


# def render_set(model_path, name, iteration, views, gaussians, pipeline, classifier, n_spectra, n_bcoeffs, scene):
#     # TEMP: removed segmentation
#     # colormask_path = os.path.join(model_path, name, "ours_{}".format(iteration), "objects_feature16")
#     # gt_colormask_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt_objects_color")
#     # pred_obj_path = os.path.join(model_path, name, "ours_{}".format(iteration), "objects_pred")
#     # makedirs(colormask_path, exist_ok=True)
#     # makedirs(gt_colormask_path, exist_ok=True)
#     # makedirs(pred_obj_path, exist_ok=True)

#     #need to later make this per spectra
#     # TEMP: removed segmentation
#     # for idx, view in enumerate(tqdm(views, desc="segmentation rendering progress")):
#     #         torch.cuda.synchronize()
#     #         results = render(view, gaussians, pipeline, background, debug=True, spectral_id=0)
#     #         torch.cuda.synchronize()
#     #         rendering_obj = results["render_object"]
            
#     #         logits = classifier(rendering_obj)
#     #         pred_obj = torch.argmax(logits,dim=0)
#     #         pred_obj_mask = visualize_obj(pred_obj.cpu().numpy().astype(np.uint8))
            

#     #         gt_objects = view.objects
#     #         gt_rgb_mask = visualize_obj(gt_objects.cpu().numpy().astype(np.uint8))

#     #         rgb_mask = feature_to_rgb(rendering_obj)
#     #         Image.fromarray(rgb_mask).save(os.path.join(colormask_path, '{0:05d}'.format(idx) + ".png"))
#     #         Image.fromarray(gt_rgb_mask).save(os.path.join(gt_colormask_path, '{0:05d}'.format(idx) + ".png"))
#     #         Image.fromarray(pred_obj_mask).save(os.path.join(pred_obj_path, '{0:05d}'.format(idx) + ".png"))

#     # E = torch.tensor([50.0, 120.0], device="cuda")
#     E = torch.tensor([50.0, 80.0, 120.0], device="cuda")
#     # E_keV  = torch.tensor([50.0, 80.0], device="cuda")

#     if n_bcoeffs == 3:
#         ckpt = torch.load(os.path.join(model_path, f"kedge_{iteration:06d}.pt"),
#                           map_location="cuda")
#         E_grid = ckpt["E_grid"].cuda()
#         bfunc_K = BFuncKedge(E_grid).cuda()
#         bfunc_K.load_state_dict(ckpt["state_dict"])
#         bfunc_K.eval()
#         # uniform mid-point rule weights from stored grid
#         # dE = (E_grid[-1] - E_grid[0]) / E_grid.numel()
#         # w_k = torch.full_like(E_grid, dE)
#         dE = (E_grid[1] - E_grid[0]).clamp_min(1e-6)
#         w_k = torch.full_like(E_grid, dE)
#     else:
#         # use the same quadrature as training
#         N_samples = 64
#         E_grid, w_k = energy_quadrature(20.0, 140.0, N_samples)
#         bfunc_K = None

#     I0_low  = 1136.8784583333334
#     I0_mid = 1594.836
#     I0_high = 2026.8482152777776

#     E_grid_np = E_grid.detach().cpu().numpy()
#     w_k_np = w_k.detach().cpu().numpy()

#     S_low  = get_scaled_spectrum( 50, None, 0.0, I0_low,  E_grid_np, w_k_np)
#     S_mid = get_scaled_spectrum(80, "Al", 1.0, I0_mid, E_grid_np, w_k_np)
#     S_high = get_scaled_spectrum(120, "Cu", 0.5, I0_high, E_grid_np, w_k_np)

#     # S = torch.stack([S_low, S_high], dim=0)                             # (C_spectra, K)
#     S = torch.stack([S_low, S_mid, S_high], dim=0)                             # (C_spectra, K)
#     I0_scalar = torch.sum(S * w_k, dim=1)

#     with torch.no_grad():
#         B_E = get_B_E(E_grid, bfunc_K)

#         # B_E_raw = get_B_E(E_grid, bfunc_K if n_bcoeffs == 3 else None, normalize=True, E_ref=70.0)
#         # if n_bcoeffs == 3:
#         #     S_avg = 0.33 * (S[0] + S[1] + S[2])
#         #     W = (w_k * S_avg).clamp_min(1e-12)
#         #     B_E = orthogonalize_k_to_pe_compton(B_E_raw, W)
#         # else:
#         #     B_E = B_E_raw

#         # B_E_unnorm = get_B_E(E_grid, bfunc_K if n_bcoeffs == 3 else None, normalize=False) 
            
#         # if n_bcoeffs == 3: 
#         #     S_sum = S.sum(dim=0) # (K,) 
#         #     W = (w_k * S_sum).clamp_min(1e-12)
#         #     B12 = B_E_unnorm[:, :2] # (K,2) 
#         #     b3 = B_E_unnorm[:, 2:3] # (K,1) 
#         #     BW = B12 * W[:, None] 
#         #     M = BW.T @ B12 + 1e-6 * torch.eye(2, device=B12.device, dtype=B12.dtype) 
#         #     rhs = BW.T @ b3 
#         #     c = torch.cholesky_solve(rhs, torch.linalg.cholesky(M)) # (2,1) # Orthogonalized columns (still UNNORMALIZED) 
#         #     B_E_ortho = torch.cat([B12, b3 - B12 @ c], dim=1) # Now normalize *all* columns at the same pivot 
#         #     # s_pe = renormalize_col_at_pivot(E_grid, B_E_ortho, 0, E_ref=70.0) 
#         #     # s_co = renormalize_col_at_pivot(E_grid, B_E_ortho, 1, E_ref=70.0) 
#         #     # s_ke = renormalize_col_at_pivot(E_grid, B_E_ortho, 2, E_ref=70.0) 
#         #     B_E, s = renormalize_col_at_pivot(E_grid, B_E_ortho, E_ref=70.0)
#         #     s_pe, s_co, s_ke = s.unbind()
#         #     # B_E = B_E_ortho 
#         # else: # 2-basis: just normalize at the pivot 
#         #     B_E = get_B_E(E_grid, None, normalize=True, E_ref=70.0)

        

#     for spectral_id in list(range(0, n_spectra)):
#         gaussians.active_spectra = spectral_id
#         render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders_" + str(spectral_id))
#         gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt_" + str(spectral_id))

#         makedirs(render_path, exist_ok=True)
#         makedirs(gts_path, exist_ok=True)

#         # E_s = E[spectral_id]
#         # bfuncs = get_B_E_at_energy(E, None, E_s)

#         for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
#             torch.cuda.synchronize()
#             results = render_mu(view, gaussians, pipeline)
#             bcoeff_img = results["bcoeffs"]
#             torch.cuda.synchronize()

#             # rendering = (bcoeff_img * bfuncs[:, None, None]).sum(dim=0, keepdim=True)
            
#             mu_E = (bcoeff_img.unsqueeze(0) * B_E[..., None, None]).sum(dim=1)

#             I_pred = torch.sum(S[spectral_id, :, None, None] *
#                                torch.exp(-mu_E) * w_k[:, None, None],
#                                dim=0)

#             I_pred = torch.clamp(I_pred, min=1e-6 * I0_scalar[spectral_id])
#             rendering  = -torch.log(I_pred / I0_scalar[spectral_id]).unsqueeze(0)

#             gt = view.spectral_images[spectral_id][0:1,:,:]
            
#             torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
#             torchvision.utils.save_image(gt, os.path.join(gts_path, '{0:05d}'.format(idx) + ".png"))

def render_set(model_path, name, iteration, views, gaussians, pipeline, classifier, n_spectra, n_bcoeffs, scene):
    import torch
    import os
    import torchvision
    from tqdm import tqdm
    from os import makedirs
    from utils.spectralCT_utils import (
        BFuncKedge, get_B_E, energy_quadrature, get_scaled_spectrum, renormalize_col_at_pivot
    )

    # ---- helpers (mirroring train.py) ----
    def whiten_basis(B_raw: torch.Tensor,
                     W: torch.Tensor,
                     E_grid: torch.Tensor,
                     E_ref: float = 70.0,
                     eps: float = 1e-6):
        """
        Project K-edge off PE/Compton with W-weighted LS, then normalize columns at E_ref.
        Args:
            B_raw: (K,3) columns [PE, Compton, Kraw]
            W:     (K,)  non-negative weights (e.g., w_k * spectrum_sum)
            E_grid:(K,)
        Returns:
            B_w: (K,3) whitened+normalized basis
        """
        B12, b3 = B_raw[:, :2], B_raw[:, 2:3]                      # (K,2), (K,1)
        BW = B12 * W[:, None]
        M = BW.T @ B12 + eps * torch.eye(2, device=B_raw.device, dtype=B_raw.dtype)
        rhs = BW.T @ b3
        c = torch.cholesky_solve(rhs, torch.linalg.cholesky(M))    # (2,1)
        B_ortho = torch.cat([B12, b3 - B12 @ c], dim=1)            # (K,3)
        B_w, _s = renormalize_col_at_pivot(E_grid, B_ortho, E_ref=E_ref)
        return B_w

    # ---------------------------------------------------------------

    # Rendering energies (used only to pick spectra indices to output)
    E_render = torch.tensor([50.0, 80.0, 120.0], device="cuda", dtype=torch.float32)

    # X-ray fluence targets (match training)
    I0_low  = 1136.8784583333334
    I0_mid  = 1594.836
    I0_high = 2026.8482152777776

    # Load the same energy grid & weights as training
    if n_bcoeffs == 3:
        # Load K-edge basis checkpoint to recover E_grid; build uniform midpoint weights
        ckpt = torch.load(os.path.join(model_path, f"kedge_{iteration:06d}.pt"), map_location="cuda")
        E_grid = ckpt["E_grid"].cuda()
        bfunc_K = BFuncKedge(E_grid).cuda()
        bfunc_K.load_state_dict(ckpt["state_dict"])
        bfunc_K.eval()
        # approximate midpoint weights from stored uniform grid
        dE = (E_grid[1] - E_grid[0]).clamp_min(1e-6)
        w_k = torch.full_like(E_grid, dE)
    else:
        # Recreate training quadrature
        N_samples = 64
        E_grid, w_k = energy_quadrature(20.0, 140.0, N_samples)
        bfunc_K = None

    # Build the tube spectra on the same grid
    E_grid_np = E_grid.detach().cpu().numpy()
    w_k_np    = w_k.detach().cpu().numpy()
    S_low  = get_scaled_spectrum( 50, None, 0.0, I0_low,  E_grid_np, w_k_np)
    S_mid  = get_scaled_spectrum( 80, "Al", 1.0, I0_mid,  E_grid_np, w_k_np)
    S_high = get_scaled_spectrum(120, "Cu", 0.5, I0_high, E_grid_np, w_k_np)
    S = torch.stack([S_low, S_mid, S_high], dim=0)                 # (C=3, K)
    I0_scalar = torch.sum(S * w_k, dim=1)                          # (3,)

    # Build the whitened basis ONCE for this render pass
    with torch.no_grad():
        if n_bcoeffs == 3:
            # Unnormalized physical basis at all energies
            B_raw = get_B_E(E_grid, bfunc_K, normalize=False)      # (K,3)
            # Spectrum-weighted orthogonalization (sum spectra like in training)
            W = (w_k * S.sum(dim=0)).clamp_min(1e-12)              # (K,)
            B_E = whiten_basis(B_raw, W, E_grid, E_ref=70.0)       # (K,3) whitened+normalized
        else:
            # 2-basis: just normalize both columns at the pivot
            B_E = get_B_E(E_grid, None, normalize=True, E_ref=70.0)  # (K,2)

    # -------- render all views for each spectrum --------
    for spectral_id in list(range(0, n_spectra)):
        gaussians.active_spectra = spectral_id
        render_path = os.path.join(model_path, name, f"ours_{iteration}", f"renders_{spectral_id}")
        gts_path    = os.path.join(model_path, name, f"ours_{iteration}", f"gt_{spectral_id}")
        makedirs(render_path, exist_ok=True)
        makedirs(gts_path, exist_ok=True)

        for idx, view in enumerate(tqdm(views, desc=f"Rendering progress (spec {spectral_id})")):
            torch.cuda.synchronize()
            results = render_mu(view, gaussians, pipeline)
            bcoeff_img = results["bcoeffs"]                         # (C,H,W) where C==n_bcoeffs
            torch.cuda.synchronize()

            # Forward Beer–Lambert with whitened basis integrated over spectrum
            mu_E = (bcoeff_img.unsqueeze(0) * B_E[..., None, None]).sum(dim=1)  # (K,H,W)

            I_pred = torch.sum(
                S[spectral_id, :, None, None] * torch.exp(-mu_E) * w_k[:, None, None],
                dim=0,
            )  # (H,W)

            I_pred = torch.clamp(I_pred, min=1e-6 * I0_scalar[spectral_id])
            rendering = -torch.log(I_pred / I0_scalar[spectral_id]).unsqueeze(0)  # (1,H,W)

            gt = view.spectral_images[spectral_id][0:1, :, :]

            torchvision.utils.save_image(rendering, os.path.join(render_path, f"{idx:05d}.png"))
            torchvision.utils.save_image(gt,        os.path.join(gts_path,    f"{idx:05d}.png"))



def render_sets(dataset : ModelParams, iteration : int, pipeline : PipelineParams, skip_train : bool, skip_test : bool, n_spectra : int, n_bcoeffs : int):
    
    with torch.no_grad():
        scene = Scene(dataset, n_spectra, n_bcoeffs, load_iteration=iteration, shuffle=False)
        
        scanner_cfg       = scene.scanner_cfg
        volume_to_world   = max(scanner_cfg["sVoxel"])
        scale_bound = None
        if dataset.scale_min > 0 and dataset.scale_max > 0:
            scale_bound = np.array([dataset.scale_min,
                                    dataset.scale_max]) * volume_to_world
            
        gaussians = GaussianModel(n_spectra, n_bcoeffs, scale_bound)

        scene.init_gaussians(gaussians)

        print("Loaded", gaussians.get_xyz.shape[0], "Gaussians")

        # TEMP: removed segmentation
        # num_classes = dataset.num_classes
        # print("Num classes: ",num_classes)

        # TEMP: removed segmentation
        classifier = None
        # classifier = torch.nn.Conv2d(gaussians.num_objects, num_classes, kernel_size=1)
        # classifier.cuda()
        # classifier.load_state_dict(torch.load(os.path.join(dataset.model_path,"point_cloud","iteration_"+str(scene.loaded_iter)+"_" + str(n_spectra),"classifier.pth")))

        if not skip_train:
            render_set(dataset.model_path, "train", scene.loaded_iter, scene.getTrainCameras(), gaussians, pipeline, classifier, n_spectra, n_bcoeffs, scene)

        if (not skip_test) and (len(scene.getTestCameras()) > 0):
            render_set(dataset.model_path, "test", scene.loaded_iter, scene.getTestCameras(), gaussians, pipeline, classifier, n_spectra, n_bcoeffs, scene)

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--n_spectra", type=int, default=2)
    parser.add_argument("--n_bcoeffs", type=int, default=2)
    args = get_combined_args(parser)
    print("Rendering " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    render_sets(
        model.extract(args), 
        args.iteration, 
        pipeline.extract(args), 
        args.skip_train, 
        args.skip_test, 
        args.n_spectra,
        args.n_bcoeffs,
    )
        # render_lightings