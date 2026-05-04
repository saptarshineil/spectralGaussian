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
from gaussian_renderer import render_bcoeffs, render_r2, render_lighting
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
from utils.spectralCT_utils import (
    BFuncKedge, get_B_E, energy_quadrature, get_scaled_spectrum, get_B_E_at_energy, renormalize_col_at_pivot, bfunc_Compton, bfunc_PE, get_spectral_constants, render_image
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


def render_set(model_path, name, iteration, views, gaussians, pipeline, classifier, n_spectra, n_bcoeffs, scene):
    # Load the same energy grid & weights as training
    if n_bcoeffs == 3:
        # Load K-edge basis checkpoint to recover E_grid
        ckpt = torch.load(os.path.join(model_path, f"kedge_{iteration:06d}.pt"), map_location="cuda")
        E_grid = ckpt["E_grid"].cuda()
        w_k = ckpt["w_k"].cuda()
        bfunc_Kedge = BFuncKedge(E_grid).cuda()
        bfunc_Kedge.load_state_dict(ckpt["state_dict"])
        bfunc_Kedge.eval()
    else:
        # Recreate training quadrature
        N_samples = 64
        E_grid, w_k = energy_quadrature(20.0, 140.0, N_samples)
        bfunc_Kedge = None

    E, _, _, S, W, I0_scalar = get_spectral_constants(E_grid, w_k, spectral_cfg=scene.spectral_cfg)

    # log_I0_bias = torch.tensor([0.0019, 0.0041, 0.0069], device='cuda')
    # log_tilt = torch.tensor([-0.2113, -0.1383,  0.0403], device='cuda')
    # scatter_frac = torch.tensor([-1.9099, -2.4315, -3.9720], device='cuda')

    # -------- render all views for each spectrum --------
    for spectral_id in list(range(0, n_spectra)):
        gaussians.active_spectra = spectral_id
        render_path = os.path.join(model_path, name, f"ours_{iteration}", f"renders_{spectral_id}")
        gts_path    = os.path.join(model_path, name, f"ours_{iteration}", f"gt_{spectral_id}")
        makedirs(render_path, exist_ok=True)
        makedirs(gts_path, exist_ok=True)

        for idx, view in enumerate(tqdm(views, desc=f"Rendering progress (spec {spectral_id})")):
            torch.cuda.synchronize()
            results = render_bcoeffs(view, gaussians, pipeline)
            bcoeff_image = results["bcoeffs"]
            torch.cuda.synchronize()

            rendering, _ = render_image(bcoeff_image, bfunc_Kedge, E, E_grid, w_k, S, W, I0_scalar, n_bcoeffs, spectral_id)
            # rendering, _ = render_image(bcoeff_image, bfunc_Kedge, E, E_grid, w_k, S, W, I0_scalar, n_bcoeffs, spectral_id, scatter_frac)
            # rendering, _ = render_image(bcoeff_image, bfunc_Kedge, E, E_grid, w_k, S, W, I0_scalar, n_bcoeffs, spectral_id, log_tilt)
            # rendering = rendering + log_I0_bias[spectral_id]

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

        print(len(scene.getTestCameras()))

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