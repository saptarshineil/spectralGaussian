# Copyright (C) 2023, Gaussian-Grouping
# Gaussian-Grouping research group, https://github.com/lkeab/gaussian-grouping
# All rights reserved.
#
# ------------------------------------------------------------------------
# Modified from codes in Gaussian-Splatting 
# GRAPHDECO research group, https://team.inria.fr/graphdeco

import os
import torch
from random import randint
from utils.loss_utils import (
    l1_loss, ssim, tv_3d_loss,
    joint_tv, tnv_tv, linf_tv
)

# from utils.loss_utils import l1_loss, ssim, tv_3d_loss, loss_cls_3d, predicted_normal_loss, delta_normal_loss, zero_one_loss
from gaussian_renderer import render_r2, render_mu, render_fracs, network_gui
from utils.spectralCT_utils import BFuncKedge, get_B_E, get_B_E_at_energy, energy_quadrature, get_scaled_spectrum
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
import wandb
import json
import torchvision
import numpy as np
from gaussian_renderer.render_query import query #, render

import matplotlib.pyplot as plt
# from utils.spectralCT_utils import bfunc_PE, bfunc_Compton, closest_material
import tifffile as tiff

def _print_grad_stats(tag, g : torch.Tensor):
    if g is None:
        print(f"[grad] {tag:<12s} –- no grad yet")
        return
    g_abs = g.detach().abs()          # |∂L/∂w|
    print(
        f"[grad] {tag:<12s}"
        f"  mean {g_abs.mean():.3e}"
        f"  med {g_abs.median():.3e}"
        f"  p90  {torch.quantile(g_abs, .90):.3e}"
        f"  max  {g_abs.max():.3e}"
    )


def training(
    dataset: ModelParams, 
    opt: OptimizationParams, 
    pipe: PipelineParams, 
    n_spectra,
    n_bcoeffs, 
    testing_iterations, 
    saving_iterations, 
    checkpoint_iterations, 
    checkpoint, 
    debug_from, 
    use_wandb, 
    use_warm_up_iteration
):
    first_iter = 0
    prepare_output_and_logger(dataset)

    # Initialize Scene according to R2-Gaussian (Initialize CT parameters)
    scene = Scene(dataset, n_spectra, n_bcoeffs)

    # Set up CT parameters
    scanner_cfg = scene.scanner_cfg
    bbox = scene.bbox
    volume_to_world = max(scanner_cfg["sVoxel"])
    max_scale = opt.max_scale * volume_to_world if opt.max_scale else None
    densify_scale_threshold = (
        opt.densify_scale_threshold * volume_to_world
        if opt.densify_scale_threshold
        else None
    )
    scale_bound = None
    if dataset.scale_min > 0 and dataset.scale_max > 0:
        scale_bound = np.array([dataset.scale_min, dataset.scale_max]) * volume_to_world

    # ------------------------------------------
    # Set up Gaussians
    gaussians = GaussianModel(
        n_spectra,
        n_bcoeffs,
        scale_bound
    )
    scene.init_gaussians(gaussians)
    gaussians.training_setup(opt)

    
    # E = torch.tensor([50.0, 70.0], dtype=torch.float32).cuda()
    E = torch.tensor([50.0, 120.0], device='cuda', dtype=torch.float32)
    # E = torch.tensor([50.0, 80.0], device='cuda', dtype=torch.float32)

    I0_low = 1136.8784583333334
    I0_high = 2026.8482152777776
    # I0_high = 1594.836

    N_samples = 64
    # If dataset is EID (no bins) --> sample full spectrum
    E_grid, w_k = energy_quadrature(20.0, 140.0, N_samples)
    E_grid_np = E_grid.cpu().numpy()

    S_low  = get_scaled_spectrum( 50, None, 0.0, I0_low,  E_grid_np)
    S_high = get_scaled_spectrum(120, "Cu", 0.5, I0_high, E_grid_np)
    # S_high = get_scaled_spectrum(80, "Al", 1.0, I0_high, E_grid_np)

    S         = torch.stack([S_low, S_high])      # (C=2, K)
    I0_scalar = torch.sum(S * w_k, dim=1)


    # ------------------------------------------
    bfunc_Kedge = None
    basis_opt = None
    # if n_bcoeffs == 3:
    #     bfunc_Kedge = BFuncKedge(E_grid).cuda()
    #     basis_opt = torch.optim.Adam(bfunc_Kedge.parameters(), lr=1e-4)

    B_E = get_B_E(E_grid, bfunc_Kedge)

    # B_E_raw = get_B_E(E_grid, None, normalize=True, E_ref=70.0)  # (K,C)
    # S_avg   = 0.5 * (S[0] + S[1])
    # W = (w_k * S_avg).clamp_min(1e-12)
    # G = (B_E_raw * W[:, None]).T @ B_E_raw
    # G = G + 1e-6 * torch.eye(G.shape[0], device=G.device)
    # L = torch.linalg.cholesky(G)
    # B_E = B_E_raw @ torch.cholesky_inverse(L)
    
    
    energy_logits = torch.nn.Parameter(
        torch.tensor([1.0, 1.0], device='cuda', dtype=torch.float32)).requires_grad_(False)

    FRAC_WARMUP_ITERS = getattr(opt, "frac_warmup_iters", 1000)
    gaussians.freeze_rhos()

    # --------------------------------------
    # Set up 3D TV loss
    use_tv = opt.lambda_tv > 0
    # use_tv = False
    if use_tv:
        print("Use total variation loss")
        tv_vol_size = opt.tv_vol_size
        tv_vol_nVoxel = torch.tensor([tv_vol_size, tv_vol_size, tv_vol_size])
        tv_vol_sVoxel = torch.tensor(scanner_cfg["dVoxel"]) * tv_vol_nVoxel
    

    # Set up training loop
    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    viewpoint_stack = None
    ema_loss_for_log = 0.0
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1
    if use_warm_up_iteration:
        combine_spectra = False
        spectral_ids = list(range(0, n_spectra)) if n_spectra > 0 else [0]
    else:
        combine_spectra = True
        spectral_ids = list(range(0, n_spectra)) if n_spectra > 0 else [0]

    # Start training loop
    for iteration in range(first_iter, opt.iterations + 1):   
        iter_start.record()

        # Debug
        if (iteration - 1) == debug_from:
            pipe.debug = True

        # Get one camera for training
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))

        if (iteration - 1) == FRAC_WARMUP_ITERS:
            gaussians.unfreeze_rhos()
            gaussians.update_learning_rate(iteration)
            print(f"[warmup] Unfroze ρ at iter {iteration}, lr now scheduled.")
        
        # Track losses
        # total_loss = 0.0
        losses_per_energy = []

        render_pkg = render_mu(viewpoint_cam, gaussians, pipe)
        bcoeff_image, viewspace_point_tensor, visibility_filter, radii = (
            render_pkg["bcoeffs"], 
            render_pkg["viewspace_points"], 
            render_pkg["visibility_filter"], 
            render_pkg["radii"], 
        )

        tv_pe   = joint_tv(bcoeff_image[0:1], reduction='mean')
        tv_co   = joint_tv(bcoeff_image[1:2], reduction='mean')
        bcoeff_loss  = opt.lambda_tv_basis_pe * tv_pe + opt.lambda_tv_basis_comp * tv_co

        # Compute loss per spectral channel
        for spectral_id in spectral_ids:
            
            # ---------- Spectral Beer-Lambert ----------
            bfuncs = get_B_E_at_energy(E, None, E[spectral_id])
            
            # image = (bcoeff_image * bfuncs[:, None, None]).sum(dim=0, keepdim=True)

            μ_E = (bcoeff_image.unsqueeze(0) * B_E[..., None, None]).sum(1)

            I_pred = torch.sum(S[spectral_id,:,None,None] *
                            torch.exp(-μ_E) * w_k[:,None,None],
                            dim=0)
            I_pred = torch.clamp(I_pred, min=1e-6 * I0_scalar[spectral_id])
            image = -torch.log(I_pred / I0_scalar[spectral_id]).unsqueeze(0)

            # ---------- Loss ----------
            # L1 Loss
            gt_image = viewpoint_cam.spectral_images[spectral_id].cuda()

            # if (iteration - 1) % 500 == 0:
            #     print(spectral_id, ":")
            #     print("μ mean/gt mean:", image[0].mean().item(), gt_image[0].mean().item())
            #     print("μ min/max     :", image[0].min().item(), image[0].max().item())
            #     print("gt min/max    :", gt_image[0].min().item(), gt_image[0].max().item())
            #     with torch.no_grad():
            #         # Per-basis contribution images: C×H×W
            #         contrib = bcoeff_image * bfuncs[:, None, None]

            #         # Quick scalars per basis
            #         mean_per_basis = contrib.mean(dim=(1, 2)).tolist()
            #         min_per_basis  = contrib.amin(dim=(1, 2)).tolist()
            #         max_per_basis  = contrib.amax(dim=(1, 2)).tolist()

            #         print("μ per-basis mean:", [float(x) for x in mean_per_basis])
            #         print("μ per-basis min :", [float(x) for x in min_per_basis])
            #         print("μ per-basis max :", [float(x) for x in max_per_basis])
            #     print("---------------")

            if (iteration - 1) % 1000 == 0 or iteration == opt.iterations:
                with torch.no_grad():
                    fr = render_fracs(viewpoint_cam, gaussians, pipe)
                    fracs_img = fr["fracs"].detach().cpu()     # (C,H,W)
                    out_dir = os.path.join(scene.model_path, "analysis", "fracs")
                    os.makedirs(out_dir, exist_ok=True)

                    if fracs_img.shape[0] == 2:
                        f_pe, f_comp = fracs_img[0], fracs_img[1]
                        ratio = (f_pe + 1e-8) / (f_comp + 1e-8)
                        torchvision.utils.save_image(f_pe[None],   os.path.join(out_dir, f"f_pe_{iteration:06d}.png"))
                        torchvision.utils.save_image(f_comp[None], os.path.join(out_dir, f"f_comp_{iteration:06d}.png"))
                        r = ratio.numpy()
                        lo, hi = np.percentile(r[np.isfinite(r)], [1, 99])
                        r = np.clip((r - lo) / (hi - lo + 1e-12), 0, 1)
                        torchvision.utils.save_image(torch.from_numpy(r)[None], os.path.join(out_dir, f"ratio_{iteration:06d}.png"))
                    else:
                        f = fracs_img.numpy()
                        f = f / (f.sum(axis=0, keepdims=True) + 1e-8)
                        rgb = np.moveaxis(f, 0, -1)  # (H,W,3)
                        torchvision.utils.save_image(torch.from_numpy(rgb).permute(2,0,1), os.path.join(out_dir, f"fracs_rgb_{iteration:06d}.png"))

                    bc = render_mu(viewpoint_cam, gaussians, pipe)
                    bcoeffs_img = bc["bcoeffs"].detach().cpu()  
                    out_dir = os.path.join(scene.model_path, "analysis", "bcoeffs")
                    os.makedirs(out_dir, exist_ok=True)

                    if bcoeffs_img.shape[0] == 2:
                        f_pe, f_comp = bcoeffs_img[0], bcoeffs_img[1]
                        ratio = (f_pe + 1e-8) / (f_comp + 1e-8)
                        torchvision.utils.save_image(f_pe[None],   os.path.join(out_dir, f"f_pe_{iteration:06d}.png"))
                        torchvision.utils.save_image(f_comp[None], os.path.join(out_dir, f"f_comp_{iteration:06d}.png"))
                        r = ratio.numpy()
                        lo, hi = np.percentile(r[np.isfinite(r)], [1, 99])
                        r = np.clip((r - lo) / (hi - lo + 1e-12), 0, 1)
                        torchvision.utils.save_image(torch.from_numpy(r)[None], os.path.join(out_dir, f"ratio_{iteration:06d}.png"))
                    else:
                        f = bcoeffs_img.numpy()
                        f = f / (f.sum(axis=0, keepdims=True) + 1e-8)
                        rgb = np.moveaxis(f, 0, -1)  # (H,W,3)
                        torchvision.utils.save_image(torch.from_numpy(rgb).permute(2,0,1), os.path.join(out_dir, f"fracs_rgb_{iteration:06d}.png"))


            loss_l1 = l1_loss(image, gt_image)
            loss = (1.0 - opt.lambda_dssim) * loss_l1

            # DSSIM Loss
            loss_dssim = (1.0 - ssim(image, gt_image))
            loss += opt.lambda_dssim * loss_dssim

            # R2-Gaussian
            # 3D TV loss
            if use_tv:
                # Randomly get the tiny volume center
                tv_vol_center = (bbox[0] + tv_vol_sVoxel / 2) + (
                    bbox[1] - tv_vol_sVoxel - bbox[0]
                ) * torch.rand(3)

                vol_pred = query(
                    gaussians,
                    tv_vol_center,
                    tv_vol_nVoxel,
                    tv_vol_sVoxel,
                    pipe,
                    bfuncs,
                    spectral_id,
                )["ct"]
                loss_tv = tv_3d_loss(vol_pred, reduction="mean")
                loss += opt.lambda_tv * loss_tv

            losses_per_energy.append(loss)

        total_loss = (energy_logits[:len(losses_per_energy)] * torch.stack(losses_per_energy)).sum()
        total_loss += bcoeff_loss

        p = gaussians.get_fracs.clamp_min(1e-6)
        entropy = -(p * p.log()).sum(dim=1).mean()
        total_loss += 1e-3 * entropy

        total_loss.backward()
       
        iter_end.record()

        if (iteration - 1) % 500 == 0:
            _print_grad_stats("rhos", gaussians._rhos.grad)
            for j in gaussians.bcoeff_ids:
                _print_grad_stats(f"fracs_{j}", gaussians._fracs[j].grad)
            f = gaussians.get_fracs 
            f0 = f[:, 0]
            f1 = f[:, 1]
            print("mean f0, f1:", f0.mean().item(), f1.mean().item())
            # print("cos(f0, f1):",
            #     torch.nn.functional.cosine_similarity(f0, f1, dim=0).item())
            g0 = gaussians._fracs[0].grad.view(-1)
            g1 = gaussians._fracs[1].grad.view(-1)
            print("mean g0, g1:", g0.mean().item(), g1.mean().item())
            print("cos(g0, -g1):",
                torch.nn.functional.cosine_similarity(g0, -g1, dim=0).item())
            print("RHOS MEAN: ", gaussians.get_rhos.mean().item())
            print("NUM GAUSSIANS: ", len(gaussians.get_xyz))
            print("-------------------")
        
        with torch.no_grad():
            torch.cuda.empty_cache()
            
            # Progress bar
            ema_loss_for_log = 0.4 * total_loss + 0.6 * ema_loss_for_log
            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}"})
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # Log and save
            #todo : implement evaluation. Important for Evaluation
            # training_report(iteration, total_l1_loss, total_loss, l1_loss, iter_start.elapsed_time(iter_end), testing_iterations, scene, render, (pipe, background), loss_obj_3d, use_wandb, spectral_ids)
            # todo:check the with gt objects for each spectra and save the classifier path per spectra 
            # losses_extra['psnr'] = psnr(image, gt_image).mean()
            if (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)
                # TEMP: removed segmentation
                # torch.save(classifier.state_dict(), os.path.join(scene.model_path, "point_cloud/iteration_{}_{}".format(iteration, spectral_id),'classifier.pth'))

            # # Densification
            # if iteration < opt.densify_until_iter:
            #     # Keep track of max radii in image-space for pruning
            #     gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
            #     gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)
            #     if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
            #         size_threshold = 20 if iteration > opt.opacity_reset_interval else None
            #         gaussians.densify_and_prune(opt.densify_grad_threshold, 0.005, scene.cameras_extent, size_threshold)
                
            #     if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
            #         gaussians.reset_opacity()

            # R2-Gaussian uses adaptive control instead of densification
            # Adaptive control
            gaussians.max_radii2D[visibility_filter] = torch.max(
                gaussians.max_radii2D[visibility_filter], radii[visibility_filter]
            )
            gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)
            
            if iteration < opt.densify_until_iter:
                if (
                    iteration > opt.densify_from_iter
                    and iteration % opt.densification_interval == 0
                ):
                    # _print_grad_stats("ρ / opacity", gaussians._rhos.grad)       # density / opacity
                    # min_density_for_prune = 0.0 if gaussians.rhos_are_frozen() else opt.density_min_threshold
                    gaussians.densify_and_prune(
                        opt.densify_grad_threshold,
                        opt.density_min_threshold,
                        # min_density_for_prune,
                        scene.cameras_extent,
                        opt.max_screen_size,
                        max_scale,
                        opt.max_num_gaussians,
                        densify_scale_threshold,
                        bbox,
                    )
            if gaussians.get_bcoeffs.shape[0] == 0:
                raise ValueError(
                    "No Gaussian left. Change adaptive control hyperparameters!"
                )
            if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                if not gaussians.rhos_are_frozen():
                    gaussians.reset_bcoeffs()
                    # t = iteration / float(opt.iterations)
                    # # prior_mix = max(0.0, 0.6 * (1.0 - t))
                    # prior_mix = 0.0 if iteration >= 1000 else 0.6 * (1.0 - t)
                    # gaussians.reset_bcoeffs(
                    #     reset_rho=None,     # let it pick q-clip
                    #     min_frac=1e-3,
                    #     use_prior=True,
                    #     prior_mix=prior_mix,
                    #     q_clip=0.98,
                    # )

            # Optimizer step
            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none = True)
                # TEMP: removed segmentation
                # cls_optimizer.step()
                # cls_optimizer.zero_grad()
                gaussians.update_learning_rate(iteration)
                # energy_opt.step()
                # energy_opt.zero_grad(set_to_none=True)
                # if basis_opt:
                #     basis_opt.step()
                #     basis_opt.zero_grad(set_to_none=True)

            # if n_spectra > 0:
            #     if iteration > 1000 and not combine_spectra:
            #         spectral_ids.insert(0, 0)
            #         combine_spectra = True
            #         gaussians.average_params_full_spectra()
            #         gaussians.activate_spectra(spectral_id)
            #         spectral_ids = [0]
            #         spectral_id = 0

           
            
            torch.cuda.empty_cache()

   
def prepare_output_and_logger(args):    

    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])
        
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))


def training_report(iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene : Scene, renderFunc, renderArgs, loss_obj_3d, use_wandb, spectral_ids):

    if use_wandb:
        if loss_obj_3d:
            wandb.log({"train_loss_patches/l1_loss": Ll1, "train_loss_patches/total_loss": loss, "train_loss_patches/loss_obj_3d": loss_obj_3d.item(), "iter_time": elapsed, "iter": iteration})
        else:
            wandb.log({"train_loss_patches/l1_loss": Ll1, "train_loss_patches/total_loss": loss, "iter_time": elapsed, "iter": iteration})
    
    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()}, 
                              {'name': 'train', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                for spectral_id in spectral_ids:
                    for idx, viewpoint in enumerate(config['cameras']):
                        image = torch.clamp(renderFunc(viewpoint, scene.gaussians, *renderArgs, spectral_id=spectral_id)["render_" + str(spectral_id)], 0.0, 1.0)
                        gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                        if use_wandb:
                            if idx < 5:
                                wandb.log({config['name'] + "_view_{}/render".format(viewpoint.image_name): [wandb.Image(image)]})
                                if iteration == testing_iterations[0]:
                                    wandb.log({config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name): [wandb.Image(gt_image)]})
                        l1_test += l1_loss(image, gt_image).mean().double()
                        psnr_test += psnr(image, gt_image).mean().double()
                    psnr_test /= len(config['cameras'])
                    l1_test /= len(config['cameras'])  
                psnr_test /= len(spectral_ids)
                l1_test /= len(spectral_ids)            
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
                if use_wandb:
                    wandb.log({config['name'] + "/loss_viewpoint - l1_loss": l1_test, config['name'] + "/loss_viewpoint - psnr": psnr_test})
        if use_wandb:
            wandb.log({"scene/opacity_histogram": scene.gaussians.get_opacity, "total_points": scene.gaussians.get_xyz.shape[0], "iter": iteration})
        torch.cuda.empty_cache()

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)

    parser.add_argument('--n_spectra', type=int, default=2)
    parser.add_argument('--n_bcoeffs', type=int, default=2)

    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[1_000, 7_000, 30_000, 30_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[1_000, 2_000, 3_000, 5_000,7_000, 10_000,15_000,20_000,25_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    # Add an argument for the configuration file
    parser.add_argument("--config_file", type=str, default="config.json", help="Path to the configuration file")
    parser.add_argument("--use_wandb", action='store_true', default=False, help="Use wandb to record loss value")
    parser.add_argument("--use_warm_up_iteration", action='store_true', default=False, help="Use spectral info from other spectra")

    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)

    # Read and parse the configuration file
    try:
        with open(args.config_file, 'r') as file:
            config = json.load(file)
    except FileNotFoundError:
        print(f"Error: Configuration file '{args.config_file}' not found.")
        exit(1)
    except json.JSONDecodeError as e:
        print(f"Error: Failed to parse the JSON configuration file: {e}")
        exit(1)

    args.densify_until_iter = config.get("densify_until_iter", 15000)
    args.num_classes = config.get("num_classes", 200)
    args.reg3d_interval = config.get("reg3d_interval", 2)
    args.reg3d_k = config.get("reg3d_k", 5)
    args.reg3d_lambda_val = config.get("reg3d_lambda_val", 2)
    args.reg3d_max_points = config.get("reg3d_max_points", 300000)
    args.reg3d_sample_size = config.get("reg3d_sample_size", 1000)
    
    print("Optimizing " + args.model_path)

    if args.use_wandb:
        wandb.init(project="spectral-gaussian-splatting")
        wandb.config.args = args
        wandb.run.name = args.model_path

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    # network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(
        lp.extract(args), 
        op.extract(args), 
        pp.extract(args), 
        args.n_spectra, 
        args.n_bcoeffs, 
        args.test_iterations, 
        args.save_iterations, 
        args.checkpoint_iterations, 
        args.start_checkpoint, 
        args.debug_from, 
        args.use_wandb, 
        args.use_warm_up_iteration
    )

    # All done
    print("\nTraining complete.")
