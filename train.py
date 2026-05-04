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
from gaussian_renderer import render_r2, render_bcoeffs, render_fracs, network_gui
from utils.spectralCT_utils import BFuncKedge, render_image, get_B_E, get_B_E_at_energy, energy_quadrature, get_scaled_spectrum, renormalize_col_at_pivot, orthogonalize_k_to_pe_compton, bfunc_PE, bfunc_Compton, interp1d, get_spectral_constants
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state
import uuid
from tqdm import tqdm
from utils.image_utils import psnr, apply_colormap_viridis_train
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

    # ------------------------------------------
    E, E_grid, w_k, S, W, I0_scalar = get_spectral_constants(spectral_cfg=scene.spectral_cfg)

    bfunc_Kedge = None
    basis_opt = None
    
    if n_bcoeffs == 3:
        bfunc_Kedge = BFuncKedge(E_grid).cuda()

        with torch.no_grad():
            # bfunc_Kedge.set_gamma(2.5)
            # bfunc_Kedge.set_EK_init(55.0)

            # γ = 0.5 → transition width ≈ 8.8 keV (4–5 quadrature bins)
            # vs γ = 2.5 → ≈ 1.76 keV (1 bin), giving 200× smaller gradient
            bfunc_Kedge.set_gamma(0.5)
            # EK = 33.2 keV (iodine-like): visible in all 3 spectral channels (50/80/120 kVp).
            # Optimizer will adjust EK to the true material edge once unfrozen at kedge_start_iter.
            # bfunc_Kedge.set_EK_init(33.2)
            bfunc_Kedge.set_EK_init(33.2)

        # Freeze K-edge shape params until kedge_start_iter so the noisy
        # early-training signal cannot corrupt EK and gamma before geometry converges.
        for p in bfunc_Kedge.parameters():
            p.requires_grad_(False)
        print(f"[k-edge] Frozen until iteration {opt.kedge_start_iter}")

    # Set up 3D TV loss
    use_tv = opt.lambda_tv > 0
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

        losses_per_energy = []

        render_pkg = render_bcoeffs(viewpoint_cam, gaussians, pipe)
        bcoeff_image, viewspace_point_tensor, visibility_filter, radii = (
            render_pkg["bcoeffs"], 
            render_pkg["viewspace_points"], 
            render_pkg["visibility_filter"], 
            render_pkg["radii"], 
        )

        # Hoist B_E outside the spectral loop: the Cholesky orthogonalization is
        # identical for every spectral channel — compute it once per iteration.
        if n_bcoeffs == 3:
            _B_raw = get_B_E(E_grid, bfunc_Kedge, normalize=False)
            _B_raw = orthogonalize_k_to_pe_compton(_B_raw, W)
            B_E_cur, _ = renormalize_col_at_pivot(E_grid, _B_raw, E_ref=70.0)
        else:
            B_E_cur = get_B_E(E_grid, None, normalize=True)

        # Accumulators for per-500-iter loss logging
        _log_l1, _log_dssim, _log_tv = 0.0, 0.0, 0.0

        # Compute loss per spectral channel
        for spectral_id in spectral_ids:

            # ---------- Spectral Beer-Lambert (B_E pre-computed above) ----------
            image, bfuncs = render_image(bcoeff_image, bfunc_Kedge, E, E_grid, w_k, S, W, I0_scalar, n_bcoeffs, spectral_id, B_E=B_E_cur)

            # ---------- Loss ----------
            gt_image = viewpoint_cam.spectral_images[spectral_id].cuda()

            loss_l1 = l1_loss(image, gt_image)
            loss = (1.0 - opt.lambda_dssim) * loss_l1

            loss_dssim = (1.0 - ssim(image, gt_image))
            loss += opt.lambda_dssim * loss_dssim

            _log_l1    += loss_l1.item()
            _log_dssim += loss_dssim.item()

            # 3D TV loss — gated every 5 iterations to reduce voxelization cost
            if use_tv:
            # if use_tv and iteration % 5 == 0:
            # if use_tv and spectral_id == 0:
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
                loss += (opt.lambda_tv / len(spectral_ids)) * loss_tv
                _log_tv += loss_tv.item()

            losses_per_energy.append(loss)

        total_loss = torch.stack(losses_per_energy).sum()
        
        tv_pe   = joint_tv(bcoeff_image[0:1], reduction='mean')
        tv_co   = joint_tv(bcoeff_image[1:2], reduction='mean')
        bcoeff_loss  = opt.lambda_tv_basis_pe * tv_pe + opt.lambda_tv_basis_comp * tv_co
        if n_bcoeffs == 3:
            tv_K = joint_tv(bcoeff_image[2:3], reduction='mean')
            bcoeff_loss  += opt.lambda_tv_basis_K * tv_K
        total_loss += bcoeff_loss

        

        # p = gaussians.get_fracs.clamp_min(1e-6)
        # entropy = -(p * p.log()).sum(dim=1).mean()
        # total_loss += opt.lambda_entropy * entropy

        # if n_bcoeffs == 3:
        #     # Entropy only on PE/Compton sub-distribution; separate L1 sparsity on K-edge.
        #     # This prevents the entropy term from pushing f_ke away from zero.
        #     p_pc = p[:, :2] / p[:, :2].sum(dim=1, keepdim=True).clamp_min(1e-6)
        #     entropy = -(p_pc * p_pc.log()).sum(dim=1).mean()
        #     # Density-gated sparsity: only penalize background Gaussians (rho < median).
        #     # High-density (metal/K-edge) Gaussians are exempt so sparsity does not
        #     # compete with the photometric signal pushing f_ke up on metal.
        #     rho_vals = gaussians.get_rhos[:, 0].detach()
        #     bg_mask  = (rho_vals < rho_vals.median()).float()
        #     kedge_sparsity = (p[:, 2] * bg_mask).sum() / (bg_mask.sum() + 1e-8)
        #     total_loss += opt.lambda_entropy * entropy
        #     total_loss += opt.lambda_kedge_sparsity * kedge_sparsity
        # else:
        #     entropy = -(p * p.log()).sum(dim=1).mean()
        #     total_loss += opt.lambda_entropy * entropy

        total_loss.backward()
       
        iter_end.record()

        if (iteration - 1) % 500 == 0:
            # ---- Gradient stats ----
            _print_grad_stats("rhos",    gaussians._rhos.grad)
            for j in gaussians.bcoeff_ids:
                _print_grad_stats(f"fracs_{j}", gaussians._fracs[j].grad)

            # ---- Fraction statistics ----
            f = gaussians.get_fracs
            f0, f1 = f[:, 0], f[:, 1]
            if n_bcoeffs == 2:
                print(f"fracs PE  : mean={f0.mean():.4f}  min={f0.min():.4f}  max={f0.max():.4f}")
                print(f"fracs Comp: mean={f1.mean():.4f}  min={f1.min():.4f}  max={f1.max():.4f}")
            else:
                f2 = f[:, 2]
                print(f"fracs PE  : mean={f0.mean():.4f}  min={f0.min():.4f}  max={f0.max():.4f}")
                print(f"fracs Comp: mean={f1.mean():.4f}  min={f1.min():.4f}  max={f1.max():.4f}")
                print(f"fracs K   : mean={f2.mean():.4f}  min={f2.min():.4f}  max={f2.max():.4f}")

            # ---- Loss components ----
            print(f"Loss components [iter {iteration}]:")
            print(f"  L1={_log_l1:.4e}  DSSIM={_log_dssim:.4e}  TV3D={_log_tv:.4e}")
            print(f"  tv_PE={tv_pe.item():.4e}  tv_Comp={tv_co.item():.4e}", end="")
            if n_bcoeffs == 3:
                print(f"  tv_K={tv_K.item():.4e}", end="")
            print()
            # if n_bcoeffs == 3:
            #     # print(f"  entropy(PE+Co)={entropy.item():.4e}  ke_sparsity={kedge_sparsity.item():.4e}")
            #     print(f"  entropy(PE+Co)={entropy.item():.4e}")
            # else:
            #     print(f"  entropy={entropy.item():.4e}")
            print(f"  total={total_loss.item():.4e}")

            # ---- BFuncKedge parameters ----
            if bfunc_Kedge is not None and basis_opt is not None:
                with torch.no_grad():
                    EK_val    = bfunc_Kedge.EK().item()
                    alpha_val = bfunc_Kedge.log_alpha.exp().item()
                    gamma_val = bfunc_Kedge.log_gamma.exp().item()
                print(f"[k-edge] EK={EK_val:.2f} keV  alpha={alpha_val:.5f}  gamma={gamma_val:.4f}")
            elif bfunc_Kedge is not None:
                print(f"[k-edge] FROZEN (unfreezes at iter {opt.kedge_start_iter})")

            # ---- Cloud + timing ----
            print(f"rho_mean={gaussians.get_rhos.mean().item():.5f}  N_gauss={len(gaussians.get_xyz)}")
            torch.cuda.synchronize()
            iter_ms = iter_start.elapsed_time(iter_end)
            print(f"iter_time={iter_ms:.1f} ms")
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
                    gaussians.densify_and_prune(
                        opt.densify_grad_threshold,
                        opt.densify_split_grad_threshold,
                        opt.density_min_threshold,
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
                gaussians.reset_bcoeffs()

            # K-edge: unfreeze parameters and create optimizer at kedge_start_iter
            if (bfunc_Kedge is not None
                    and basis_opt is None
                    and iteration >= opt.kedge_start_iter):
                for p in bfunc_Kedge.parameters():
                    p.grad = None
                    p.requires_grad_(True)
                basis_opt = torch.optim.Adam(bfunc_Kedge.parameters(), lr=0.001)
                print(f"[k-edge] Unfrozen & optimizer created at iter {iteration}")

            # Optimizer step
            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none = True)
                gaussians.update_learning_rate(iteration)
                if basis_opt is not None:
                    basis_opt.step()
                    basis_opt.zero_grad(set_to_none=True)
        
            torch.cuda.empty_cache()

    if n_bcoeffs == 3:
        torch.save(
            {
                "state_dict": bfunc_Kedge.state_dict(),
                "E_grid"    : E_grid.cpu(),
                "w_k"       : w_k.cpu(),
            },
            os.path.join(scene.model_path, f"kedge_{iteration:06d}.pt")
        )

   
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
    parser.add_argument("--seed", type=int, default=0, help="Random seed for reproducibility")
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
    # safe_state(args.quiet, seed=args.seed)

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
