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
from gaussian_renderer import render_r2, render_mu, network_gui
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
    
    # dark      = tiff.imread('./testing/spectrum/good_metadata/di0000.tif').astype(np.float32)
    # flat_pre  = tiff.imread('./testing/spectrum/good_metadata/io0000.tif').astype(np.float32)
    # flat_post = tiff.imread('./testing/spectrum/good_metadata/io0001.tif').astype(np.float32)
    # flat      = ((flat_pre - dark) + (flat_post - dark)) / 2.0
    # I0_high   = flat.mean()                                    # photons / pixel
    # print("I0_high:", I0_high)

    # dark      = tiff.imread('./testing/spectrum/noisy_metadata/di0000.tif').astype(np.float32)
    # flat_pre  = tiff.imread('./testing/spectrum/noisy_metadata/io0000.tif').astype(np.float32)
    # flat_post = tiff.imread('./testing/spectrum/noisy_metadata/io0001.tif').astype(np.float32)
    # flat      = ((flat_pre - dark) + (flat_post - dark)) / 2.0
    # I0_low    = flat.mean()                                    # photons / pixel
    # print("I0_low:", I0_low)
    
    # TODO: Read from dateset (or preprocessing)
    # E = torch.tensor([50.0, 70.0], dtype=torch.float32).cuda()
    E = torch.tensor([50.0, 120.0], device='cuda', dtype=torch.float32)

    I0_low = 1136.8784583333334
    I0_high = 2026.8482152777776

    N_samples = 64
    # If dataset is EID (no bins) --> sample full spectrum
    E_grid, w_k = energy_quadrature(20.0, 140.0, N_samples)
    E_grid_np = E_grid.cpu().numpy()

    S_low  = get_scaled_spectrum( 50, None, 0.0, I0_low,  E_grid_np)
    S_high = get_scaled_spectrum(120, "Cu", 0.5, I0_high, E_grid_np)

    S         = torch.stack([S_low, S_high])      # (C=2, K)
    I0_scalar = torch.sum(S * w_k, dim=1)

    # ------------------------------------------
    energy_logits = torch.nn.Parameter(
        torch.tensor([0.5, -0.5], device='cuda', dtype=torch.float32))
    # energy_logits = torch.nn.Parameter(
    #     torch.tensor([0.0], device='cuda', dtype=torch.float32))

    def get_energy_weights():
        w = torch.softmax(energy_logits, dim=0) * n_spectra
        return w

    energy_opt = torch.optim.Adam([energy_logits], lr=1e-3)

    # ------------------------------------------
    bfunc_Kedge = BFuncKedge(E_grid).cuda()
    basis_opt = torch.optim.Adam(bfunc_Kedge.parameters(), lr=1e-4)

    # ------------------------------------------
    # Set up Gaussians
    gaussians = GaussianModel(
        n_spectra,
        n_bcoeffs,
        scale_bound
    )
    scene.init_gaussians(gaussians)
    gaussians.training_setup(opt)

    # Set up 3D TV loss
    use_tv = opt.lambda_tv > 0
    # use_tv = False
    if use_tv:
        print("Use total variation loss")
        tv_vol_size = opt.tv_vol_size
        tv_vol_nVoxel = torch.tensor([tv_vol_size, tv_vol_size, tv_vol_size])
        tv_vol_sVoxel = torch.tensor(scanner_cfg["dVoxel"]) * tv_vol_nVoxel
    
    # TEMP: removed segmentation
    # Set up segmentation
    # num_classes = dataset.num_classes
    # print("Num classes: ",num_classes)
    # classifier = torch.nn.Conv2d(gaussians.num_objects, num_classes, kernel_size=1)
    # cls_criterion = torch.nn.CrossEntropyLoss(reduction='none')
    # cls_optimizer = torch.optim.Adam(classifier.parameters(), lr=5e-4)
    # classifier.cuda()
    
    # if checkpoint:
        # (model_params, first_iter) = torch.load(checkpoint)
        # gaussians.restore(model_params, opt)

    # Set up training loop
    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    viewpoint_stack = None
    ema_loss_for_log = 0.0
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1
    if use_warm_up_iteration:
        combine_spectra = False
        spectral_ids = list(range(1, n_spectra)) if n_spectra > 0 else [0]
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
        
        # Track losses
        losses_per_energy = []
        # total_loss = 0
        # total_l1_loss = 0

        render_pkg = render_mu(viewpoint_cam, gaussians, pipe)
        bcoeff_image, viewspace_point_tensor, visibility_filter, radii = (
            render_pkg["bcoeffs"], 
            render_pkg["viewspace_points"], 
            render_pkg["visibility_filter"], 
            render_pkg["radii"], 
        )

        B_E = get_B_E(E_grid, bfunc_Kedge)
        bfuncs = get_B_E_at_energy(E_grid, bfunc_Kedge, E)

        loss_tv_basis = joint_tv(bcoeff_image, reduction='mean')
        # loss_tv_basis = tnv_tv(bcoeff_image, reduction='mean')
        # loss_tv_basis = linf_tv(bcoeff_image, reduction='mean')
        bcoeff_loss = opt.lambda_tv_basis * loss_tv_basis

        # loss_dirich = dirichlet_kl(gaussians.get_fracs) * opt.lambda_dirichlet
        # bcoeff_loss += loss_dirich

        # _B  = torch.nn.functional.normalize(B_E, dim=0)   # (K, 3)
        # loss_ortho = basis_orthogonality(_B) * opt.lambda_ortho
        # bcoeff_loss += loss_ortho
        

        # Compute loss per spectral channel
        for spectral_id in spectral_ids:
            # Update active spectrum
            # gaussians.activate_spectra(spectral_id)

            # ---------- Spectral Beer-Lambert ----------
            μ_E = (bcoeff_image.unsqueeze(0) * B_E[..., None, None]).sum(1)

            I_pred = torch.sum(S[spectral_id,:,None,None] *
                            torch.exp(-μ_E) * w_k[:,None,None],
                            dim=0)
            I_pred = torch.clamp(I_pred, min=1e-6 * I0_scalar[spectral_id])
            image = -torch.log(I_pred / I0_scalar[spectral_id]).unsqueeze(0)
            # image = I_pred.unsqueeze(0)

            # ---------- Loss ----------
            # L1 Loss
            # gt_image = viewpoint_cam.spectral_images[spectral_id - 1].cuda() if spectral_id > 0 else viewpoint_cam.original_image
            gt_image = viewpoint_cam.spectral_images[spectral_id].cuda()

            if (iteration - 1) % 500 == 0:
                print(image[0].mean().item() , gt_image[0].mean().item())
                print(image[0].min().item() , gt_image[0].min().item())
                print(image[0].max().item() , gt_image[0].max().item())
                print("---------------")

            # fig, axes = plt.subplots(1, 2, figsize=(6, 3), constrained_layout=True)

            # axes[0].imshow(image[0].detach().cpu().numpy(), cmap='gray', vmin=image.min(), vmax=image.max())
            # axes[1].imshow(gt_image[0].detach().cpu().numpy(), cmap='gray', vmin=gt_image.min(), vmax=gt_image.max())

            # plt.show()

            loss_l1 = l1_loss(image, gt_image)
            loss = (1.0 - opt.lambda_dssim) * loss_l1

            # DSSIM Loss
            loss_dssim = (1.0 - ssim(image, gt_image))
            loss += opt.lambda_dssim * loss_dssim

            # TEMP: removed segmentation
            # 2D Object Loss (here objects can be made per spectra) - used for segmentation / grouping
            # gt_obj = viewpoint_cam.objects.cuda().long()
            # logits = classifier(objects)
            # loss_obj = cls_criterion(logits.unsqueeze(0), gt_obj.unsqueeze(0)).squeeze().mean()
            # loss_obj = loss_obj / torch.log(torch.tensor(num_classes))  # normalize to (0,1)

            # TEMP: removed segmentation
            # 3D Object Loss (here objects can be made per spectra) - used for segmentation / grouping
            # loss_obj_3d = None
            # if iteration % opt.reg3d_interval == 0:
                # regularize at certain intervals
                # logits3d = classifier(gaussians._objects_dc[spectral_id].permute(2,0,1))
                # prob_obj3d = torch.softmax(logits3d,dim=0).squeeze().permute(1,0)
                # loss_obj_3d = loss_cls_3d(gaussians._xyz.squeeze().detach(), prob_obj3d, opt.reg3d_k, opt.reg3d_lambda_val, opt.reg3d_max_points, opt.reg3d_sample_size)
                # loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image)) + loss_obj + loss_obj_3d
            # else:
                # loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image)) + loss_obj

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
                    bfuncs[spectral_id],
                    spectral_id,
                )["ct"]
                loss_tv = tv_3d_loss(vol_pred, reduction="mean")
                loss += opt.lambda_tv * loss_tv

            

            # Loss backward
            # for k in losses_extra.keys():
            #     key = k[:k.rfind('_')]
            #     loss += getattr(opt, f'lambda_{key}')* losses_extra[k]
            # total_loss += loss.item()
            # total_loss += loss
            losses_per_energy.append(loss)
        
        energy_loss_weights = get_energy_weights()
        data_loss = (energy_loss_weights[:len(losses_per_energy)] * torch.stack(losses_per_energy)).sum()

        total_loss = data_loss + bcoeff_loss
        total_loss.backward()

        iter_end.record()

        if (iteration - 1) % 500 == 0:
            # bcoeffs = gaussians._rhos
            # print("Raw: ", bcoeffs[0].mean().item(), bcoeffs[1].mean().item())
            # print("Raw: ", bcoeffs.mean().item())
            bcoeffs_act = gaussians.get_rhos
            # print("Activated: ", bcoeffs_act[:,0].mean().item(), bcoeffs_act[:,1].mean().item())
            print("Activated density: ", bcoeffs_act.mean().item())
            # print("Grad: ", bcoeffs[0].grad.mean().item(), bcoeffs[1].grad.mean().item())
            # print("Grad: ", bcoeffs.grad.mean().item())

            # fracs = gaussians._fracs
            # print("Raw: ", fracs[0].mean().item(), fracs[1].mean().item())
            fracs_act = gaussians.get_fracs
            print("Activated fractions: ", fracs_act[:,0].mean().item(), fracs_act[:,1].mean().item(), fracs_act[:,2].mean().item())
            # print("Grad: ", fracs[0].grad.mean().item(), fracs[1].grad.mean().item())
            print("NUM GAUSSIANS: ", gaussians.get_bcoeffs.shape[0])
            print("learned w:", get_energy_weights().data.cpu().numpy())
            print("-------------------------------------")
        
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
                    gaussians.densify_and_prune(
                        opt.densify_grad_threshold,
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

            # Optimizer step
            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none = True)
                # TEMP: removed segmentation
                # cls_optimizer.step()
                # cls_optimizer.zero_grad()
                gaussians.update_learning_rate(iteration)
                energy_opt.step()
                energy_opt.zero_grad(set_to_none=True)
                basis_opt.step()
                basis_opt.zero_grad(set_to_none=True)

            if n_spectra > 0:
                if iteration > 1000 and not combine_spectra:
                    spectral_ids.insert(0, 0)
                    combine_spectra = True
                    gaussians.average_params_full_spectra()
                    gaussians.activate_spectra(spectral_id)
                    spectral_ids = [0]
                    spectral_id = 0

            # if (iteration in checkpoint_iterations):
            #     print("\n[ITER {}] Saving Checkpoint".format(iteration))
            #     torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")
            
            torch.cuda.empty_cache()

    torch.save(
        {
            "state_dict": bfunc_Kedge.state_dict(),
            "E_grid"    : E_grid.cpu(),              # needed to reconstruct
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
