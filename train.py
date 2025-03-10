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
from utils.loss_utils import l1_loss, ssim, loss_cls_3d, predicted_normal_loss, delta_normal_loss, zero_one_loss
from gaussian_renderer import render, network_gui
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

def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from, use_wandb, n_spectra, use_warm_up_iteration):
    first_iter = 0
    prepare_output_and_logger(dataset)
    gaussians = GaussianModel(dataset.sh_degree, dataset.brdf_dim, dataset.brdf_mode, dataset.brdf_envmap_res, n_spectra)
    scene = Scene(dataset, gaussians)
    gaussians.training_setup(opt)
    
    # TEMP: removed segmentation
    # num_classes = dataset.num_classes
    # print("Num classes: ",num_classes)
    # classifier = torch.nn.Conv2d(gaussians.num_objects, num_classes, kernel_size=1)
    # cls_criterion = torch.nn.CrossEntropyLoss(reduction='none')
    # cls_optimizer = torch.optim.Adam(classifier.parameters(), lr=5e-4)
    # classifier.cuda()
    
    # if checkpoint:
        # (model_params, first_iter) = torch.load(checkpoint)
        # gaussians.restore(model_params, opt)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    viewpoint_stack = None
    ema_loss_for_log = 0.0
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1
    if use_warm_up_iteration:
        combine_spectra = False
        spectral_ids = list(range(1, n_spectra + 1)) if n_spectra > 0 else [0]
    else:
        combine_spectra = True
        spectral_ids = list(range(0, n_spectra + 1)) if n_spectra > 0 else [0]
   
    # Start training loop
    for iteration in range(first_iter, opt.iterations + 1):   
        iter_start.record()
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))
        total_loss = 0
        total_l1_loss = 0
        
        for spectral_id in spectral_ids:

            # Network GUI     
            if network_gui.conn == None:
                network_gui.try_connect()
            while network_gui.conn != None:
                try:
                    net_image_bytes = None
                    custom_cam, do_training, pipe.convert_SHs_python, pipe.compute_cov3D_python, keep_alive, scaling_modifer = network_gui.receive()
                    if custom_cam != None:
                        net_image = render(custom_cam, gaussians, pipe, background, scaling_modifer, spectral_id=spectral_id)["render"]
                        net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())
                    network_gui.send(net_image_bytes, dataset.source_path)
                    if do_training and ((iteration < int(opt.iterations)) or not keep_alive):
                        break
                except Exception as e:
                    network_gui.conn = None

            # Every 1000 its we increase the levels of SH up to a maximum degree
            if iteration % 1000 == 0:
                gaussians.oneupSHdegree()
                # if combine_spectra:
                #     gaussians.average_params_full_spectra()

            # Update active spectrum
            gaussians.activate_spectra(spectral_id, iteration >= opt.normal_reg_from_iter)

            # ---------- Loss ----------
            # Render
            if (iteration - 1) == debug_from:
                pipe.debug = True
            render_pkg = render(viewpoint_cam, gaussians, pipe, background,spectral_id=spectral_id)
            losses_extra = {}        
            image, viewspace_point_tensor, visibility_filter, radii, objects = render_pkg["render_"+str(spectral_id)], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"], render_pkg["render_object"]
            if pipe.brdf and iteration > opt.normal_reg_from_iter:
                if iteration<opt.normal_reg_util_iter:
                    losses_extra['predicted_normal_'+str(spectral_id)] = predicted_normal_loss(render_pkg["normal_"+str(spectral_id)], render_pkg["normal_ref_"+str(spectral_id)], render_pkg["alpha_"+str(spectral_id)])
                losses_extra['zero_one_'+str(spectral_id)] = zero_one_loss(render_pkg["alpha_"+str(spectral_id)])
                if "delta_normal_norm_"+str(spectral_id) not in render_pkg.keys() and opt.lambda_delta_reg>0: assert()
                if "delta_normal_norm_"+str(spectral_id)in render_pkg.keys():
                    losses_extra['delta_reg_'+str(spectral_id)] = delta_normal_loss(render_pkg["delta_normal_norm_"+str(spectral_id)], render_pkg["alpha_"+str(spectral_id)])

            # L1 Loss
            gt_image = viewpoint_cam.spectral_images[spectral_id - 1].cuda() if spectral_id > 0 else viewpoint_cam.original_image
            Ll1 = l1_loss(image, gt_image)
            total_l1_loss = total_l1_loss + Ll1.item()
            # if iteration > 100 :
            # torchvision.utils.save_image(image, "img_"+ str(spectral_id) + ".png")

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
            
            # TEMP: removed segmentation
            loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image))

            for k in losses_extra.keys():
                key = k[:k.rfind('_')]
                loss += getattr(opt, f'lambda_{key}')* losses_extra[k]
            total_loss += loss.item()
            loss.backward()

            # torch.cuda.empty_cache()
        iter_end.record()
        with torch.no_grad():
            torch.cuda.empty_cache()
            # Progress bar
            ema_loss_for_log = 0.4 * total_loss + 0.6 * ema_loss_for_log
            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}"})
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            losses_extra['psnr'] = psnr(image, gt_image).mean()
            # Log and save
            #todo : implement evaluation. Important for Evaluation
            # training_report(iteration, total_l1_loss, total_loss, l1_loss, iter_start.elapsed_time(iter_end), testing_iterations, scene, render, (pipe, background), loss_obj_3d, use_wandb, spectral_ids)
            # todo:check the with gt objects for each spectra and save the classifier path per spectra 
            if (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)
                # TEMP: removed segmentation
                # torch.save(classifier.state_dict(), os.path.join(scene.model_path, "point_cloud/iteration_{}_{}".format(iteration, spectral_id),'classifier.pth'))

            # Densification
            if iteration < opt.densify_until_iter:
                # Keep track of max radii in image-space for pruning
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)
                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    gaussians.densify_and_prune(opt.densify_grad_threshold, 0.005, scene.cameras_extent, size_threshold)
                
                if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                    gaussians.reset_opacity()

            # Optimizer step
            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none = True)
                # cls_optimizer.step()
                # cls_optimizer.zero_grad()
                gaussians.update_learning_rate(iteration)
            
            if pipe.brdf and pipe.brdf_mode=="envmap":
                for spectral_id in list(range(0, gaussians.num_spectral_bands + 1)):
                    gaussians.brdf_mlp[spectral_id].clamp_(min=0.0, max=1.0)
            
            if n_spectra > 0:
                if iteration > 1000 and not combine_spectra:
                    spectral_ids.insert(0, 0)
                    combine_spectra = True
                    gaussians.average_params_full_spectra()
                    gaussians.activate_spectra(spectral_id, iteration >= opt.normal_reg_from_iter)
                    spectral_ids = [0]
                    spectral_id = 0

            # if (iteration in checkpoint_iterations):
            #     print("\n[ITER {}] Saving Checkpoint".format(iteration))
            #     torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")
            
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
    parser.add_argument('--n_spectra', type=int, default=0)
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
    network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from, args.use_wandb, args.n_spectra, args.use_warm_up_iteration)

    # All done
    print("\nTraining complete.")
