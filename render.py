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
from gaussian_renderer import render, render_lighting
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


def render_set(model_path, name, iteration, views, gaussians, pipeline, background, classifier, n_spectra):
    # TEMP: removed segmentation
    # colormask_path = os.path.join(model_path, name, "ours_{}".format(iteration), "objects_feature16")
    # gt_colormask_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt_objects_color")
    # pred_obj_path = os.path.join(model_path, name, "ours_{}".format(iteration), "objects_pred")
    # makedirs(colormask_path, exist_ok=True)
    # makedirs(gt_colormask_path, exist_ok=True)
    # makedirs(pred_obj_path, exist_ok=True)

    #need to later make this per spectra
    # TEMP: removed segmentation
    # for idx, view in enumerate(tqdm(views, desc="segmentation rendering progress")):
    #         torch.cuda.synchronize()
    #         results = render(view, gaussians, pipeline, background, debug=True, spectral_id=0)
    #         torch.cuda.synchronize()
    #         rendering_obj = results["render_object"]
            
    #         logits = classifier(rendering_obj)
    #         pred_obj = torch.argmax(logits,dim=0)
    #         pred_obj_mask = visualize_obj(pred_obj.cpu().numpy().astype(np.uint8))
            

    #         gt_objects = view.objects
    #         gt_rgb_mask = visualize_obj(gt_objects.cpu().numpy().astype(np.uint8))

    #         rgb_mask = feature_to_rgb(rendering_obj)
    #         Image.fromarray(rgb_mask).save(os.path.join(colormask_path, '{0:05d}'.format(idx) + ".png"))
    #         Image.fromarray(gt_rgb_mask).save(os.path.join(gt_colormask_path, '{0:05d}'.format(idx) + ".png"))
    #         Image.fromarray(pred_obj_mask).save(os.path.join(pred_obj_path, '{0:05d}'.format(idx) + ".png"))

    for spectral_id in list(range(0, n_spectra + 1)):
        gaussians.active_spectra = spectral_id
        render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders_" + str(spectral_id))
        gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt_" + str(spectral_id))

        diffuse_path = os.path.join(model_path, name, "ours_{}".format(iteration), "diffuse_" + str(spectral_id))
        diffuse_color_path = os.path.join(model_path, name, "ours_{}".format(iteration), "diffuse_color_" + str(spectral_id))
        specular_path = os.path.join(model_path, name, "ours_{}".format(iteration), "specular_" + str(spectral_id))
        specular_color_path = os.path.join(model_path, name, "ours_{}".format(iteration), "specular_color_" + str(spectral_id))
        roughness_path = os.path.join(model_path, name, "ours_{}".format(iteration), "roughness_" + str(spectral_id))
        # roughness_color_path = os.path.join(model_path, name, "ours_{}".format(iteration), "roughness_color")
        # color_delta = os.path.join(model_path, name, "ours_{}".format(iteration), "color_delta")
        normal = os.path.join(model_path, name, "ours_{}".format(iteration), "normal_" + str(spectral_id))
        normal_axis = os.path.join(model_path, name, "ours_{}".format(iteration), "normal_axis_" + str(spectral_id))
        delta_normal_norm = os.path.join(model_path, name, "ours_{}".format(iteration), "delta_normal_norm_" + str(spectral_id))
        normal_cam = os.path.join(model_path, name, "ours_{}".format(iteration), "normal_cam_" + str(spectral_id))
        normal_ref = os.path.join(model_path, name, "ours_{}".format(iteration), "normal_ref_" + str(spectral_id))
        depth = os.path.join(model_path, name, "ours_{}".format(iteration), "depth_" + str(spectral_id))
        alpha = os.path.join(model_path, name, "ours_{}".format(iteration), "alpha_" + str(spectral_id))

        makedirs(render_path, exist_ok=True)
        makedirs(gts_path, exist_ok=True)

        makedirs(diffuse_path, exist_ok=True)
        makedirs(diffuse_color_path, exist_ok=True)
        makedirs(specular_path, exist_ok=True)
        makedirs(specular_color_path, exist_ok=True)
        makedirs(roughness_path, exist_ok=True)
        # makedirs(color_delta, exist_ok=True)
        makedirs(normal, exist_ok=True)
        makedirs(normal_axis, exist_ok=True)
        makedirs(delta_normal_norm, exist_ok=True)
        makedirs(normal_cam, exist_ok=True)
        makedirs(normal_ref, exist_ok=True)
        makedirs(depth, exist_ok=True)
        makedirs(alpha, exist_ok=True)
        

        for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
            torch.cuda.synchronize()
            results = render(view, gaussians, pipeline, background, debug=True, spectral_id=spectral_id)
            torch.cuda.synchronize()
            rendering = results["render_" + str(spectral_id)]
            
            gt = view.original_image[0:3, :, :] if spectral_id == 0 else view.spectral_images[spectral_id - 1][0:3,:,:]
            torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
            torchvision.utils.save_image(gt, os.path.join(gts_path, '{0:05d}'.format(idx) + ".png"))


            torchvision.utils.save_image(results["diffuse_" + str(spectral_id)], os.path.join(diffuse_path, '{0:05d}'.format(idx) + ".png"))
            torchvision.utils.save_image(results["diffuse_color_" + str(spectral_id)], os.path.join(diffuse_color_path, '{0:05d}'.format(idx) + ".png"))
            torchvision.utils.save_image(results["specular_" + str(spectral_id)], os.path.join(specular_path, '{0:05d}'.format(idx) + ".png"))
            torchvision.utils.save_image(results["specular_color_" + str(spectral_id)], os.path.join(specular_color_path, '{0:05d}'.format(idx) + ".png"))
            torchvision.utils.save_image(results["roughness_" + str(spectral_id)], os.path.join(roughness_path, '{0:05d}'.format(idx) + ".png"))
            # torchvision.utils.save_image(results["color_delta_" + str(spectral_id)], os.path.join(color_delta, '{0:05d}'.format(idx) + ".png"))
            torchvision.utils.save_image(results["normal_" + str(spectral_id)], os.path.join(normal, '{0:05d}'.format(idx) + ".png"))
            torchvision.utils.save_image(results["normal_axis_" + str(spectral_id)], os.path.join(normal_axis, '{0:05d}'.format(idx) + ".png"))
            torchvision.utils.save_image(results["delta_normal_norm_" + str(spectral_id)], os.path.join(delta_normal_norm, '{0:05d}'.format(idx) + ".png"))
            torchvision.utils.save_image(results["normal_cam_" + str(spectral_id)], os.path.join(normal_cam, '{0:05d}'.format(idx) + ".png"))
            torchvision.utils.save_image(results["normal_ref_" + str(spectral_id)], os.path.join(normal_ref, '{0:05d}'.format(idx) + ".png"))
            torchvision.utils.save_image(results["depth_" + str(spectral_id)], os.path.join(depth, '{0:05d}'.format(idx) + ".png"))
            torchvision.utils.save_image(results["alpha_" + str(spectral_id)], os.path.join(alpha, '{0:05d}'.format(idx) + ".png"))

        gt = views[0].original_image[0:3, :, :]
        out_path = os.path.join(render_path[:-8],'concat_reflectance_'+str(spectral_id))
        makedirs(out_path,exist_ok=True)
        fourcc = cv2.VideoWriter.fourcc(*'DIVX') 
        size = (gt.shape[-1]*6,gt.shape[-2])
        fps = float(5) if 'train' in out_path else float(1)
        writer = cv2.VideoWriter(os.path.join(out_path,'result.mp4'), fourcc, fps, size)

        for file_name in sorted(os.listdir(gts_path)):
            diffuse = np.array(Image.open(os.path.join(diffuse_path,file_name)))
            diffuse_color = np.array(Image.open(os.path.join(diffuse_color_path,file_name)))
            specular = np.array(Image.open(os.path.join(specular_path,file_name)))
            specular_color = np.array(Image.open(os.path.join(specular_color_path,file_name)))
            roughness = np.array(Image.open(os.path.join(roughness_path,file_name)))
            normal_val = np.array(Image.open(os.path.join(normal,file_name)))

            result = np.hstack([diffuse, diffuse_color, specular, specular_color, roughness, normal_val])
            result = result.astype('uint8')

            Image.fromarray(result).save(os.path.join(out_path,file_name))
            writer.write(result[:,:,::-1])

        writer.release()

        #todo : fixt output_path
        out_path = os.path.join(render_path[:-8],'concat_rendering_'+str(spectral_id))
        makedirs(out_path,exist_ok=True)
        fourcc = cv2.VideoWriter.fourcc(*'DIVX') 
        size = (gt.shape[-1]*4,gt.shape[-2])
        fps = float(5) if 'train' in out_path else float(1)
        writer = cv2.VideoWriter(os.path.join(out_path,'result.mp4'), fourcc, fps, size)

        # TEMP: removed segmentation
        # for file_name in sorted(os.listdir(gts_path)):
        #     gt = np.array(Image.open(os.path.join(gts_path,file_name)))
        #     rgb = np.array(Image.open(os.path.join(render_path,file_name)))
            
        #     pred_obj = np.array(Image.open(os.path.join(pred_obj_path,file_name)))
        #     color_mask = np.array(Image.open(os.path.join(colormask_path,file_name)))
            
        #     # pred_obj = np.array(Image.open(os.path.join(roughness_path,file_name)))
        #     # pred_obj = np.array(Image.open(os.path.join(normal,file_name)))

        #     result = np.hstack([gt,rgb,pred_obj, color_mask])
        #     result = result.astype('uint8')

        #     Image.fromarray(result).save(os.path.join(out_path,file_name))
        #     writer.write(result[:,:,::-1])

        writer.release()


def render_sets(dataset : ModelParams, iteration : int, pipeline : PipelineParams, skip_train : bool, skip_test : bool, n_spectra:int):
    
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree, dataset.brdf_dim, dataset.brdf_mode, dataset.brdf_envmap_res, n_spectra)

        scene = Scene(dataset, load_iteration=iteration, shuffle=False)
        scene.init_gaussians(gaussians)

        # TEMP: removed segmentation
        # num_classes = dataset.num_classes
        # print("Num classes: ",num_classes)

        # TEMP: removed segmentation
        classifier = None
        # classifier = torch.nn.Conv2d(gaussians.num_objects, num_classes, kernel_size=1)
        # classifier.cuda()
        # classifier.load_state_dict(torch.load(os.path.join(dataset.model_path,"point_cloud","iteration_"+str(scene.loaded_iter)+"_" + str(n_spectra),"classifier.pth")))

        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        if not skip_train:
             render_set(dataset.model_path, "train", scene.loaded_iter, scene.getTrainCameras(), gaussians, pipeline, background, classifier, n_spectra)

        if (not skip_test) and (len(scene.getTestCameras()) > 0):
             render_set(dataset.model_path, "test", scene.loaded_iter, scene.getTestCameras(), gaussians, pipeline, background, classifier, n_spectra)
        
        if pipeline.brdf:
             render_lightings(dataset.model_path, "lighting", scene.loaded_iter, gaussians, 1, n_spectra)

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--n_spectra", type=int)
    args = get_combined_args(parser)
    print("Rendering " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    render_sets(model.extract(args), args.iteration, pipeline.extract(args), args.skip_train, args.skip_test, args.n_spectra)