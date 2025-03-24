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

import os
import random
import json
from utils.system_utils import searchForMaxIteration, searchForMaxSpectra
from scene.dataset_readers import sceneLoadTypeCallbacks
from scene.gaussian_model import GaussianModel
from arguments import ModelParams
from utils.camera_utils import cameraList_from_camInfos, camera_to_JSON
from utils.system_utils import mkdir_p
from scene.NVDIFFREC import save_env_map, load_env

class Scene:

    gaussians : GaussianModel

    def __init__(self, args : ModelParams, gaussians : GaussianModel, load_iteration=None, shuffle=True, resolution_scales=[1.0]):
        """b
        :param path: Path to colmap scene main folder.
        """
        self.model_path = args.model_path
        self.loaded_iter = None
        self.gaussians = gaussians

        if load_iteration:
            if load_iteration == -1:
                self.loaded_iter = searchForMaxIteration(os.path.join(self.model_path, "point_cloud"))
            else:
                self.loaded_iter = load_iteration
            print("Loading trained model at iteration {}".format(self.loaded_iter))

        self.train_cameras = {}
        self.test_cameras = {}

        if os.path.exists(os.path.join(args.source_path, "sparse")):
            scene_info = sceneLoadTypeCallbacks["Colmap"](args.source_path, args.images, args.eval, args.object_path, n_views=args.n_views, random_init=args.random_init, train_split=args.train_split,white_background=args.white_background, rgba=args.transparentbg)
        elif os.path.exists(os.path.join(args.source_path, "transforms_train.json")):
            print("Found transforms_train.json file, assuming Blender data set!")
            scene_info = sceneLoadTypeCallbacks["Blender"](args.source_path, args.white_background, args.eval)
        # R2-Gaussian CT dataset format
        elif os.path.exists(os.path.join(args.source_path, "meta_data.json")):
            print("Found meta_data.json file, assuming Blender CT data set!")
            scene_info = sceneLoadTypeCallbacks["CT"](
                args.source_path,
                args.images,
                args.eval,
                random_init = args.random_init,
                uniform_init = args.uniform_init,
                train_split = args.train_split,
                white_background = args.white_background,
                rgba=args.transparentbg
            )
        else:
            assert False, "Could not recognize scene type!"

        # No pre-trained model --> read e.g. sparse/0/points3D.ply file
        if not self.loaded_iter:
            with open(scene_info.ply_path, 'rb') as src_file, open(os.path.join(self.model_path, "input.ply") , 'wb') as dest_file:
                dest_file.write(src_file.read())
            json_cams = []
            camlist = []
            if scene_info.test_cameras:
                camlist.extend(scene_info.test_cameras)
            if scene_info.train_cameras:
                camlist.extend(scene_info.train_cameras)
            for id, cam in enumerate(camlist):
                json_cams.append(camera_to_JSON(id, cam))
            with open(os.path.join(self.model_path, "cameras.json"), 'w') as file:
                json.dump(json_cams, file)

        if shuffle:
            random.shuffle(scene_info.train_cameras)  # Multi-res consistent random shuffling
            random.shuffle(scene_info.test_cameras)  # Multi-res consistent random shuffling

        self.cameras_extent = scene_info.nerf_normalization["radius"]
        # self.gaussians.spectral_ids = scene_info.spectral_ids

        for resolution_scale in resolution_scales:
            print("Loading Training Cameras")
            self.train_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.train_cameras, resolution_scale, args)
            print("Loading Test Cameras")
            self.test_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.test_cameras, resolution_scale, args)

        # Load pre-trained model
        if self.loaded_iter:
            if isinstance(self.loaded_iter,str):
                print("edit load path", self.loaded_iter)
                self.gaussians.load_ply(os.path.join(self.model_path,
                                                            "point_cloud"+self.loaded_iter,
                                                            "point_cloud.ply"))
            else:
                # max_spectra = searchForMaxSpectra(os.path.join(self.model_path, "point_cloud_" + self.loaded_iter))
                if self.gaussians.brdf:
                    self.gaussians.brdf_mlp = []
                    for spectral_id in list(range(0, self.gaussians.num_spectral_bands + 1)):
                        fn = os.path.join(self.model_path,
                                        "brdf_mlp",
                                        "iteration_" + str(self.loaded_iter) + "_" + str(spectral_id),
                                        "brdf_mlp.hdr")
                        self.gaussians.brdf_mlp.append(load_env(fn, scale=1.0))
                        print(f"Load envmap from: {fn}")
                for spectral_id in list(range(0,self.gaussians.num_spectral_bands + 1)):
                    self.gaussians.load_ply(os.path.join(self.model_path,
                                                            "point_cloud",
                                                            "iteration_" + str(self.loaded_iter) + "_" + str(spectral_id),
                                                            "point_cloud.ply"))

        else:
            self.gaussians.create_from_pcd(scene_info.point_cloud, self.cameras_extent)

    def save(self, iteration):
        for spectral_id in self.gaussians.spectral_ids:
            point_cloud_path = os.path.join(self.model_path, "point_cloud/iteration_{}_{}".format(iteration, spectral_id))
            self.gaussians.save_ply(os.path.join(point_cloud_path, "point_cloud.ply"), spectral_id=spectral_id)
            if self.gaussians.brdf:    
                brdf_mlp_path = os.path.join(self.model_path, f"brdf_mlp/iteration_{iteration}_{spectral_id}/brdf_mlp.hdr")
                mkdir_p(os.path.dirname(brdf_mlp_path))
                save_env_map(brdf_mlp_path, self.gaussians.brdf_mlp[spectral_id])

    def getTrainCameras(self, scale=1.0):
        return self.train_cameras[scale]

    def getTestCameras(self, scale=1.0):
        return self.test_cameras[scale]