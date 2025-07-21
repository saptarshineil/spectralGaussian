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
import torch
from utils.system_utils import searchForMaxIteration, searchForMaxSpectra
from scene.dataset_readers import sceneLoadTypeCallbacks
from scene.gaussian_model import GaussianModel
from arguments import ModelParams
from utils.camera_utils import cameraList_from_camInfos, camera_to_JSON
from utils.system_utils import mkdir_p
from scene.NVDIFFREC import save_env_map, load_env

class Scene:

    gaussians : GaussianModel

    def __init__(
        self, 
        args : ModelParams,
        n_spectra : int,
        n_bcoeffs : int, 
        load_iteration=None, 
        shuffle=True, 
        resolution_scales=[1.0]
    ):
        # Initialize parameters
        self.model_path = args.model_path
        self.loaded_iter = None

        if load_iteration:
            if load_iteration == -1:
                self.loaded_iter = searchForMaxIteration(os.path.join(self.model_path, "point_cloud"))
            else:
                self.loaded_iter = load_iteration
            print("Loading trained model at iteration {}".format(self.loaded_iter))

        self.train_cameras = {}
        self.test_cameras = {}

        # Read scene info
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
                n_spectra,
                n_bcoeffs,
                random_init = args.random_init,
                uniform_init = args.uniform_init,
                train_split = args.train_split,
                white_background = args.white_background,
                rgba=args.transparentbg
            )
        else:
            assert False, "Could not recognize scene type!"
        self.scene_info = scene_info

        if shuffle:
            random.shuffle(scene_info.train_cameras)  # Multi-res consistent random shuffling
            random.shuffle(scene_info.test_cameras)  # Multi-res consistent random shuffling

        self.cameras_extent = scene_info.nerf_normalization["radius"]

        # Load cameras
        for resolution_scale in resolution_scales:
            print("Loading Training Cameras")
            self.train_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.train_cameras, resolution_scale, args)
            print("Loading Test Cameras")
            self.test_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.test_cameras, resolution_scale, args)

        # Set up CT parameters
        self.vol_gt = scene_info.vol
        self.scanner_cfg = scene_info.scanner_cfg
        self.scene_scale = scene_info.scene_scale
        self.bbox = torch.stack(
            [
                torch.tensor(self.scanner_cfg["offOrigin"])
                - torch.tensor(self.scanner_cfg["sVoxel"]) / 2,
                torch.tensor(self.scanner_cfg["offOrigin"])
                + torch.tensor(self.scanner_cfg["sVoxel"]) / 2,
            ],
            dim=0,
        )

    def init_gaussians(self, gaussians : GaussianModel):
        self.gaussians = gaussians

        # No pre-trained model --> read e.g. sparse/0/points3D.ply file
        if not self.loaded_iter:
            with open(self.scene_info.ply_path, 'rb') as src_file, open(os.path.join(self.model_path, "input.ply") , 'wb') as dest_file:
                dest_file.write(src_file.read())
            json_cams = []
            camlist = []
            if self.scene_info.test_cameras:
                camlist.extend(self.scene_info.test_cameras)
            if self.scene_info.train_cameras:
                camlist.extend(self.scene_info.train_cameras)
            for id, cam in enumerate(camlist):
                json_cams.append(camera_to_JSON(id, cam))
            with open(os.path.join(self.model_path, "cameras.json"), 'w') as file:
                json.dump(json_cams, file)

        # Load pre-trained model
        if self.loaded_iter:
            if isinstance(self.loaded_iter,str):
                print("edit load path", self.loaded_iter)
                self.gaussians.load_ply(os.path.join(self.model_path,
                                                            "point_cloud"+self.loaded_iter,
                                                            "point_cloud.ply"))
            else:
                # for spectral_id in list(range(0,self.gaussians.num_spectral_bands)):
                self.gaussians.load_ply(os.path.join(self.model_path,
                                                            "point_cloud",
                                                            "iteration_" + str(self.loaded_iter) + "_" + str(0),
                                                            "point_cloud.ply"))

        else:
            # self.gaussians.create_from_pcd(self.scene_info.point_cloud, self.cameras_extent)
            self.gaussians.create_from_pcd(self.scene_info.point_cloud, 1.0)

    def save(self, iteration):
        for spectral_id in self.gaussians.spectral_ids:
            point_cloud_path = os.path.join(self.model_path, "point_cloud/iteration_{}_{}".format(iteration, spectral_id))
            self.gaussians.save_ply(os.path.join(point_cloud_path, "point_cloud.ply"), spectral_id=spectral_id)

    def getTrainCameras(self, scale=1.0):
        return self.train_cameras[scale]

    def getTestCameras(self, scale=1.0):
        return self.test_cameras[scale]