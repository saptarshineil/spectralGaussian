# Copyright (C) 2023, Gaussian-Grouping
# Gaussian-Grouping research group, https://github.com/lkeab/gaussian-grouping
# All rights reserved.
#
# ------------------------------------------------------------------------
# Modified from codes in Gaussian-Splatting 
# GRAPHDECO research group, https://team.inria.fr/graphdeco

import os
import sys
from PIL import Image
from typing import NamedTuple
from scene.colmap_loader import read_extrinsics_text, read_intrinsics_text, qvec2rotmat, \
    read_extrinsics_binary, read_intrinsics_binary, read_points3D_binary, read_points3D_text
from utils.graphics_utils import getWorld2View2, focal2fov, fov2focal
import numpy as np
import math
import json
from pathlib import Path
from plyfile import PlyData, PlyElement
from utils.sh_utils import SH2RGB
from scene.gaussian_model import BasicPointCloud
from glob import glob
import torch

import matplotlib.pyplot as plt
from utils.spectralCT_utils import pe_compton_from_density, analytic_fraction_2, analytic_fraction_3 


mode_id = {
    "parallel": 0,
    "cone": 1,
}

# ------ Default SpectralGaussian info ------
class CameraInfo(NamedTuple):
    uid: int
    R: np.array
    T: np.array
    FovY: np.array
    FovX: np.array
    image: np.array
    image_path: str
    image_name: str
    width: int
    height: int
    objects: np.array
    spectral_images:list

class SceneInfo(NamedTuple):
    point_cloud: BasicPointCloud
    train_cameras: list
    test_cameras: list
    nerf_normalization: dict
    ply_path: str
    spectral_ids: list

# ------ Spectral CT info -------
class CameraInfoSpecCT(NamedTuple):
    uid: int
    R: np.array
    T: np.array
    angle: float                # CT
    FovY: np.array
    FovX: np.array
    image: np.array
    image_path: str
    image_name: str
    width: int
    height: int
    objects: np.array           # spectral
    spectral_images:list        # spectral
    mode: int                   # CT
    scanner_cfg: dict           # CT

class SceneInfoSpecCT(NamedTuple):
    point_cloud: BasicPointCloud
    scanner_cfg: dict               # CT
    vol: torch.tensor               # CT
    scene_scale: float              # CT
    spectral_ids: list              # spectral
    train_cameras: list             # both
    test_cameras: list              # both
    nerf_normalization: dict        # spectral
    ply_path: str                   # spectral


def getNerfppNorm(cam_info):
    def get_center_and_diag(cam_centers):
        cam_centers = np.hstack(cam_centers)
        avg_cam_center = np.mean(cam_centers, axis=1, keepdims=True)
        center = avg_cam_center
        dist = np.linalg.norm(cam_centers - center, axis=0, keepdims=True)
        diagonal = np.max(dist)
        return center.flatten(), diagonal

    cam_centers = []

    for cam in cam_info:
        W2C = getWorld2View2(cam.R, cam.T)
        C2W = np.linalg.inv(W2C)
        cam_centers.append(C2W[:3, 3:4])

    center, diagonal = get_center_and_diag(cam_centers)
    radius = diagonal * 1.1

    translate = -center

    return {"translate": translate, "radius": radius}


def fix_background_rgba(input_image, white_background=False):
     im_data = np.array(input_image.convert("RGBA"))

     bg = np.array([1,1,1]) if white_background else np.array([0, 0, 0])

     norm_data = im_data / 255.0
     arr = norm_data[:,:,:3] * norm_data[:, :, 3:4] + bg * (1 - norm_data[:, :, 3:4])
     return Image.fromarray(np.array(arr*255.0, dtype=np.byte), "RGB")



def readColmapCameras(cam_extrinsics, cam_intrinsics, images_folder, objects_folder,spectral_folder,white_background,rgba):
    cam_infos = []
    for idx, key in enumerate(cam_extrinsics):
        sys.stdout.write('\r')
        # the exact output you're looking for:
        sys.stdout.write("Reading camera {}/{}".format(idx+1, len(cam_extrinsics)))
        sys.stdout.flush()

        extr = cam_extrinsics[key]
        intr = cam_intrinsics[extr.camera_id]
        height = intr.height
        width = intr.width

        uid = intr.id
        R = np.transpose(qvec2rotmat(extr.qvec))
        T = np.array(extr.tvec)

        if intr.model=="SIMPLE_PINHOLE":
            focal_length_x = intr.params[0]
            FovY = focal2fov(focal_length_x, height)
            FovX = focal2fov(focal_length_x, width)
        elif intr.model=="PINHOLE":
            focal_length_x = intr.params[0]
            focal_length_y = intr.params[1]
            FovY = focal2fov(focal_length_y, height)
            FovX = focal2fov(focal_length_x, width)
        else:
            assert False, "Colmap camera model not handled: only undistorted datasets (PINHOLE or SIMPLE_PINHOLE cameras) supported!"

        image_path = os.path.join(images_folder, os.path.basename(extr.name))
        image_name, extension = os.path.basename(image_path).split(".")
        image = Image.open(image_path) if os.path.exists(image_path) else None
        if rgba:
            image = fix_background_rgba(image, white_background)
        object_path = os.path.join(objects_folder, image_name + '.png')
        objects = Image.open(object_path) if os.path.exists(object_path) else None
        spectral_list = sorted(glob(spectral_folder + '/*'))
        spectral_images = []
        for spectra in spectral_list:
            spectra_img = Image.open(os.path.join(spectral_folder, spectra, image_name + '.' + extension))
            if rgba:
                spectra_img = fix_background_rgba(spectra_img,white_background)
            spectral_images.append(spectra_img)

        cam_info = CameraInfo(uid=uid, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
                              image_path=image_path, image_name=image_name, width=width, height=height, objects=objects, spectral_images=spectral_images)
        cam_infos.append(cam_info)
    sys.stdout.write('\n')
    return cam_infos

def fetchPly(path):
    plydata = PlyData.read(path)
    vertices = plydata['vertex']
    positions = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T
    num_pts = vertices.count
    normals = None
    colors = None
    if 'red' in vertices:
        colors = np.vstack([vertices['red'], vertices['green'], vertices['blue']]).T / 255.0
    else:
        colors = np.zeros((num_pts, 3))
    if 'nx' in vertices:
        normals = np.vstack([vertices['nx'], vertices['ny'], vertices['nz']]).T
    else:
        normals = np.zeros((num_pts, 3))
    return BasicPointCloud(points=positions, colors=colors, normals=normals)

# def storePly(path, xyz, bcoeffs):
#     dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4')]
    
#     for i in range(bcoeffs.shape[1]):
#         dtype.append(('bcoeff' + str(i), 'f4'))
    
#     elements = np.empty(xyz.shape[0], dtype=dtype)
#     attributes = np.concatenate((xyz, bcoeffs), axis=1)
#     elements[:] = list(map(tuple, attributes))

#     # Create the PlyData object and write to file
#     vertex_element = PlyElement.describe(elements, 'vertex')
#     ply_data = PlyData([vertex_element])
#     ply_data.write(path)

def storePly(path, xyz, rhos, fracs):
    # dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4')]
    dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'), ('rhos', 'f4')]
    
    # for i in range(rhos.shape[1]):
    #     dtype.append(('rhos_' + str(i), 'f4'))
    for i in range(fracs.shape[1]):
        dtype.append(('fracs_' + str(i), 'f4'))
    
    elements = np.empty(xyz.shape[0], dtype=dtype)
    attributes = np.concatenate((xyz, rhos, fracs), axis=1)
    elements[:] = list(map(tuple, attributes))

    # Create the PlyData object and write to file
    vertex_element = PlyElement.describe(elements, 'vertex')
    ply_data = PlyData([vertex_element])
    ply_data.write(path)

def readColmapSceneInfo(path, images, eval, object_path, llffhold=8, n_views=100, random_init=False, train_split=False, spectra_path=None, white_background=False, rgba=False):
    try:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.bin")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.bin")
        cam_extrinsics = read_extrinsics_binary(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_binary(cameras_intrinsic_file)
    except:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.txt")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.txt")
        cam_extrinsics = read_extrinsics_text(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_text(cameras_intrinsic_file)

    reading_dir = "images" if images == None else images
    object_dir = 'object_mask' if object_path == None else object_path
    #add spectral info
    spectra_dir = 'spectra' if spectra_path==None else spectra_path
    spectral_folder = os.path.join(path, spectra_dir)
    num_spectra = len(glob(spectral_folder + '/*'))
    spectral_ids = list(range(1,num_spectra + 1))
    cam_infos_unsorted = readColmapCameras(cam_extrinsics=cam_extrinsics, cam_intrinsics=cam_intrinsics, images_folder=os.path.join(path, reading_dir), objects_folder=os.path.join(path, object_dir), spectral_folder=spectral_folder, white_background=white_background, rgba=rgba)
    cam_infos = sorted(cam_infos_unsorted.copy(), key = lambda x : x.image_name)

    if eval:
        if train_split:
            train_dir = os.path.join(path, "images_train")
            train_names = sorted(os.listdir(train_dir))
            train_names = [train_name.split('.')[0] for train_name in train_names]
            train_cam_infos = []
            test_cam_infos = []
            for cam_info in cam_infos:
                if cam_info.image_name in train_names:
                    train_cam_infos.append(cam_info)
                else:
                    test_cam_infos.append(cam_info)

        else:
            train_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold != 0]
            test_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold == 0]

            if n_views == 100:
                pass 
            elif n_views == 50:
                idx_sub = np.linspace(0, len(train_cam_infos)-1, round(len(train_cam_infos)*0.5)) # 50% views
                idx_sub = [round(i) for i in idx_sub]
                train_cam_infos = [train_cam_infos[i_sub] for i_sub in idx_sub]
            elif isinstance(n_views,int):
                idx_sub = np.linspace(0, len(train_cam_infos)-1, n_views) # 3views
                idx_sub = [round(i) for i in idx_sub]
                train_cam_infos = [train_cam_infos[i_sub] for i_sub in idx_sub]
                print(train_cam_infos)
            else:
                raise NotImplementedError
        print("Training images:     ", len(train_cam_infos))
        print("Testing images:     ", len(test_cam_infos))

    else:
        if train_split:
            train_dir = os.path.join(path, "images_train")
            train_names = sorted(os.listdir(train_dir))
            train_names = [train_name.split('.')[0] for train_name in train_names]
            train_cam_infos = []
            for cam_info in cam_infos:
                if cam_info.image_name in train_names:
                    train_cam_infos.append(cam_info)
            test_cam_infos = []
        else:
            train_cam_infos = cam_infos
            test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)

    if random_init:
        # Since this data set has no colmap data, we start with random points
        num_pts = 100_000
        print(f"Generating random point cloud ({num_pts})...")
        
        # We create random points inside the bounds of the synthetic Blender scenes
        xyz = np.random.random((num_pts, 3)) * 2.6 - 1.3
        shs = np.random.random((num_pts, 3)) / 255.0
        pcd = BasicPointCloud(points=xyz, colors=SH2RGB(shs), normals=np.zeros((num_pts, 3)))
        
        ply_path = os.path.join(path, "sparse/0/points3D_randinit.ply")
        storePly(ply_path, xyz, SH2RGB(shs) * 255)

    else:
        ply_path = os.path.join(path, "sparse/0/points3D.ply")
        bin_path = os.path.join(path, "sparse/0/points3D.bin")
        txt_path = os.path.join(path, "sparse/0/points3D.txt")
        if not os.path.exists(ply_path):
            print("Converting point3d.bin to .ply, will happen only the first time you open the scene.")
            try:
                xyz, rgb, _ = read_points3D_binary(bin_path)
            except:
                xyz, rgb, _ = read_points3D_text(txt_path)
            storePly(ply_path, xyz, rgb)
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path, spectral_ids=spectral_ids)
    return scene_info

def readCamerasFromTransforms(path, transformsfile, white_background, extension=".png", spectral_folder=None):
    cam_infos = []

    with open(os.path.join(path, transformsfile)) as json_file:
        contents = json.load(json_file)
        if "camera_angle_x" not in contents.keys():
            fovx = None
        else:
            fovx = contents["camera_angle_x"] 
        object_dir = os.path.join(path, 'object_mask')
        frames = contents["frames"]
        for idx, frame in enumerate(frames):
            cam_name = os.path.join(path, frame["file_path"] + extension)

            # NeRF 'transform_matrix' is a camera-to-world transform
            c2w = np.array(frame["transform_matrix"])
            # change from OpenGL/Blender camera axes (Y up, Z back) to COLMAP (Y down, Z forward)
            c2w[:3, 1:3] *= -1

            # get the world-to-camera transform and set R, T
            w2c = np.linalg.inv(c2w)
            R = np.transpose(w2c[:3,:3])  # R is stored transposed due to 'glm' in CUDA code
            T = w2c[:3, 3]

            image_path = os.path.join(path, cam_name)
            image_name = Path(cam_name).stem
            image = Image.open(image_path)

            im_data = np.array(image.convert("RGBA"))

            bg = np.array([1,1,1]) if white_background else np.array([0, 0, 0])

            norm_data = im_data / 255.0
            arr = norm_data[:,:,:3] * norm_data[:, :, 3:4] + bg * (1 - norm_data[:, :, 3:4])
            image = Image.fromarray(np.array(arr*255.0, dtype=np.byte), "RGB")
            alpha_mask = norm_data[:, :, 3]
            alpha_mask = Image.fromarray(np.array(alpha_mask*255.0, dtype=np.byte), "L")

            if fovx == None:
                focal_length = contents["fl_x"]
                FovY = focal2fov(focal_length, image.size[1])
                FovX = focal2fov(focal_length, image.size[0])
            else:
                fovy = focal2fov(fov2focal(fovx, image.size[0]), image.size[1])
                FovY = fovx 
                FovX = fovy
            object_path = os.path.join(object_dir, image_name + '.png')
            objects = Image.open(object_path) if os.path.exists(object_path) else None
            spectral_list = sorted(glob(spectral_folder + '/*'))
            spectral_images = []
            for spectra in spectral_list:
                spectra_img = Image.open(os.path.join(spectral_folder, spectra, image_name + extension))
                spectral_images.append(fix_background_rgba(spectra_img, white_background))
            cam_infos.append(CameraInfo(uid=idx, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
                            image_path=image_path, image_name=image_name, width=image.size[0], height=image.size[1],objects=objects, spectral_images=spectral_images))
            
    return cam_infos

def readNerfSyntheticInfo(path, white_background, eval, extension=".png", spectra_path=None):

        #add spectral info
    spectra_dir = 'spectra' if spectra_path==None else spectra_path
    spectral_folder = os.path.join(path, spectra_dir)
    num_spectra = len(glob(spectral_folder + '/*'))
    spectral_ids = list(range(1,num_spectra + 1))
    print("Reading Training Transforms")
    train_cam_infos = readCamerasFromTransforms(path, "transforms_train.json", white_background, extension, spectral_folder=spectral_folder)
    # print("Reading Test Transforms")
    # test_cam_infos = readCamerasFromTransforms(path, "transforms_test.json", white_background, extension, spectral_folder=spectral_folder)
    
    if not eval:
        # train_cam_infos.extend(test_cam_infos)
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(path, "points3d.ply")
    if not os.path.exists(ply_path):
        # Since this data set has no colmap data, we start with random points
        num_pts = 100_000
        print(f"Generating random point cloud ({num_pts})...")
        
        # We create random points inside the bounds of the synthetic Blender scenes
        xyz = np.random.random((num_pts, 3)) * 2.6 - 1.3
        shs = np.random.random((num_pts, 3)) / 255.0
        pcd = BasicPointCloud(points=xyz, colors=SH2RGB(shs), normals=np.zeros((num_pts, 3)))

        storePly(ply_path, xyz, SH2RGB(shs) * 255)
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path, spectral_ids=spectral_ids)
    return scene_info


def readBlenderCTInfo(path, images, eval, n_spectra, n_bcoeffs, spectra_path=None, random_init=False, uniform_init = False, train_split=None, llffhold=8, n_views=100, white_background=False, rgba=False):
    """Read blender format CT data."""
    print("reading CT dataset")

    # reading_dir   = 'spectra/spectral_0'
    # reading_dir = "images" if images == None else images
    # images_folder = os.path.join(path, reading_dir)
    images_folder = os.path.join(path, "spectra", "spectral_0")

    # Read meta data
    meta_data_path = os.path.join(path, "meta_data.json")
    with open(meta_data_path, "r") as handle:
        meta_data = json.load(handle)
    meta_data["ct"] = os.path.join(path, meta_data["ct"])

    if not "dVoxel" in meta_data["scanner"]:
        meta_data["scanner"]["dVoxel"] = list(
            np.array(meta_data["scanner"]["sVoxel"])
            / np.array(meta_data["scanner"]["nVoxel"])
        )
    if not "dDetector" in meta_data["scanner"]:
        meta_data["scanner"]["dDetector"] = list(
            np.array(meta_data["scanner"]["sDetector"])
            / np.array(meta_data["scanner"]["nDetector"])
        )

    #! We will scale the scene so that the volume of interest is in [-1, 1]^3 cube.
    scene_scale = 2 / max(meta_data["scanner"]["sVoxel"])
    for key_to_scale in [
        "dVoxel",
        "sVoxel",
        "sDetector",
        "dDetector",
        "offOrigin",
        "offDetector",
        "DSD",
        "DSO",
    ]:
        meta_data["scanner"][key_to_scale] = (
            np.array(meta_data["scanner"][key_to_scale]) * scene_scale
        ).tolist()

    # Initial reconstructed CT volume (Ground Truth)
    vol_gt = torch.from_numpy(np.load(meta_data["ct"])).float().cuda()

    # Spectral ids
    spectra_dir = 'spectra' if spectra_path== None else spectra_path
    spectral_folder = os.path.join(path, spectra_dir)
    spectral_ids = list(range(1,n_spectra))

    # Cameras (train / test)
    cam_infos_unsorted = readCTameras(meta_data, images_folder, spectral_folder, white_background, rgba, scene_scale)
    cam_infos = sorted(cam_infos_unsorted.copy(), key = lambda x : x.image_name)

    if eval:
        if train_split:
            train_dir = os.path.join(path, "images_train")
            train_names = sorted(os.listdir(train_dir))
            train_names = [train_name.split('.')[0] for train_name in train_names]
            test_dir = os.path.join(path, "images_test")
            test_names = sorted(os.listdir(test_dir))
            test_names = [test_name.split('.')[0] for test_name in test_names]

            train_cam_infos = []
            test_cam_infos = []
            for cam_info in cam_infos:
                if cam_info.image_name in train_names:
                    train_cam_infos.append(cam_info)
                elif cam_info.image_name in test_names:
                    test_cam_infos.append(cam_info)
                else:
                    continue

        else:
            train_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold != 0]
            test_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold == 0]

            if n_views == 100:
                pass 
            elif n_views == 50:
                idx_sub = np.linspace(0, len(train_cam_infos)-1, round(len(train_cam_infos)*0.5)) # 50% views
                idx_sub = [round(i) for i in idx_sub]
                train_cam_infos = [train_cam_infos[i_sub] for i_sub in idx_sub]
            elif isinstance(n_views,int):
                idx_sub = np.linspace(0, len(train_cam_infos)-1, n_views) # 3views
                idx_sub = [round(i) for i in idx_sub]
                train_cam_infos = [train_cam_infos[i_sub] for i_sub in idx_sub]
                print(train_cam_infos)
            else:
                raise NotImplementedError
        print("Training images:     ", len(train_cam_infos))
        print("Testing images:     ", len(test_cam_infos))

    else:
        if train_split:
            train_dir = os.path.join(path, "images_train")
            train_names = sorted(os.listdir(train_dir))
            train_names = [train_name.split('.')[0] for train_name in train_names]
            train_cam_infos = []
            for cam_info in cam_infos:
                if cam_info.image_name in train_names:
                    train_cam_infos.append(cam_info)
            test_cam_infos = []
        else:
            train_cam_infos = cam_infos
            test_cam_infos = []

    # Nerf normalization
    nerf_normalization = getNerfppNorm(train_cam_infos)

    # Point cloud and ply_path
    # n_bcoeffs = n_bcoeffs + 1   # add 1 for 'air' basis coefficient

    if random_init:
        n_points = 50000            # copied from R2-Gaussian
        random_density_max = 1.0    # copied from R2-Gaussian (parameter for random mode)
        
        xyz = np.array(meta_data["scanner"]["offOrigin"])[None, ...] + np.array(
            meta_data["scanner"]["sVoxel"]
        )[None, ...] * (np.random.rand(n_points, 3) - 0.5)
        # opacity = np.random.rand(n_points, 1) * random_density_max
        bcoeffs = np.full((n_points, n_bcoeffs), 1.0 / n_bcoeffs, dtype=np.float32)
        
        ply_path = os.path.join(path, "init_pcd_random.ply")
        # storePly(ply_path, xyz, opacity, bcoeffs)
        storePly(ply_path, xyz, bcoeffs)

        # pcd = BasicPointCloud(points=xyz, opacity=opacity, bcoeffs=bcoeffs)
        pcd = BasicPointCloud(points=xyz, bcoeffs=bcoeffs)
    elif uniform_init:
        points_per_axis = 40
        random_density_max = 1.0    # copied from R2-Gaussian (parameter for random mode)

        sVoxel = np.array(meta_data["scanner"]["sVoxel"])
        offOrigin = np.array(meta_data["scanner"]["offOrigin"])

        vol_min = offOrigin - sVoxel / 2
        vol_max = offOrigin + sVoxel / 2

        xs = np.linspace(vol_min[0], vol_max[0], points_per_axis)
        ys = np.linspace(vol_min[1], vol_max[1], points_per_axis)
        zs = np.linspace(vol_min[2], vol_max[2], points_per_axis)
        X, Y, Z = np.meshgrid(xs, ys, zs, indexing='ij')
        xyz = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=1)
        # opacity = np.random.rand(xyz.shape[0], 1) * random_density_max
        bcoeffs = np.full((n_points, n_bcoeffs), 1.0 / n_bcoeffs, dtype=np.float32)

        ply_path = os.path.join(path, "init_pcd_uniform.ply")
        # storePly(ply_path, xyz, opacity, bcoeffs)
        storePly(ply_path, xyz, bcoeffs)

        # pcd = BasicPointCloud(points=xyz, opacity=opacity, bcoeffs=bcoeffs)
        pcd = BasicPointCloud(points=xyz, bcoeffs=bcoeffs)
    else:
        npy_path = os.path.join(path, "init_pcd.npy")
        point_cloud = np.load(npy_path)
        
        xyz = point_cloud[:, :3]
        
        # sigma   = 0.05
        # uniform = 1.0 / n_bcoeffs

        # bcoeffs = uniform + np.random.normal(0.0, sigma, size=(point_cloud.shape[0], n_bcoeffs))
        # bcoeffs = np.clip(bcoeffs, 1e-6, None)
        # bcoeffs = bcoeffs.astype(np.float32)

        # ply_path = os.path.join(path, "init_pcd.ply")
        # storePly(ply_path, xyz, bcoeffs)

        # pcd = BasicPointCloud(points=xyz, bcoeffs=bcoeffs)

        # rhos = uniform + np.random.normal(0.0, sigma, size=(point_cloud.shape[0], n_bcoeffs))
        # rhos = np.clip(rhos, 1e-6, None)
        # rhos = rhos.astype(np.float32)

        # rhos = point_cloud[:, 3:4] * 10.0
        rhos = point_cloud[:, 3:4]

        # rho = 0.01759
        # rho = 1.0
        # rhos = np.full((point_cloud.shape[0], 1), rho, dtype=np.float32)
        # rhos = np.repeat(rhos, 2, axis=1)
        print("RHOS: ", rhos.mean())

        # ratio = [0.6, 0.4]
        # fractions = np.tile(ratio, (point_cloud.shape[0], 1)).astype(np.float32)
        # fractions += np.random.normal(0, 0.01, size=fractions.shape)  # Increased noise stddev to 0.4
        # fractions = np.clip(fractions, 1e-6, None)
        # fractions = fractions / fractions.sum(axis=1, keepdims=True)
        if n_bcoeffs == 2:
            # fractions = analytic_fraction_2(rhos[:,0])
            fractions, rho0_used, alpha_used = pe_compton_from_density(rhos[:,0])
        # else:
        #     fractions = analytic_fraction_3(rhos[:,0])
        print("FRACS: ", fractions[:,0].mean(), fractions[:,1].mean())

        # global_frac = np.array([0.0329, 0.9671], dtype=np.float32)
        # fractions = np.tile(global_frac, (point_cloud.shape[0], 1))

        ply_path = os.path.join(path, "init_pcd.ply")
        storePly(ply_path, xyz, rhos, fractions)

        pcd = BasicPointCloud(points=xyz, rhos=rhos, fracs=fractions)

    scene_info = SceneInfoSpecCT(
        point_cloud=pcd,
        scanner_cfg=meta_data["scanner"],
        vol=vol_gt,
        scene_scale=scene_scale,
        spectral_ids=spectral_ids,
        train_cameras=train_cam_infos,
        test_cameras=test_cam_infos,
        nerf_normalization=nerf_normalization,
        ply_path=ply_path
    )
    
    return scene_info


def readCTameras(meta_data, images_folder, spectral_folder, white_background, rgba, scene_scale=1.0):
    """Read camera info."""
    cam_cfg = meta_data["scanner"]

    cams = meta_data["cameras"]

    cam_infos = []
    for idx, cam in enumerate(cams):
        sys.stdout.write("\r")
        sys.stdout.write(f"Reading camera {idx + 1}")
        sys.stdout.flush()

        frame_angle = cam["angle"]
        mode = mode_id[cam_cfg["mode"]]

        # CT 'transform_matrix' is a camera-to-world transform
        c2w = angle2pose(cam_cfg["DSO"], frame_angle)  # c2w
        # get the world-to-camera transform and set R, T
        w2c = np.linalg.inv(c2w)
        R = np.transpose(
            w2c[:3, :3]
        )  # R is stored transposed due to 'glm' in CUDA code
        T = w2c[:3, 3]

        # Note, dDetector is [v, u] not [u, v]
        FovX = np.arctan2(cam_cfg["sDetector"][1] / 2, cam_cfg["DSD"]) * 2
        FovY = np.arctan2(cam_cfg["sDetector"][0] / 2, cam_cfg["DSD"]) * 2

        # Image
        image_name = cam["file_path"]
        # image_path = os.path.join(images_folder, image_name)
        # image = np.load(image_path) * scene_scale
        # image = (image * 255).astype(np.uint8)  # Convert to uint8
        # image = Image.fromarray(image).convert("RGBA") # Convert to RGBA (grayscale with alpha)
        # if rgba:
        #     image = fix_background_rgba(image, white_background)

        # Spectral images
        # spectral_list = sorted(glob(spectral_folder + '/*'))
        spectral_images = []
        # for spectra in spectral_list:
        #     spectra_img_path = os.path.join(spectral_folder, spectra, image_name)
        for spectra_dir in sorted(os.listdir(spectral_folder)):
            spectra_img_path = os.path.join(spectral_folder, spectra_dir, image_name)
            # spectra_img = np.load(spectra_img_path) * scene_scale
            # spectral_images.append(spectra_img)

            spectra_np   = np.load(spectra_img_path).astype(np.float32) * scene_scale  # (H,W)
            spectra_img  = torch.from_numpy(spectra_np).unsqueeze(0)     # (1,H,W)
            spectral_images.append(spectra_img)

            # spectra_img = (spectra_img * 255).astype(np.uint8)  # Convert to uint8
            # spectra_img = Image.fromarray(spectra_img).convert("RGBA") # Convert to RGBA (grayscale with alpha)
            # if rgba:
            #     spectra_img = fix_background_rgba(spectra_img, white_background)

        # fig, axes = plt.subplots(1, 2, figsize=(6, 3), constrained_layout=True)

        # axes[0].imshow(spectral_images[0], cmap='gray')
        # axes[1].imshow(spectral_images[1], cmap='gray')

        # plt.show()
        # return

        image = spectral_images[0].clone()
        image_path = os.path.join(spectral_folder, "spectral_0", image_name)

        cam_info = CameraInfoSpecCT(
            uid=idx,
            R=R,
            T=T,
            angle=frame_angle,
            FovY=FovY,
            FovX=FovX,
            image=image,
            image_path=image_path,
            image_name=os.path.basename(image_path).split(".")[0],
            width=cam_cfg["nDetector"][1],
            height=cam_cfg["nDetector"][0],
            objects=None,
            spectral_images=spectral_images,
            mode=mode,
            scanner_cfg=cam_cfg,
        )

        cam_infos.append(cam_info)
    sys.stdout.write("\n")
    return cam_infos

def angle2pose(DSO, angle):
    """Transfer angle to pose (c2w) based on scanner geometry.
    1. rotate -90 degree around x-axis (fixed axis),
    2. rotate 90 degree around z-axis  (fixed axis),
    3. rotate angle degree around z axis  (fixed axis)"""

    phi1 = -np.pi / 2
    R1 = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, np.cos(phi1), -np.sin(phi1)],
            [0.0, np.sin(phi1), np.cos(phi1)],
        ]
    )
    phi2 = np.pi / 2
    R2 = np.array(
        [
            [np.cos(phi2), -np.sin(phi2), 0.0],
            [np.sin(phi2), np.cos(phi2), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    R3 = np.array(
        [
            [np.cos(angle), -np.sin(angle), 0.0],
            [np.sin(angle), np.cos(angle), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    rot = np.dot(np.dot(R3, R2), R1)
    trans = np.array([DSO * np.cos(angle), DSO * np.sin(angle), 0])
    transform = np.eye(4)
    transform[:3, :3] = rot
    transform[:3, 3] = trans

    return transform

sceneLoadTypeCallbacks = {
    "Colmap": readColmapSceneInfo,
    "Blender" : readNerfSyntheticInfo,
    "CT" : readBlenderCTInfo
}
