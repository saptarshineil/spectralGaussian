import os
import numpy as np
import tigre.algorithms as algs
import open3d as o3d
import sys
import argparse
import os.path as osp
import json
import pickle
from tqdm import trange
import copy
import torch
from r2_gaussian.utils.ct_utils import recon_volume


def init_pcd(projs, angles, geo, scanner_cfg, args, save_path):
    "Initialize Gaussians."
    recon_method = args["recon_method"]
    n_points = args["n_points"]
    assert recon_method in ["random", "fdk"], "--recon_method not supported."
    if recon_method == "random":
        print(f"Initialize random point clouds.")
        sampled_positions = np.array(scanner_cfg["offOrigin"])[None, ...] + np.array(
            scanner_cfg["sVoxel"]
        )[None, ...] * (np.random.rand(n_points, 3) - 0.5)
        sampled_densities = (
            np.random.rand(
                n_points,
            )
            * args["random_density_max"]
        )
    else:
        # Use traditional algorithms for initialization
        print(
            f"Initialize point clouds with the volume reconstructed from {recon_method}."
        )
        vol = recon_volume(projs, angles, copy.deepcopy(geo), recon_method)
        # show_one_volume(vol)

        density_mask = vol > args["density_thresh"]
        valid_indices = np.argwhere(density_mask)
        offOrigin = np.array(scanner_cfg["offOrigin"])
        dVoxel = np.array(scanner_cfg["dVoxel"])
        sVoxel = np.array(scanner_cfg["sVoxel"])

        assert (
            valid_indices.shape[0] >= n_points
        ), "Valid voxels less than target number of sampling. Check threshold"

        sampled_indices = valid_indices[
            np.random.choice(len(valid_indices), n_points, replace=False)
        ]
        sampled_positions = sampled_indices * dVoxel - sVoxel / 2 + offOrigin
        sampled_densities = vol[
            sampled_indices[:, 0],
            sampled_indices[:, 1],
            sampled_indices[:, 2],
        ]
        sampled_densities = sampled_densities * args["density_rescale"]

    out = np.concatenate([sampled_positions, sampled_densities[:, None]], axis=-1)
    np.save(save_path, out)
    print(f"Initialization saved in {save_path}.")
