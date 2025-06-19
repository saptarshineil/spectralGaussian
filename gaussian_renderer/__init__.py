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

import torch
import math
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from scene.gaussian_model import GaussianModel
from utils.sh_utils import eval_sh
from utils.graphics_utils import normal_from_depth_image
from utils.general_utils import flip_align_view
from scene.NVDIFFREC import extract_env_map

from xray_gaussian_rasterization_voxelization import (
    GaussianRasterizationSettings_ct,
    GaussianRasterizer_ct,
    GaussianVoxelizationSettings,
    GaussianVoxelizer,
)

import matplotlib.pyplot as plt
import xraylib


def rendered_world2cam(viewpoint_cam, normal, alpha, bg_color):
    # normal: (3, H, W), alpha: (H, W), bg_color: (3)
    # normal_cam: (3, H, W)
    _, H, W = normal.shape
    intrinsic_matrix, extrinsic_matrix = viewpoint_cam.get_calib_matrix_nerf()
    normal_world = normal.permute(1,2,0).reshape(-1, 3) # (HxW, 3)
    normal_cam = torch.cat([normal_world, torch.ones_like(normal_world[...,0:1])], axis=-1) @ torch.inverse(torch.inverse(extrinsic_matrix).transpose(0,1))[...,:3]
    normal_cam = normal_cam.reshape(H, W, 3).permute(2,0,1) # (H, W, 3)
    
    background = bg_color[...,None,None]
    normal_cam = normal_cam*alpha[None,...] + background*(1. - alpha[None,...])

    return normal_cam

def render_normal(viewpoint_cam, depth, bg_color, alpha):
    # depth: (H, W), bg_color: (3), alpha: (H, W)
    # normal_ref: (3, H, W)
    intrinsic_matrix, extrinsic_matrix = viewpoint_cam.get_calib_matrix_nerf()

    normal_ref = normal_from_depth_image(depth, intrinsic_matrix.to(depth.device), extrinsic_matrix.to(depth.device))
    background = bg_color[None,None,...]
    normal_ref = normal_ref*alpha[...,None] + background*(1. - alpha[...,None])
    normal_ref = normal_ref.permute(2,0,1)

    return normal_ref

# render 360 lighting for a single gaussian
def render_lighting(pc : GaussianModel, resolution=(512, 1024), sampled_index=None, specral_id=None):
    if pc.brdf_mode=="envmap":
        lighting = extract_env_map(pc.brdf_mlp[specral_id], resolution) # (H, W, 3)
        lighting = lighting.permute(2,0,1) # (3, H, W)
    else:
        raise NotImplementedError

    return lighting

def normalize_normal_inplace(normal, alpha):
    # normal: (3, H, W), alpha: (H, W)
    fg_mask = (alpha[None,...]>0.).repeat(3, 1, 1)
    normal = torch.where(fg_mask, torch.nn.functional.normalize(normal, p=2, dim=0), normal)


def render_bcoeffs(
    viewpoint_camera,
    pc : GaussianModel, 
    pipe, 
    bcoeff_id=0,
    scaling_modifier = 1.0, 
    override_color = None, 
    debug=False, 
    speed=False, 
):
    """
    Render the scene. 
    Background tensor (bg_color) must be on GPU!
    """
 
    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    # -- Spectral --
    # tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    # tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    # -- R2-Gaussian --
    mode = viewpoint_camera.mode
    if mode == 0:
        tanfovx = 1.0
        tanfovy = 1.0
    elif mode == 1:
        tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
        tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)
    else:
        raise ValueError("Unsupported mode!")

    # -- R2-Gaussian --
    bcoeffs = pc.get_bcoeffs()
    raster_settings_ct = GaussianRasterizationSettings_ct(
        num_channels = bcoeffs.shape[1], 
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        mode=viewpoint_camera.mode,
        debug=pipe.debug,
    )

    rasterizer_ct = GaussianRasterizer_ct(raster_settings=raster_settings_ct)

    means3D = pc.get_xyz
    means2D = screenspace_points
    
    # opacity = pc.get_bcoeff(bcoeff_id)
     # shape = (num_gaussians, num_basis_materials)
    # fractions = torch.ones_like(bcoeffs, requires_grad=True, device="cuda") # shape = (num_gaussians, num_basis_materials)

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = None
    if pipe.compute_cov3D_python:
        cov3D_precomp = pc.get_covariance(scaling_modifier)
    else:
        scales = pc.get_scaling
        rotations = pc.get_rotation

    # -- R2-Gaussian --
    # rendered_objects = None
    rendered_image, radii = rasterizer_ct(
        means3D, means2D, bcoeffs,
        # means3D=means3D,
        # means2D=means2D,
        # bcoeffs=bcoeffs,
        # fractions=fractions,
        scales=scales,
        rotations=rotations,
        cov3D_precomp=cov3D_precomp,
    )

    # print(rendered_image.shape)
    # img = rendered_image.detach().cpu().numpy()   # shape (2, 140, 140)

    # # 2.  Plot the two channels side-by-side
    # fig, axes = plt.subplots(1, 2, figsize=(6, 3), constrained_layout=True)

    # for ch in range(2):
    #     axes[ch].imshow(img[ch], cmap='gray', vmin=img.min(), vmax=img.max())
    #     axes[ch].set_title(f'Channel {ch}')
    #     axes[ch].axis('off')

    # plt.show()
    # return
    
    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
        
    out = {
        "render": rendered_image,
        "viewspace_points": screenspace_points,
        "visibility_filter" : radii > 0,
        "radii": radii,
    }

    return out


def sigma_kn(E: torch.Tensor) -> torch.Tensor:
    # constants
    r_e = 2.8179403262e-13           # classical electron radius [cm]
    sigma_T = 8.0 * torch.pi * r_e**2 / 3.0   # Thomson σ  [cm²]
    eps = E / 511.0              # hν / mₑc², dimensionless
    one = torch.ones_like(eps)

    term1 = (1.0 + eps) / eps**3
    term2 = 2*eps*(1+eps)/(1+2*eps)
    term3 = torch.log1p(2*eps)       # ln(1+2ε) • numerically stable
    sigma = sigma_T * term1 * (term2 - term3)

    return sigma


def render(bcoeff_image, E):
    # mu_PE = 5.31e3 * E**(-3) * bcoeff_image[0]
    mu_PE = xraylib.CS_Photo(6, float(E)) * bcoeff_image[0]
    # mu_C = sigma_kn(E) * bcoeff_image[1]
    mu_C = xraylib.CS_KN(float(E)) * bcoeff_image[1]

    image = mu_PE + mu_C

    return image.unsqueeze(0)
