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
import numpy as np
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from scene.gaussian_model import GaussianModel
from utils.sh_utils import eval_sh
from utils.graphics_utils import normal_from_depth_image
from utils.general_utils import flip_align_view
from scene.NVDIFFREC import extract_env_map

# Spectral CT rasterizer
from xray_gaussian_rasterization_voxelization import (
    GaussianRasterizationSettings_ct,
    GaussianRasterizer_ct,
    GaussianVoxelizationSettings,
    GaussianVoxelizer,
)

# Old R2-Gaussian rasterizer
from xray_gaussian_rasterization_voxelization_r2 import (
    GaussianRasterizationSettings,
    GaussianRasterizer,
    GaussianVoxelizationSettings,
    GaussianVoxelizer,
)

import matplotlib.pyplot as plt
import xraylib
# import numpy as np
# import scipy.constants as sc


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


def render_r2(
    viewpoint_camera,
    pc,
    pipe,
    scaling_modifier=1.0,
):
    """
    Render an X-ray projection with rasterization.
    """

    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = (
        torch.zeros_like(
            pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda"
        )
        + 0
    )
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    mode = viewpoint_camera.mode
    if mode == 0:
        tanfovx = 1.0
        tanfovy = 1.0
    elif mode == 1:
        tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
        tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)
    else:
        raise ValueError("Unsupported mode!")

    raster_settings = GaussianRasterizationSettings(
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

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    means3D = pc.get_xyz
    means2D = screenspace_points
    # density = pc.get_opacity_old
    density = pc.get_rhos
    # density = pc.get_opacity_at_energy(spectral_id)

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

    # Rasterize visible Gaussians to image, obtain their radii (on screen).
    rendered_image, radii = rasterizer(
        means3D=means3D,
        means2D=means2D,
        opacities=density,
        scales=scales,
        rotations=rotations,
        cov3D_precomp=cov3D_precomp,
    )
    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    return {
        "render": rendered_image,
        "viewspace_points": screenspace_points,
        "visibility_filter": radii > 0,
        "radii": radii,
    }


def render_bcoeffs(
    viewpoint_camera,
    pc : GaussianModel, 
    pipe, 
    spectral_id=0,
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
    bcoeffs = pc.get_bcoeffs

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

    rendered_bcoeffs, radii = rasterizer_ct(
        means3D=means3D,
        means2D=means2D,
        bcoeffs=bcoeffs,
        scales=scales,
        rotations=rotations,
        cov3D_precomp=cov3D_precomp,
    )
    
    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    out = {
        "bcoeffs": rendered_bcoeffs,
        "viewspace_points": screenspace_points,
        "visibility_filter" : radii > 0,
        "radii": radii,
    }

    return out



def render_fracs(
    viewpoint_camera,
    pc : GaussianModel, 
    pipe, 
    spectral_id=0,
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
    bcoeffs = pc.get_fracs

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

    rendered_bcoeffs, radii = rasterizer_ct(
        means3D=means3D,
        means2D=means2D,
        bcoeffs=bcoeffs,
        scales=scales,
        rotations=rotations,
        cov3D_precomp=cov3D_precomp,
    )

    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    out = {
        "fracs": rendered_bcoeffs,
        "viewspace_points": screenspace_points,
        "visibility_filter" : radii > 0,
        "radii": radii,
    }

    return out

# def render_fracs(
#     viewpoint_camera,
#     pc : GaussianModel, 
#     pipe, 
#     spectral_id=0,
#     scaling_modifier = 1.0, 
#     override_color = None, 
#     debug=False, 
#     speed=False, 
# ):
#     """
#     Render the scene. 
#     Background tensor (bg_color) must be on GPU!
#     """
 
#     # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
#     screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
#     try:
#         screenspace_points.retain_grad()
#     except:
#         pass

#     # Set up rasterization configuration
#     # -- Spectral --
#     # tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
#     # tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

#     # -- R2-Gaussian --
#     mode = viewpoint_camera.mode
#     if mode == 0:
#         tanfovx = 1.0
#         tanfovy = 1.0
#     elif mode == 1:
#         tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
#         tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)
#     else:
#         raise ValueError("Unsupported mode!")
    

#     # -- R2-Gaussian --
#     # bcoeffs = pc.get_fracs
#     bcoeffs = pc.get_bcoeffs

#     raster_settings_ct = GaussianRasterizationSettings_ct(
#         num_channels = bcoeffs.shape[1], 
#         image_height=int(viewpoint_camera.image_height),
#         image_width=int(viewpoint_camera.image_width),
#         tanfovx=tanfovx,
#         tanfovy=tanfovy,
#         scale_modifier=scaling_modifier,
#         viewmatrix=viewpoint_camera.world_view_transform,
#         projmatrix=viewpoint_camera.full_proj_transform,
#         campos=viewpoint_camera.camera_center,
#         prefiltered=False,
#         mode=viewpoint_camera.mode,
#         debug=pipe.debug,
#     )

#     rasterizer_ct = GaussianRasterizer_ct(raster_settings=raster_settings_ct)

#     means3D = pc.get_xyz
#     means2D = screenspace_points
    
#     # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
#     # scaling / rotation by the rasterizer.
#     scales = None
#     rotations = None
#     cov3D_precomp = None
#     if pipe.compute_cov3D_python:
#         cov3D_precomp = pc.get_covariance(scaling_modifier)
#     else:
#         scales = pc.get_scaling
#         rotations = pc.get_rotation

#     # -- R2-Gaussian --
#     # rendered_objects = None

#     rendered_bcoeffs, radii = rasterizer_ct(
#         means3D=means3D,
#         means2D=means2D,
#         bcoeffs=bcoeffs,
#         scales=scales,
#         rotations=rotations,
#         cov3D_precomp=cov3D_precomp,
#     )

#     # print(rendered_image.shape)

#     # img = rendered_image.detach().cpu().numpy()   # shape (2, 140, 140)
#     # plt.imshow(img[0])
#     # plt.show()

#     # # 2.  Plot the two channels side-by-side
#     # fig, axes = plt.subplots(1, 2, figsize=(6, 3), constrained_layout=True)

#     # for ch in range(2):
#     #     axes[ch].imshow(img[ch], cmap='gray', vmin=img.min(), vmax=img.max())
#     #     axes[ch].set_title(f'Channel {ch}')
#     #     axes[ch].axis('off')

#     # plt.show()
    
#     # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
#     # They will be excluded from value updates used in the splitting criteria.
#     # out = {
#     #     "bcoeffs": rendered_bcoeffs,
#     #     "viewspace_points": screenspace_points,
#     #     "visibility_filter" : radii > 0,
#     #     "radii": radii,
#     # }

#     # return out

#     rho_img = rendered_bcoeffs.sum(dim=0, keepdim=True)          # (1, H, W)

#     # 3) per-pixel fractions in [0,1], sum to 1 (where rho>0)
#     frac_img = rendered_bcoeffs / (rho_img.clamp_min(1e-8))      # (C, H, W)

#     out = {
#         "fracs": frac_img,
#         "bcoeffs": rendered_bcoeffs,
#         "rho": rho_img,
#         "viewspace_points": screenspace_points,
#         "visibility_filter": radii > 0,
#         "radii": radii,
#     }
#     return out

# # def render_mu_mask(
# #     viewpoint_camera,
# #     pc : GaussianModel, 
# #     pipe, 
# #     spectral_id=0,
# #     basis_mask=None,
# #     scaling_modifier = 1.0, 
# #     override_color = None, 
# #     debug=False, 
# #     speed=False, 
# # ):
# #     """
# #     Render the scene. 
# #     Background tensor (bg_color) must be on GPU!
# #     """
 
# #     # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
# #     screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
# #     try:
# #         screenspace_points.retain_grad()
# #     except:
# #         pass

# #     # Set up rasterization configuration
# #     # -- Spectral --
# #     # tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
# #     # tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

# #     # -- R2-Gaussian --
# #     mode = viewpoint_camera.mode
# #     if mode == 0:
# #         tanfovx = 1.0
# #         tanfovy = 1.0
# #     elif mode == 1:
# #         tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
# #         tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)
# #     else:
# #         raise ValueError("Unsupported mode!")
    

# #     # -- R2-Gaussian --
# #     # bcoeffs = pc.get_bcoeffs
# #     fracs = pc.get_fracs
# #     rhos = pc.get_rhos

# #     if basis_mask is not None:
# #         if not torch.is_tensor(basis_mask):
# #             basis_mask = torch.tensor(basis_mask, device=fracs.device,
# #                                         dtype=fracs.dtype)
# #         basis_mask = basis_mask.to(device=fracs.device, dtype=fracs.dtype)

# #         if basis_mask.ndim == 1:
# #             if basis_mask.shape[0] == fracs.shape[1]:
# #                 # (C,) → (1,C) → (N,C)
# #                 basis_mask = basis_mask.unsqueeze(0).expand(fracs.shape[0], -1)
# #             elif basis_mask.shape[0] == fracs.shape[0]:
# #                 # (N,) → (N,1) → (N,C)
# #                 basis_mask = basis_mask.unsqueeze(1).expand(-1, fracs.shape[1])
# #             else:
# #                 raise ValueError(f"basis_mask length {basis_mask.shape[0]} "
# #                                     f"must equal C={fracs.shape[1]} or N={fracs.shape[0]}")
# #         elif basis_mask.ndim == 2:
# #             # Try to broadcast (1,C) or (N,1)
# #             if basis_mask.shape == (1, fracs.shape[1]):
# #                 basis_mask = basis_mask.expand(fracs.shape[0], -1)
# #             elif basis_mask.shape == (fracs.shape[0], 1):
# #                 basis_mask = basis_mask.expand(-1, fracs.shape[1])
# #             elif basis_mask.shape != fracs.shape:
# #                 raise ValueError(f"basis_mask has shape {basis_mask.shape}, "
# #                                     f"expected (N,C)={(fracs.shape[0], fracs.shape[1])}, "
# #                                     f"(1,C), or (N,1)")
# #         else:
# #             raise ValueError("basis_mask must be 1D or 2D")
        
# #     fracs = fracs * basis_mask
# #     bcoeffs = rhos * fracs

# #     raster_settings_ct = GaussianRasterizationSettings_ct(
# #         num_channels = bcoeffs.shape[1], 
# #         image_height=int(viewpoint_camera.image_height),
# #         image_width=int(viewpoint_camera.image_width),
# #         tanfovx=tanfovx,
# #         tanfovy=tanfovy,
# #         scale_modifier=scaling_modifier,
# #         viewmatrix=viewpoint_camera.world_view_transform,
# #         projmatrix=viewpoint_camera.full_proj_transform,
# #         campos=viewpoint_camera.camera_center,
# #         prefiltered=False,
# #         mode=viewpoint_camera.mode,
# #         debug=pipe.debug,
# #     )

# #     rasterizer_ct = GaussianRasterizer_ct(raster_settings=raster_settings_ct)

# #     means3D = pc.get_xyz
# #     means2D = screenspace_points
    
# #     # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
# #     # scaling / rotation by the rasterizer.
# #     scales = None
# #     rotations = None
# #     cov3D_precomp = None
# #     if pipe.compute_cov3D_python:
# #         cov3D_precomp = pc.get_covariance(scaling_modifier)
# #     else:
# #         scales = pc.get_scaling
# #         rotations = pc.get_rotation

# #     # -- R2-Gaussian --
# #     # rendered_objects = None

# #     rendered_bcoeffs, radii = rasterizer_ct(
# #         means3D=means3D,
# #         means2D=means2D,
# #         bcoeffs=bcoeffs,
# #         scales=scales,
# #         rotations=rotations,
# #         cov3D_precomp=cov3D_precomp,
# #     )

# #     # print(rendered_image.shape)

# #     # img = rendered_image.detach().cpu().numpy()   # shape (2, 140, 140)
# #     # plt.imshow(img[0])
# #     # plt.show()

# #     # # 2.  Plot the two channels side-by-side
# #     # fig, axes = plt.subplots(1, 2, figsize=(6, 3), constrained_layout=True)

# #     # for ch in range(2):
# #     #     axes[ch].imshow(img[ch], cmap='gray', vmin=img.min(), vmax=img.max())
# #     #     axes[ch].set_title(f'Channel {ch}')
# #     #     axes[ch].axis('off')

# #     # plt.show()
    
# #     # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
# #     # They will be excluded from value updates used in the splitting criteria.
# #     out = {
# #         "bcoeffs": rendered_bcoeffs,
# #         "viewspace_points": screenspace_points,
# #         "visibility_filter" : radii > 0,
# #         "radii": radii,
# #     }

# #     return out

def _row_normalize(x, eps=1e-12):
    return x / (x.norm(dim=1, keepdim=True) + eps)

def _broadcast_mask(basis_mask, fracs):
    # Your existing broadcasting logic, wrapped
    if not torch.is_tensor(basis_mask):
        basis_mask = torch.tensor(basis_mask, device=fracs.device, dtype=fracs.dtype)
    basis_mask = basis_mask.to(device=fracs.device, dtype=fracs.dtype)

    if basis_mask.ndim == 1:
        if basis_mask.shape[0] == fracs.shape[1]:
            basis_mask = basis_mask.unsqueeze(0).expand(fracs.shape[0], -1)
        elif basis_mask.shape[0] == fracs.shape[0]:
            basis_mask = basis_mask.unsqueeze(1).expand(-1, fracs.shape[1])
        else:
            raise ValueError("basis_mask length must be C or N")
    elif basis_mask.ndim == 2:
        if basis_mask.shape == (1, fracs.shape[1]):
            basis_mask = basis_mask.expand(fracs.shape[0], -1)
        elif basis_mask.shape == (fracs.shape[0], 1):
            basis_mask = basis_mask.expand(-1, fracs.shape[1])
        elif basis_mask.shape != fracs.shape:
            raise ValueError(f"basis_mask has shape {basis_mask.shape}, expected (N,C), (1,C), or (N,1)")
    else:
        raise ValueError("basis_mask must be 1D or 2D")
    return basis_mask


def render_mu_mask(
    viewpoint_camera,
    pc: GaussianModel, 
    pipe, 
    spectral_id=0,
    ratio=None,     # <--- formerly basis_mask
    channel_gate=None,       # <--- new
    scaling_modifier=1.0, 
    override_color=None, 
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
    

    fracs = pc.get_fracs
    rhos = pc.get_rhos
    bcoeffs_nomask = rhos * fracs

    raster_settings_ct = GaussianRasterizationSettings_ct(
        num_channels = bcoeffs_nomask.shape[1], 
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

    # if ratio is not None:
    #     basis_mask = _broadcast_mask(ratio, fracs)
    #     masked_bcoeffs = bcoeffs_nomask * basis_mask
    #     eps = 1e-12  # tune
    #     keep_bool = (masked_bcoeffs.abs().sum(dim=1) > eps)
    # else:
    #     # No mask => keep all
    #     keep_bool = torch.ones(fracs.shape[0], dtype=torch.bool, device=fracs.device)

    if ratio is not None:
        T = _broadcast_mask(ratio, fracs)    # (N,C)
        f_dir = _row_normalize(fracs)
        t_dir = _row_normalize(T)
        cos_sim = (f_dir * t_dir).sum(dim=1).clamp(-1.0, 1.0)  # (N,)

        # STRICT keep rule: only keep near-template gaussians
        # If you truly expect ratios ~ [0.01, 0.99], set high threshold
        tau_keep = 0.999   # try 0.985–0.999 depending on how selective you want
        keep_bool = cos_sim >= tau_keep

        # Optional: soft gate if you want a ramp (comment out if you want hard-only)
        # tau_prune = 0.95
        # gate = ((cos_sim - tau_prune) / (tau_keep - tau_prune)).clamp(0, 1)
    else:
        keep_bool = torch.ones(fracs.shape[0], dtype=torch.bool, device=fracs.device)
    
    means3D_sel = pc.get_xyz[keep_bool]
    means2D     = screenspace_points
    scales_sel  = pc.get_scaling[keep_bool]          if not pipe.compute_cov3D_python else None
    rots_sel    = pc.get_rotation[keep_bool]         if not pipe.compute_cov3D_python else None
    cov3D_sel   = pc.get_covariance(scaling_modifier)[keep_bool] if pipe.compute_cov3D_python else None

    bcoeffs_sel = bcoeffs_nomask[keep_bool]

    # apply binary channel gate *only* if you want to zero certain bases
    if channel_gate is not None:
        G = _broadcast_mask(channel_gate, fracs)    # (N,C)
        bcoeffs_sel = bcoeffs_sel * G[keep_bool]   # zero masked channels; no brightness scaling

    print("COS stats:", cos_sim.min().item(), cos_sim.mean().item(), cos_sim.median().item(), cos_sim.max().item())
    print("Kept:", int(keep_bool.sum().item()), "Pruned:", int((~keep_bool).sum().item()))
    # print("fracs per-channel mean:", fracs.mean(dim=0))
    # print("fracs per-channel min :", fracs.min(dim=0).values)
    # print("fracs per-channel max :", fracs.max(dim=0).values)

    # Rasterize only survivors
    rendered_bcoeffs, radii = rasterizer_ct(
        means3D=means3D_sel,
        means2D=means2D[keep_bool],  # keep screenspace_points aligned with survivors for gradients
        bcoeffs=bcoeffs_sel,
        scales=scales_sel,
        rotations=rots_sel,
        cov3D_precomp=cov3D_sel,
    )

    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    out = {
        "bcoeffs": rendered_bcoeffs,
        "viewspace_points": screenspace_points,
        "visibility_filter" : radii > 0,
        "radii": radii,
    }

    return out

# def render_mu_mask(
#     viewpoint_camera,
#     pc : GaussianModel, 
#     pipe, 
#     spectral_id=0,
#     basis_mask=None,
#     scaling_modifier = 1.0, 
#     override_color = None, 
#     debug=False, 
#     speed=False,
#     soft_mode="none",      # new: "none"|"cosine"|"l2"|"projection"
#     sigma=0.1,             # only used if soft_mode=="l2"
# ):
#     """
#     Render the scene with optional soft masking.
#     """

#     screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
#     try:
#         screenspace_points.retain_grad()
#     except:
#         pass

#     mode = viewpoint_camera.mode
#     if mode == 0:
#         tanfovx = 1.0; tanfovy = 1.0
#     elif mode == 1:
#         tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
#         tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)
#     else:
#         raise ValueError("Unsupported mode!")

#     fracs = pc.get_fracs
#     rhos = pc.get_rhos

#     if basis_mask is not None:
#         if not torch.is_tensor(basis_mask):
#             basis_mask = torch.tensor(basis_mask, device=fracs.device, dtype=fracs.dtype)
#         basis_mask = basis_mask.to(device=fracs.device, dtype=fracs.dtype)

#         # normalize target
#         target = basis_mask / (basis_mask.sum() + 1e-8)
#         target = target.view(1, -1)  # (1,C)

#         fracs_norm = fracs / (fracs.sum(dim=1, keepdim=True) + 1e-8)

#         if soft_mode == "none":
#             # old behavior: hard multiply
#             fracs = fracs * target

#         elif soft_mode == "cosine":
#             sim = torch.nn.functional.cosine_similarity(fracs_norm, target, dim=1).clamp(min=0.0)
#             weights = sim[:, None]
#             fracs = fracs * weights

#         elif soft_mode == "l2":
#             d2 = ((fracs_norm - target) ** 2).sum(dim=1)
#             weights = torch.exp(-d2 / (2 * sigma**2))
#             fracs = fracs * weights[:, None]

#         elif soft_mode == "projection":
#             coeff = (fracs * target).sum(dim=1, keepdim=True)
#             fracs = coeff * target

#         else:
#             raise ValueError(f"Unknown soft_mode {soft_mode}")

#     bcoeffs = rhos * fracs

#     raster_settings_ct = GaussianRasterizationSettings_ct(
#         num_channels = bcoeffs.shape[1], 
#         image_height=int(viewpoint_camera.image_height),
#         image_width=int(viewpoint_camera.image_width),
#         tanfovx=tanfovx,
#         tanfovy=tanfovy,
#         scale_modifier=scaling_modifier,
#         viewmatrix=viewpoint_camera.world_view_transform,
#         projmatrix=viewpoint_camera.full_proj_transform,
#         campos=viewpoint_camera.camera_center,
#         prefiltered=False,
#         mode=viewpoint_camera.mode,
#         debug=pipe.debug,
#     )

#     rasterizer_ct = GaussianRasterizer_ct(raster_settings=raster_settings_ct)

#     means3D = pc.get_xyz
#     means2D = screenspace_points

#     scales = rotations = cov3D_precomp = None
#     if pipe.compute_cov3D_python:
#         cov3D_precomp = pc.get_covariance(scaling_modifier)
#     else:
#         scales = pc.get_scaling
#         rotations = pc.get_rotation

#     rendered_bcoeffs, radii = rasterizer_ct(
#         means3D=means3D,
#         means2D=means2D,
#         bcoeffs=bcoeffs,
#         scales=scales,
#         rotations=rotations,
#         cov3D_precomp=cov3D_precomp,
#     )

#     out = {
#         "bcoeffs": rendered_bcoeffs,
#         "viewspace_points": screenspace_points,
#         "visibility_filter": radii > 0,
#         "radii": radii,
#     }

#     return out

# def render_mu_mask(
#     viewpoint_camera,
#     pc : GaussianModel, 
#     pipe, 
#     spectral_id=0,
#     basis_mask=None,
#     scaling_modifier = 1.0, 
#     override_color = None, 
#     debug=False, 
#     speed=False,
#     soft_mode="none",      # "none"|"cosine"|"l2"|"projection"
#     sigma=0.1,             # only if soft_mode=="l2"
#     gaussian_mask=None,    # NEW: (N,) or (N,1) boolean/float mask over Gaussians
# ):
#     """
#     Render the scene with optional masks.
#       - basis_mask   : soft selection in basis space (operates on fractions).
#       - gaussian_mask: hard selection of Gaussians (operates on densities).
#     """

#     screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
#     try:
#         screenspace_points.retain_grad()
#     except:
#         pass

#     mode = viewpoint_camera.mode
#     if mode == 0:
#         tanfovx = 1.0; tanfovy = 1.0
#     elif mode == 1:
#         tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
#         tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)
#     else:
#         raise ValueError("Unsupported mode!")

#     fracs = pc.get_fracs                  # (N, C)
#     rhos  = pc.get_rhos                   # (N, 1)

#     # -------------------- (A) per-basis selection (optional) --------------------
#     if basis_mask is not None:
#         if not torch.is_tensor(basis_mask):
#             basis_mask = torch.tensor(basis_mask, device=fracs.device, dtype=fracs.dtype)
#         basis_mask = basis_mask.to(device=fracs.device, dtype=fracs.dtype)

#         # normalise target to a probability vector
#         target = basis_mask / (basis_mask.sum() + 1e-8)
#         target = target.view(1, -1)  # (1, C)

#         fracs_norm = fracs / (fracs.sum(dim=1, keepdim=True) + 1e-8)

#         if soft_mode == "none":
#             fracs = fracs * target                     # legacy: direct multiply
#         elif soft_mode == "cosine":
#             sim = torch.nn.functional.cosine_similarity(fracs_norm, target, dim=1).clamp(min=0.0)
#             fracs = fracs * sim[:, None]
#         elif soft_mode == "l2":
#             d2 = ((fracs_norm - target) ** 2).sum(dim=1)
#             w = torch.exp(-d2 / (2 * sigma**2))
#             fracs = fracs * w[:, None]
#         elif soft_mode == "projection":
#             coeff = (fracs * target).sum(dim=1, keepdim=True)
#             fracs = coeff * target
#         else:
#             raise ValueError(f"Unknown soft_mode {soft_mode}")

#     # -------------------- (B) per-Gaussian selection (NEW) --------------------
#     if gaussian_mask is not None:
#         if not torch.is_tensor(gaussian_mask):
#             gaussian_mask = torch.tensor(gaussian_mask, device=rhos.device, dtype=rhos.dtype)
#         gaussian_mask = gaussian_mask.to(device=rhos.device, dtype=rhos.dtype)
#         if gaussian_mask.ndim == 1:
#             gaussian_mask = gaussian_mask[:, None]     # (N,1)
#         # Gate the total density — preserves learned fractions for selected Gaussians
#         rhos = rhos * gaussian_mask

#     # Compose per-basis coefficients
#     bcoeffs = rhos * fracs                              # (N, C)

#     raster_settings_ct = GaussianRasterizationSettings_ct(
#         num_channels=bcoeffs.shape[1],
#         image_height=int(viewpoint_camera.image_height),
#         image_width=int(viewpoint_camera.image_width),
#         tanfovx=tanfovx,
#         tanfovy=tanfovy,
#         scale_modifier=scaling_modifier,
#         viewmatrix=viewpoint_camera.world_view_transform,
#         projmatrix=viewpoint_camera.full_proj_transform,
#         campos=viewpoint_camera.camera_center,
#         prefiltered=False,
#         mode=viewpoint_camera.mode,
#         debug=pipe.debug,
#     )
#     rasterizer_ct = GaussianRasterizer_ct(raster_settings=raster_settings_ct)

#     means3D = pc.get_xyz
#     means2D = screenspace_points

#     scales = rotations = cov3D_precomp = None
#     if pipe.compute_cov3D_python:
#         cov3D_precomp = pc.get_covariance(scaling_modifier)
#     else:
#         scales = pc.get_scaling
#         rotations = pc.get_rotation

#     rendered_bcoeffs, radii = rasterizer_ct(
#         means3D=means3D,
#         means2D=means2D,
#         bcoeffs=bcoeffs,
#         scales=scales,
#         rotations=rotations,
#         cov3D_precomp=cov3D_precomp,
#     )

#     return {
#         "bcoeffs": rendered_bcoeffs,
#         "viewspace_points": screenspace_points,
#         "visibility_filter": radii > 0,
#         "radii": radii,
#     }
