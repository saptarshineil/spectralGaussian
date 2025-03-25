# Copyright (C) 2023, Gaussian-Grouping
# Gaussian-Grouping research group, https://github.com/lkeab/gaussian-grouping
# All rights reserved.
#
# ------------------------------------------------------------------------
# Modified from codes in Gaussian-Splatting 
# GRAPHDECO research group, https://team.inria.fr/graphdeco

import torch
import numpy as np
from utils.general_utils import inverse_sigmoid, get_expon_lr_func, build_rotation
from torch import nn
import os
from utils.system_utils import mkdir_p
from plyfile import PlyData, PlyElement
from utils.sh_utils import RGB2SH
from simple_knn._C import distCUDA2
from utils.graphics_utils import BasicPointCloud
from utils.general_utils import strip_symmetric, build_scaling_rotation, get_minimum_axis, flip_align_view
# from utils.general_utils import strip_symmetric, get_minimum_axis, flip_align_view
from scipy.spatial import KDTree
from scene.NVDIFFREC import create_trainable_env_rnd
import open3d as o3d
# from utils.gaussian_utils import (
#     inverse_sigmoid,
#     get_expon_lr_func,
#     build_rotation,
#     inverse_softplus,
#     strip_symmetric,
#     build_scaling_rotation,
# )


class GaussianModel:

    def setup_functions(self):
        def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
            L = build_scaling_rotation(scaling_modifier * scaling, rotation)
            actual_covariance = L @ L.transpose(1, 2)
            symm = strip_symmetric(actual_covariance)
            return symm
        
        # -- R2-Gaussian --
        # if self.scale_bound is not None:
        #     scale_min_bound, scale_max_bound = self.scale_bound
        #     assert (
        #         scale_min_bound < scale_max_bound
        #     ), "scale_min must be smaller than scale_max."
        #     self.scaling_activation = (
        #         lambda x: torch.sigmoid(x) * (scale_max_bound - scale_min_bound)
        #         + scale_min_bound
        #     )
        #     self.scaling_inverse_activation = lambda x: inverse_sigmoid(
        #         torch.relu((x - scale_min_bound) / (scale_max_bound - scale_min_bound))
        #     )
        # else:
        #     self.scaling_activation = torch.exp
        #     self.scaling_inverse_activation = torch.log

        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log

        self.covariance_activation = build_covariance_from_scaling_rotation

        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid
        # -- R2-Gaussian --
        # self.opacity_activation = torch.nn.Softplus()  # use softplus for [0, +inf]
        # self.inverse_opacity_activation = inverse_softplus

        self.rotation_activation = torch.nn.functional.normalize

    def __init__(self, sh_degree : int, brdf_dim : int, brdf_mode : str, brdf_envmap_res: int, n_spectra : int, scale_bound=None):

        if (brdf_dim>=0 and sh_degree>=0) or (brdf_dim<0 and sh_degree<0):
            raise Exception('Please provide exactly one of either brdf_dim or sh_degree!')
        self.brdf = brdf_dim>=0

        self.active_sh_degree = 0
        self.max_sh_degree = sh_degree  

        # -- General attributes --
        self._xyz = torch.empty(0) # world coordinate
        self._scaling = torch.empty(0) # 3d scale
        self._rotation = torch.empty(0) # rotation expressed in quaternions
        self._opacity = torch.empty(0) # opacity
        self.max_radii2D = torch.empty(0)
        self.xyz_gradient_accum = torch.empty(0)
        self.denom = torch.empty(0)
        self.optimizer = None
        self.spatial_lr_scale = 0
        
        # -- Spectral attributes --
        self.percent_dense = 0
        self.num_spectral_bands = n_spectra
        self.active_spectra = 0
        # segmentation attributes
        self.num_objects = 16
        self._objects_dc = []
        self._features_dc = []
        self._features_rest = []
        # brdf setting
        self.brdf_dim = brdf_dim  
        self.brdf_mode = brdf_mode  
        self.brdf_envmap_res = brdf_envmap_res  
        # brdf attributes
        self._normal = []
        self._normal2 = []
        self._specular = []
        self._roughness = []
        self.spectral_ids = list(range(0, self.num_spectral_bands + 1))
        if self.brdf:
            self.create_brdf_mlps()
        else:
            self.brdf_mlp = None    
        self.diffuse_activation = torch.sigmoid
        self.specular_activation = torch.sigmoid
        self.default_roughness = 0.0
        self.roughness_activation = torch.sigmoid
        self.roughness_bias = 0.
        self.default_roughness = 0.6

        # -- CT attributes --
        self.scale_bound = scale_bound

        self.setup_functions()

    def set_requires_grad(self, tensor, requires_grad):
            """Returns a new tensor with the specified requires_grad setting."""
            return tensor.detach().clone().requires_grad_(requires_grad)

    def capture(self):
        return (
            self._xyz,
            self._scaling,
            self._rotation,
            self._opacity,
            self.max_radii2D,
            self.xyz_gradient_accum,
            self.denom,
            self.optimizer.state_dict(),
            self.spatial_lr_scale,
            self.active_sh_degree, # spectral
            self._features_dc,     # spectral
            self._features_rest,   # spectral
            self._objects_dc,      # spectral
            self.scale_bound,      # CT
        )
    
    # Spectral 
    def create_brdf_mlps(self):
        self.brdf_mlp = []
        for spectral_id in self.spectral_ids:
            self.brdf_mlp.append(create_trainable_env_rnd(self.brdf_envmap_res, scale=0.0, bias=0.8))

    # Spectral    
    def get_normal(self, dir_pp_normalized=None, return_delta=False):
        normal_axis = self.get_minimum_axis
        normal_axis = normal_axis
        normal_axis, positive = flip_align_view(normal_axis, dir_pp_normalized)
        delta_normal1 = self._normal[self.active_spectra]  # (N, 3) 
        delta_normal2 = self._normal2[self.active_spectra] # (N, 3) 
        delta_normal = torch.stack([delta_normal1, delta_normal2], dim=-1) # (N, 3, 2)
        idx = torch.where(positive, 0, 1).long()[:,None,:].repeat(1, 3, 1) # (N, 3, 1)
        delta_normal = torch.gather(delta_normal, index=idx, dim=-1).squeeze(-1) # (N, 3)
        normal = delta_normal + normal_axis 
        normal = normal/normal.norm(dim=1, keepdim=True) # (N, 3)
        if return_delta:
            return normal, delta_normal
        else:
            return normal
    
    # Spectral
    def get_diffuse(self):
        return self._features_dc[self.active_spectra]
    
    # Spectral
    def get_specular(self):
        return self.specular_activation(self._specular[self.active_spectra])
    
    # Spectral
    def get_roughness(self):
        return self.roughness_activation(self._roughness[self.active_spectra] + self.roughness_bias)

    # Spectral
    def get_brdf_features(self):
        return self._features_rest[self.active_spectra]
    
    # Spectral
    def average_params_full_spectra(self):
        self.set_grad_params(self.active_spectra, False, False)
        self._normal[0] = torch.stack(self._normal[1:]).mean(dim=0)
        self._normal2[0] = torch.stack(self._normal2[1:]).mean(dim=0)
        # self._features_dc[0] = torch.stack(self._features_dc[1:]).mean(dim=0)
        # self._specular[0] = torch.stack(self._specular[1:]).mean(dim=0)
        self._roughness[0] = torch.stack(self._roughness[1:]).mean(dim=0)

    # Spectral
    @property
    def get_minimum_axis(self):
        return get_minimum_axis(self.get_scaling, self.get_rotation)
    
    # Spectral
    def get_features(self):
        features_dc = self._features_dc[self.active_spectra]
        features_rest = self._features_rest[self.active_spectra]
        return torch.cat((features_dc, features_rest), dim=1)
   
    # Spectral
    def get_objects(self):
        return self._objects_dc[self.active_spectra]
    
    # Spectral
    def oneupSHdegree(self):
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1

    # Spectral
    def set_grad_params(self, spectral_id, state, normal_state):
        self.set_requires_grad(self.brdf_mlp[spectral_id].base, state)
        self.set_requires_grad(self.brdf_mlp[spectral_id].env_base, state)
        self.set_requires_grad(self._normal[spectral_id], normal_state)
        self.set_requires_grad(self._normal2[spectral_id], normal_state)
        self.set_requires_grad(self._roughness[spectral_id], state)
        self.set_requires_grad(self._specular[spectral_id], state)
        self.set_requires_grad(self._features_dc[spectral_id], state)
        self.set_requires_grad(self._features_rest[spectral_id], state)
        self.set_requires_grad(self._objects_dc[spectral_id], state)

    # Spectral
    def activate_spectra(self, active_spectral_id, set_grad_normals):
        if self.brdf:
            self.active_spectra =  active_spectral_id
            self.set_grad_params(self.active_spectra, True, set_grad_normals)
            if self.brdf_mode=="envmap":
                    self.brdf_mlp[self.active_spectra].build_mips()
            for spectral_id in list(self.spectral_ids):
                if spectral_id != self.active_spectra:
                    self.set_grad_params(spectral_id, False, False)

    @property
    def get_scaling(self):
        return self.scaling_activation(self._scaling)
    
    @property
    def get_rotation(self):
        return self.rotation_activation(self._rotation)
    
    @property
    def get_xyz(self):
        return self._xyz
    
    @property
    def get_opacity(self):
        return self.opacity_activation(self._opacity)
    
    def get_covariance(self, scaling_modifier = 1):
        return self.covariance_activation(self.get_scaling, scaling_modifier, self._rotation)
    
    # ------------------ Initialization (loading / storing) ------------------
    def load_ply(self, path):
        plydata = PlyData.read(path)

        xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])),  axis=1)
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

        if not self.brdf:
            features_dc = np.zeros((xyz.shape[0], 3, 1))
            features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
            features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
            features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])
        else:
            features_dc = np.zeros((xyz.shape[0], 3))
            features_dc[:, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
            features_dc[:, 1] = np.asarray(plydata.elements[0]["f_dc_1"])
            features_dc[:, 2] = np.asarray(plydata.elements[0]["f_dc_2"])
                # features_dc[:, index + 1] = np.asarray(plydata.elements[0]["f_dc_" + str(index)])
                # features_dc[:, index + 2] = np.asarray(plydata.elements[0]["f_dc_" + str(index)])

        extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
        extra_f_names = sorted(extra_f_names, key = lambda x: int(x.split('_')[-1]))
        if not self.brdf:
            if not self.brdf:
                assert len(extra_f_names)==3*(self.max_sh_degree + 1) ** 2 - 3
            features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
            for idx, attr_name in enumerate(extra_f_names):
                features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
            # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
            features_extra = features_extra.reshape((features_extra.shape[0], 3, (self.max_sh_degree + 1) ** 2 - 1))
        elif self.brdf_mode=="envmap" and self.brdf_dim>0:
            features_extra = np.zeros((xyz.shape[0], 3*(self.brdf_dim + 1) ** 2 ))
            if len(extra_f_names)==3*(self.brdf_dim + 1) ** 2:
                for idx, attr_name in enumerate(extra_f_names):
                    features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
                features_extra = features_extra.reshape((features_extra.shape[0], (self.brdf_dim + 1) ** 2, 3))
                features_extra = features_extra.swapaxes(1,2)
            else:
                print(f"NO INITIAL SH FEATURES FOUND!!! USE ZERO SH AS INITIALIZE.")
                features_extra = features_extra.reshape((features_extra.shape[0], 3, (self.brdf_dim + 1) ** 2))
        else:
            assert len(extra_f_names)==self.brdf_dim
            features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
            for idx, attr_name in enumerate(extra_f_names):
                features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])

        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scale_names = sorted(scale_names, key = lambda x: int(x.split('_')[-1]))
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
        rot_names = sorted(rot_names, key = lambda x: int(x.split('_')[-1]))
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])
        
        if self.brdf:
            roughness = np.asarray(plydata.elements[0]["roughness"])[..., np.newaxis]

            specular_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("specular")]
            specular = np.zeros((xyz.shape[0], len(specular_names)))
            for idx, attr_name in enumerate(specular_names):
                specular[:, idx] = np.asarray(plydata.elements[0][attr_name])

            normal = np.stack((np.asarray(plydata.elements[0]["nx"]),
                            np.asarray(plydata.elements[0]["ny"]),
                            np.asarray(plydata.elements[0]["nz"])),  axis=1)
            normal2 = np.stack((np.asarray(plydata.elements[0]["nx2"]),
                            np.asarray(plydata.elements[0]["ny2"]),
                            np.asarray(plydata.elements[0]["nz2"])),  axis=1)

        objects_dc = np.zeros((xyz.shape[0], self.num_objects, 1))
        for idx in range(self.num_objects):
            objects_dc[:,idx,0] = np.asarray(plydata.elements[0]["obj_dc_"+str(idx)])

        self._xyz = nn.Parameter(torch.tensor(xyz, dtype=torch.float, device="cuda").requires_grad_(True))
        self._features_dc.append(nn.Parameter(torch.tensor(features_dc, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True)) if not self.brdf else nn.Parameter(torch.tensor(features_dc, dtype=torch.float, device="cuda").requires_grad_(True)))
        self._features_rest.append(nn.Parameter(torch.tensor(features_extra, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True)) if not ((self.brdf and self.brdf_mode=="envmap")) else nn.Parameter(torch.tensor(features_extra, dtype=torch.float, device="cuda").requires_grad_(True)))
        self._opacity = nn.Parameter(torch.tensor(opacities, dtype=torch.float, device="cuda").requires_grad_(True))
        self._scaling = nn.Parameter(torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(True))
        self._rotation = nn.Parameter(torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(True))
        self._objects_dc.append(nn.Parameter(torch.tensor(objects_dc, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True)))
        if self.brdf:
            self._roughness.append(nn.Parameter(torch.tensor(roughness, dtype=torch.float, device="cuda").requires_grad_(True)))
            self._specular.append(nn.Parameter(torch.tensor(specular, dtype=torch.float, device="cuda").requires_grad_(True)))
            self._normal.append(nn.Parameter(torch.tensor(normal, dtype=torch.float, device="cuda").requires_grad_(True)))
            self._normal2.append(nn.Parameter(torch.tensor(normal2, dtype=torch.float, device="cuda").requires_grad_(True)))

        self.active_sh_degree = self.max_sh_degree

    def save_ply(self, path, viewer_fmt=False, spectral_id = 0):
        mkdir_p(os.path.dirname(path))

        xyz = self._xyz.detach().cpu().numpy()
        normals = np.zeros_like(xyz) if not self.brdf else self._normal[spectral_id].detach().cpu().numpy()
        normals2 = self._normal2[spectral_id].detach().cpu().numpy() if (self.brdf) else np.zeros_like(xyz)
        roughness = None if not self.brdf else self._roughness[spectral_id].detach().cpu().numpy()
        specular = None if not self.brdf else self._specular[spectral_id].detach().cpu().numpy()
        f_dc = self._features_dc[spectral_id].detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy() if not self.brdf else self._features_dc[spectral_id].detach().cpu().numpy()
        f_rest = self._features_rest[spectral_id].detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy() if not ((self.brdf and self.brdf_mode=="envmap" and self.brdf_dim==0)) else self._features_rest[spectral_id].detach().cpu().numpy()
        opacities = self._opacity.detach().cpu().numpy()
        scale = self._scaling.detach().cpu().numpy()
        rotation = self._rotation.detach().cpu().numpy()
        obj_dc = self._objects_dc[spectral_id].detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()

        # if viewer_fmt:
        #     f_dc = 0.5 + (0.5*normals)
        #     f_rest = np.zeros((f_rest.shape[0], 45))
        #     normals = np.zeros_like(normals)

        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes(spectral_id=spectral_id)]

        elements = np.empty(xyz.shape[0], dtype=dtype_full)

        if self.brdf and not viewer_fmt:
            attributes = np.concatenate((xyz, normals, normals2, f_dc, f_rest, opacities, scale, rotation, obj_dc, roughness, specular), axis=1)
        else:
            attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scale, rotation, obj_dc), axis=1)
        
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)

        pcd_o3d = o3d.geometry.PointCloud()
        pcd_o3d.points = o3d.utility.Vector3dVector(xyz)
        start_index = 0
        pcd_o3d.colors = o3d.utility.Vector3dVector(f_dc[:,start_index:(start_index + 3)])
        return pcd_o3d
    
    def construct_list_of_attributes(self, viewer_fmt=False, spectral_id = 0):
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        # All channels except the 3 DC
        if not self.brdf:
            for i in range(self._features_dc[spectral_id].shape[1]*self._features_dc[spectral_id].shape[2]):
                l.append('f_dc_{}'.format(i))
            for i in range(self._features_rest[spectral_id].shape[1]*self._features_rest[spectral_id].shape[2]):
                l.append('f_rest_{}'.format(i))
        else:
            l.extend(['nx2', 'ny2', 'nz2'])
            for i in range(self._features_dc[spectral_id].shape[1]):
                l.append('f_dc_{}'.format(i))
            if viewer_fmt:
                features_rest_len = 45
            elif (self.brdf_mode=="envmap" and self.brdf_dim==0):
                features_rest_len = self._features_rest[spectral_id].shape[1]
            elif self.brdf_mode=="envmap":
                features_rest_len = self._features_rest[spectral_id].shape[1]*self._features_rest[spectral_id].shape[2]
            for i in range(features_rest_len):
                l.append('f_rest_{}'.format(i))
        l.append('opacity')
        for i in range(self._scaling.shape[1]):
            l.append('scale_{}'.format(i))
        for i in range(self._rotation.shape[1]):
            l.append('rot_{}'.format(i))
        for i in range(self._objects_dc[spectral_id].shape[1]*self._objects_dc[spectral_id].shape[2]):
            l.append('obj_dc_{}'.format(i))
        if not viewer_fmt and self.brdf:
            l.append('roughness')
            for i in range(self._specular[spectral_id].shape[1]):
                l.append('specular{}'.format(i))
        return l

    def create_from_pcd(self, pcd : BasicPointCloud, spatial_lr_scale : float):
        self.spatial_lr_scale = spatial_lr_scale
        fused_point_cloud = torch.tensor(np.asarray(pcd.points)).float().cuda()
        if not self.brdf:
            fused_color = RGB2SH(torch.tensor(np.asarray(pcd.colors)).float().cuda())
            features = torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().cuda()
            features[:, :3, 0 ] = fused_color
            features[:, 3:, 1:] = 0.0
        elif (self.brdf_mode=="envmap" and self.brdf_dim==0):
            fused_color = torch.tensor(np.asarray(pcd.colors)).float().cuda()
            features = torch.zeros((fused_color.shape[0], self.brdf_dim + 3)).float().cuda()
            features[:, :3] = 0.0
            features[:, 3: ] = 0.0
        elif self.brdf_mode=="envmap" and self.brdf_dim>0:
            fused_color = torch.tensor(np.asarray(pcd.colors)).float().cuda()
            features = torch.zeros((fused_color.shape[0], 3)).float().cuda()
            features[:, :3 ] = fused_color
            features[:, 3: ] = 0.0
            features_rest = torch.zeros((fused_color.shape[0], 3, (self.brdf_dim + 1) ** 2)).float().cuda()
        else:
            raise NotImplementedError

        # random init obj_id now
        fused_objects = RGB2SH(torch.rand((fused_point_cloud.shape[0],self.num_objects), device="cuda"))
        fused_objects = fused_objects[:,:,None]

        print("Number of points at initialisation : ", fused_point_cloud.shape[0])

        dist2 = torch.clamp_min(distCUDA2(torch.from_numpy(np.asarray(pcd.points)).float().cuda()), 0.0000001)
        scales = torch.log(torch.sqrt(dist2))[...,None].repeat(1, 3)
        rots = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
        rots[:, 0] = 1

        opacities = inverse_sigmoid(0.1 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"))

        self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
        
        for spectral_id in self.spectral_ids:
            #parameters per spectra
            if not self.brdf:
                self._features_dc.append(nn.Parameter(features[:,:,0:1].transpose(1, 2).contiguous().requires_grad_(True)))
                self._features_rest.append(nn.Parameter(features[:,:,1:].transpose(1, 2).contiguous().requires_grad_(True)))
            else:
                self._features_dc.append(nn.Parameter(features[:,:3].contiguous().requires_grad_(True)))
                if (self.brdf_mode=="envmap" and self.brdf_dim==0):
                    self._features_rest.append(nn.Parameter(features[:,3:].contiguous().requires_grad_(True)))
                elif self.brdf_mode=="envmap":
                    self._features_rest.append(nn.Parameter(features_rest.contiguous().requires_grad_(True)))

                normals =np.zeros_like(np.asarray(pcd.points, dtype=np.float32))
                normals2 = np.copy(normals)
                self._normal.append(nn.Parameter(torch.from_numpy(normals).to(self._xyz.device).requires_grad_(True)))
                specular_len = 3 
                self._specular.append(nn.Parameter(torch.zeros((fused_point_cloud.shape[0], specular_len), device="cuda").requires_grad_(True)))
                self._roughness.append(nn.Parameter(self.default_roughness*torch.ones((fused_point_cloud.shape[0], 1), device="cuda").requires_grad_(True)))
                self._normal2.append(nn.Parameter(torch.from_numpy(normals2).to(self._xyz.device).requires_grad_(True)))
            self._objects_dc.append(nn.Parameter(fused_objects.transpose(1, 2).contiguous().requires_grad_(True)))
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(True))
        self._opacity = nn.Parameter(opacities.requires_grad_(True))
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

    # ------------------ Training / LR ------------------  
    def training_setup(self, training_args):
        self.fix_brdf_lr = training_args.fix_brdf_lr
        self.percent_dense = training_args.percent_dense
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        l = [
            {'params': [self._xyz], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "xyz"},       
            {'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"},
            {'params': [self._scaling], 'lr': training_args.scaling_lr, "name": "scaling"},
            {'params': [self._rotation], 'lr': training_args.rotation_lr, "name": "rotation"}
        ]
        for spectral_id in list(self.spectral_ids):
            l.extend([{'params': [self._features_dc[spectral_id]], 'lr': training_args.feature_lr, "name": "f_dc_"+ str(spectral_id)},
            {'params': [self._features_rest[spectral_id]], 'lr': training_args.feature_lr / 20.0, "name": "f_rest_" + str(spectral_id)}])
        if self.brdf:
            # self._normal.requires_grad_(requires_grad=False)
            self.brdf_mlp_scheduler_args = []
            for spectral_id in list(self.spectral_ids):
                l.extend([{'params': list(self.brdf_mlp[spectral_id].parameters()), 'lr': training_args.brdf_mlp_lr_init, "name": "brdf_mlp_" + str(spectral_id)}])
                self.brdf_mlp_scheduler_args.append(get_expon_lr_func(lr_init=training_args.brdf_mlp_lr_init,
                                        lr_final=training_args.brdf_mlp_lr_final,
                                        lr_delay_mult=training_args.brdf_mlp_lr_delay_mult,
                                        max_steps=training_args.brdf_mlp_lr_max_steps))
                l.extend([  
                    {'params': [self._roughness[spectral_id]], 'lr': training_args.roughness_lr, "name": "roughness_" + str(spectral_id)},
                    {'params': [self._specular[spectral_id]], 'lr': training_args.specular_lr, "name": "specular_" + str(spectral_id)},
                    {'params': [self._normal[spectral_id]], 'lr': training_args.normal_lr, "name": "normal_" + str(spectral_id)},
                    {'params': [self._objects_dc[spectral_id]], 'lr': training_args.feature_lr, "name": "obj_dc_"+str(spectral_id)},
                ])
                # self._normal2.requires_grad_(requires_grad=False)
                l.extend([
                    {'params': [self._normal2[spectral_id]], 'lr': training_args.normal_lr, "name": "normal2_" + str(spectral_id)},
                ])

        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        self.xyz_scheduler_args = get_expon_lr_func(lr_init=training_args.position_lr_init*self.spatial_lr_scale,
                                                    lr_final=training_args.position_lr_final*self.spatial_lr_scale,
                                                    lr_delay_mult=training_args.position_lr_delay_mult,
                                                    max_steps=training_args.position_lr_max_steps)

    def _update_learning_rate(self, iteration, param):
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == param:
                try:
                    if param.endswith(str(self.active_spectra)):
                        spectral_id = int(param[-1])
                        param_new = param[:-2]
                        lr = getattr(self, f"{param_new}_scheduler_args[{spectral_id}]")(iteration)
                    else:    
                        lr = getattr(self, f"{param}_scheduler_args")(iteration)
                    param_group['lr'] = lr
                    return lr
                except AttributeError:
                    pass

    def update_learning_rate(self, iteration):
        ''' Learning rate scheduling per step '''
        self._update_learning_rate(iteration, "xyz")
        if self.brdf and not self.fix_brdf_lr:
            parameter_list = []
            for spectral_id in self.spectral_ids:
                parameter_list.extend(["brdf_mlp_" + str(spectral_id),"roughness_"+str(spectral_id),"specular_"+str(spectral_id),"normal_"+str(spectral_id),"f_dc_"+str(spectral_id), "f_rest_"+str(spectral_id)])
            for param in parameter_list:
                lr = self._update_learning_rate(iteration, param)

    def reset_opacity(self):
        opacities_new = inverse_sigmoid(torch.min(self.get_opacity, torch.ones_like(self.get_opacity)*0.01))
        optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, "opacity")
        self._opacity = optimizable_tensors["opacity"]

    def replace_tensor_to_optimizer(self, tensor, name):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"].startswith("brdf_mlp"):
                continue
            # elif group["name"][-1].isdigit():
            #     key = group["name"]
            #     if int(key[key.rindex("_") + 1:]) != spectral_id:
            #         continue
            if group["name"] == name:
                stored_state = self.optimizer.state.get(group['params'][0], None)
                stored_state["exp_avg"] = torch.zeros_like(tensor)
                stored_state["exp_avg_sq"] = torch.zeros_like(tensor)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def cat_tensors_to_optimizer(self, tensors_dict, spectral_id):
        optimizable_tensors = {} 
        for group in self.optimizer.param_groups:
            if group["name"].startswith("brdf_mlp"):
                continue
            elif group["name"][-1].isdigit():
                key = group["name"]
                if int(key[key.rindex("_") + 1:]) != spectral_id:
                    continue
            elif not group["name"][-1].isdigit() and spectral_id > -1:
                continue
            assert len(group["params"]) == 1
            extension_tensor = tensors_dict[group["name"]]
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:

                stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=0)
                stored_state["exp_avg_sq"] = torch.cat((stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)), dim=0)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]

        return optimizable_tensors
    
    # ------------------ Densification / Pruning ------------------
    def densify_and_prune(self, max_grad, min_opacity, extent, max_screen_size, only_active_spectra=False):
        grads = self.xyz_gradient_accum / self.denom
        grads[grads.isnan()] = 0.0

        self.densify_and_clone(grads, max_grad, extent, only_active_spectra=only_active_spectra)
        self.densify_and_split(grads, max_grad, extent, only_active_spectra=only_active_spectra)

        prune_mask = (self.get_opacity < min_opacity).squeeze()
        if max_screen_size:
            big_points_vs = self.max_radii2D > max_screen_size
            big_points_ws = self.get_scaling.max(dim=1).values > 0.1 * extent
            prune_mask = torch.logical_or(torch.logical_or(prune_mask, big_points_vs), big_points_ws)
        self.prune_points(prune_mask, only_active_spectra=only_active_spectra)

        torch.cuda.empty_cache()

    def densify_and_clone(self, grads, grad_threshold, scene_extent, only_active_spectra):
        # Extract points that satisfy the gradient condition
        selected_pts_mask = torch.where(torch.norm(grads, dim=-1) >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values <= self.percent_dense*scene_extent)
        
        new_xyz = self._xyz[selected_pts_mask]
        new_opacities = self._opacity[selected_pts_mask]
        new_scaling = self._scaling[selected_pts_mask]
        new_rotation = self._rotation[selected_pts_mask]
        self.densification_postfix(new_xyz,new_opacities, new_scaling, new_rotation)
        ids = [self.active_spectra] if only_active_spectra else self.spectral_ids
        for spectral_id in ids:
            new_features_dc = self._features_dc[spectral_id][selected_pts_mask]
            new_features_rest = self._features_rest[spectral_id][selected_pts_mask]
            new_objects_dc = self._objects_dc[spectral_id][selected_pts_mask]
            new_roughness = self._roughness[spectral_id][selected_pts_mask] if self.brdf else None
            new_specular = self._specular[spectral_id][selected_pts_mask] if self.brdf else None
            new_normal = self._normal[spectral_id][selected_pts_mask] if self.brdf else None
            new_normal2 = self._normal2[spectral_id][selected_pts_mask] if (self.brdf) else None
            self.densification_postfix_per_spectra(new_features_dc, new_features_rest, new_objects_dc,  new_roughness, new_specular, new_normal, new_normal2, spectral_id)

    def densify_and_split(self, grads, grad_threshold, scene_extent, only_active_spectra, N=2):
        n_init_points = self.get_xyz.shape[0]
        # Extract points that satisfy the gradient condition
        padded_grad = torch.zeros((n_init_points), device="cuda")
        padded_grad[:grads.shape[0]] = grads.squeeze()
        selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values > self.percent_dense*scene_extent)

        stds = self.get_scaling[selected_pts_mask].repeat(N,1)
        means =torch.zeros((stds.size(0), 3),device="cuda")
        samples = torch.normal(mean=means, std=stds)
        rots = build_rotation(self._rotation[selected_pts_mask]).repeat(N,1,1)
        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[selected_pts_mask].repeat(N, 1)
        new_scaling = self.scaling_inverse_activation(self.get_scaling[selected_pts_mask].repeat(N,1) / (0.8*N))
        new_rotation = self._rotation[selected_pts_mask].repeat(N,1)
        new_opacity = self._opacity[selected_pts_mask].repeat(N,1)
        self.densification_postfix(new_xyz,new_opacity, new_scaling, new_rotation)
        ids = [self.active_spectra] if only_active_spectra else self.spectral_ids
        for spectral_id in ids:
            new_features_dc = self._features_dc[spectral_id][selected_pts_mask].repeat(N,1,1) if not self.brdf else self._features_dc[spectral_id][selected_pts_mask].repeat(N,1)
            new_features_rest = self._features_rest[spectral_id][selected_pts_mask].repeat(N,1,1) if not ((self.brdf and self.brdf_mode=="envmap" and self.brdf_dim==0)) else self._features_rest[spectral_id][selected_pts_mask].repeat(N,1)
            new_objects_dc = self._objects_dc[spectral_id][selected_pts_mask].repeat(N,1,1)
            new_roughness = self._roughness[spectral_id][selected_pts_mask].repeat(N,1) if self.brdf else None
            new_specular = self._specular[spectral_id][selected_pts_mask].repeat(N,1) if self.brdf else None
            new_normal = self._normal[spectral_id][selected_pts_mask].repeat(N,1) if self.brdf else None
            new_normal2 = self._normal2[spectral_id][selected_pts_mask].repeat(N,1) if (self.brdf) else None
            self.densification_postfix_per_spectra(new_features_dc, new_features_rest, new_objects_dc, new_roughness, new_specular, new_normal, new_normal2, spectral_id)

        prune_filter = torch.cat((selected_pts_mask, torch.zeros(N * selected_pts_mask.sum(), device="cuda", dtype=bool)))
        self.prune_points(prune_filter, only_active_spectra=only_active_spectra)
    
    def densification_postfix_per_spectra(self, new_features_dc, new_features_rest,new_objects_dc,  new_roughness, new_specular, new_normal, new_normal2, spectral_id):
        d = {}
        d.update({"f_dc_"+ str(spectral_id): new_features_dc,"f_rest_"+ str(spectral_id): new_features_rest,"obj_dc_"+ str(spectral_id): new_objects_dc})
        if self.brdf:
            d.update({
                "roughness_"+ str(spectral_id): new_roughness,
                "specular_"+ str(spectral_id) : new_specular,
                "normal_"+ str(spectral_id) : new_normal,
                "normal2_"+ str(spectral_id) : new_normal2,
            })

        optimizable_tensors = self.cat_tensors_to_optimizer(d, spectral_id)
        self._features_dc[spectral_id] = optimizable_tensors["f_dc_"+ str(spectral_id)]
        self._features_rest[spectral_id] = optimizable_tensors["f_rest_"+ str(spectral_id)]

        self._objects_dc[spectral_id] = optimizable_tensors["obj_dc_"+ str(spectral_id)]
        if self.brdf:
            self._roughness[spectral_id] = optimizable_tensors["roughness_"+ str(spectral_id)]
            self._specular[spectral_id] = optimizable_tensors["specular_"+ str(spectral_id)]
            self._normal[spectral_id] = optimizable_tensors["normal_"+ str(spectral_id)]
            self._normal2[spectral_id] = optimizable_tensors["normal2_"+ str(spectral_id)]

    def densification_postfix(self, new_xyz, new_opacities, new_scaling, new_rotation):
        d = {"xyz": new_xyz,
        "opacity": new_opacities,
        "scaling" : new_scaling,
        "rotation" : new_rotation,
        }
        optimizable_tensors = self.cat_tensors_to_optimizer(d, -1)
        self._xyz = optimizable_tensors["xyz"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

    def prune_points(self, mask, only_active_spectra):
        valid_points_mask = ~mask
        optimizable_tensors = self._prune_optimizer(valid_points_mask)

        self._xyz = optimizable_tensors["xyz"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]
        ids = [self.active_spectra] if only_active_spectra else self.spectral_ids
        for spectral_id in ids:
            self._features_dc[spectral_id] = optimizable_tensors["f_dc_"+str(spectral_id)]
            self._features_rest[spectral_id] = optimizable_tensors["f_rest_"+str(spectral_id)]
            self._objects_dc[spectral_id] = optimizable_tensors["obj_dc_"+str(spectral_id)]
            if self.brdf:
                self._roughness[spectral_id] = optimizable_tensors["roughness_"+str(spectral_id)]
                self._specular[spectral_id] = optimizable_tensors["specular_"+str(spectral_id)]
                self._normal[spectral_id] = optimizable_tensors["normal_"+str(spectral_id)]
                self._normal2[spectral_id] = optimizable_tensors["normal2_"+str(spectral_id)]

        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]

        self.denom = self.denom[valid_points_mask]
        self.max_radii2D = self.max_radii2D[valid_points_mask]

    def _prune_optimizer(self, mask):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"].startswith("brdf_mlp"):
                continue
            # elif group["name"][-1].isdigit():
            #     key = group["name"]
            #     if int(key[key.rindex("_") + 1:]) != spectral_id:
            #         continue
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter((group["params"][0][mask].requires_grad_(True)))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(group["params"][0][mask].requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def add_densification_stats(self, viewspace_point_tensor, update_filter):
        self.xyz_gradient_accum[update_filter] += torch.norm(viewspace_point_tensor.grad[update_filter,:2], dim=-1, keepdim=True)
        self.denom[update_filter] += 1


# TODO: Uncomment
    # #segmentation, inpainting and removal
    # #style transfer    
    # def finetune_setup(self, training_args, mask3d, spectral_id):
    #     # Define a function that applies the mask to the gradients
    #     def mask_hook(grad):
    #         return grad * mask3d
    #     def mask_hook2(grad):
    #         return grad * mask3d.squeeze(-1)
        

    #     # Register the hook to the parameter (only once!)
    #     hook_xyz = self._xyz.register_hook(mask_hook2)
    #     hook_dc = self._features_dc.register_hook(mask_hook)
    #     hook_rest = self._features_rest.register_hook(mask_hook)
    #     hook_opacity = self._opacity.register_hook(mask_hook2)
    #     hook_scaling = self._scaling.register_hook(mask_hook2)
    #     hook_rotation = self._rotation.register_hook(mask_hook2)

    #     self._objects_dc.requires_grad = False

    #     self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")
    #     self.percent_dense = training_args.percent_dense
    #     self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
    #     self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        
    #     l = [
    #         {'params': [self._xyz], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "xyz"},
    #         {'params': [self._features_dc[spectral_id]], 'lr': training_args.feature_lr, "name": "f_dc_"+str(spectral_id)},
    #         {'params': [self._features_rest[spectral_id]], 'lr': training_args.feature_lr / 20.0, "name": "f_rest_"+str(spectral_id)},
    #         {'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"},
    #         {'params': [self._scaling], 'lr': training_args.scaling_lr, "name": "scaling"},
    #         {'params': [self._rotation], 'lr': training_args.rotation_lr, "name": "rotation"},
    #         {'params': [self._objects_dc[spectral_id]], 'lr': training_args.feature_lr, "name": "obj_dc_"+str(spectral_id)},
    #     ]

    #     self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)

    # def removal_setup(self, training_args, mask3d, spectral_id):

    #     mask3d = ~mask3d.bool().squeeze()

    #     # Extracting subsets using the mask
    #     xyz_sub = self._xyz[mask3d].detach()
    #     features_dc_sub = self._features_dc[spectral_id][mask3d].detach()
    #     features_rest_sub = self._features_rest[spectral_id][mask3d].detach()
    #     opacity_sub = self._opacity[mask3d].detach()
    #     scaling_sub = self._scaling[mask3d].detach()
    #     rotation_sub = self._rotation[mask3d].detach()
    #     objects_dc_sub = self._objects_dc[spectral_id][mask3d].detach()
    #     normals_sub = self._normal[spectral_id][mask3d].detach()
    #     normals2_sub = self._normal2[spectral_id][mask3d].detach()
    #     roughness_sub = self._roughness[spectral_id][mask3d].detach()
    #     specular_sub = self._specular[spectral_id][mask3d].detach()
        
    #     def set_requires_grad(tensor, requires_grad):
    #         """Returns a new tensor with the specified requires_grad setting."""
    #         return tensor.detach().clone().requires_grad_(requires_grad)

    #     # Construct nn.Parameters with specified gradients
    #     self._xyz = nn.Parameter(set_requires_grad(xyz_sub, False))
    #     self._features_dc[spectral_id] = nn.Parameter(set_requires_grad(features_dc_sub, False))
    #     self._features_rest[spectral_id] = nn.Parameter(set_requires_grad(features_rest_sub, False))
    #     self._opacity = nn.Parameter(set_requires_grad(opacity_sub, False))
    #     self._scaling = nn.Parameter(set_requires_grad(scaling_sub, False))
    #     self._rotation = nn.Parameter(set_requires_grad(rotation_sub, False))
    #     self._objects_dc[spectral_id] = nn.Parameter(set_requires_grad(objects_dc_sub, False))
    #     self._normal[spectral_id] = nn.Parameter(set_requires_grad(normals_sub, False))
    #     self._normal2[spectral_id] = nn.Parameter(set_requires_grad(normals2_sub, False))
    #     self._roughness[spectral_id] = nn.Parameter(set_requires_grad(roughness_sub, False))
    #     self._specular[spectral_id] = nn.Parameter(set_requires_grad(specular_sub, False))

    # def inpaint_setup(self, training_args, mask3d, spectral_id):

    #     def initialize_new_features(features, num_new_points, mask_xyz_values, distance_threshold=0.25, max_distance_threshold=1, k=5):
    #         """Initialize new points for multiple features based on neighbouring points in the remaining area."""
    #         new_features = {}
            
    #         if num_new_points == 0:
    #             for key in features:
    #                 new_features[key] = torch.empty((0, *features[key].shape[1:]), device=features[key].device)
    #             return new_features

    #         # Get remaining points from features
    #         remaining_xyz_values = features["xyz"]
    #         remaining_xyz_values_np = remaining_xyz_values.cpu().numpy()
            
    #         # Build a KD-Tree for fast nearest-neighbor lookup
    #         kdtree = KDTree(remaining_xyz_values_np)
            
    #         # Sample random points from mask_xyz_values as query points
    #         mask_xyz_values_np = mask_xyz_values.cpu().numpy()
    #         query_points = mask_xyz_values_np

    #         # Find the k nearest neighbors in the remaining points for each query point
    #         distances, indices = kdtree.query(query_points, k=k)
    #         selected_indices = indices

    #         # Initialize new points for each feature
    #         for key, feature in features.items():
    #             # Convert feature to numpy array
    #             feature_np = feature.cpu().numpy()
                
    #             # If we have valid neighbors, calculate the mean of neighbor points
    #             if feature_np.ndim == 2:
    #                 neighbor_points = feature_np[selected_indices]
    #             elif feature_np.ndim == 3:
    #                 neighbor_points = feature_np[selected_indices, :, :]
    #             else:
    #                 raise ValueError(f"Unsupported feature dimension: {feature_np.ndim}")
    #             new_points_np = np.mean(neighbor_points, axis=1)
                
    #             # Convert back to tensor
    #             new_features[key] = torch.tensor(new_points_np, device=feature.device, dtype=feature.dtype)
            
    #         return new_features['xyz'], new_features['features_dc_'+str(spectral_id)], new_features['scaling'], new_features['objects_dc_'+str(spectral_id)], new_features['features_rest_'+str(spectral_id)], new_features['opacity'], new_features['rotation'],new_features['roughness_'+str(spectral_id)],new_features['specular_'+str(spectral_id)],new_features['normal_'+str(spectral_id)],new_features['normal2_'+str(spectral_id)]
        
    #     mask3d = ~mask3d.bool().squeeze()
    #     mask_xyz_values = self._xyz[~mask3d]

    #     # Extracting subsets using the mask
    #     xyz_sub = self._xyz[mask3d].detach()
    #     features_dc_sub = self._features_dc[spectral_id][mask3d].detach()
    #     features_rest_sub = self._features_rest[spectral_id][mask3d].detach()
    #     opacity_sub = self._opacity[mask3d].detach()
    #     scaling_sub = self._scaling[mask3d].detach()
    #     rotation_sub = self._rotation[mask3d].detach()
    #     objects_dc_sub = self._objects_dc[spectral_id][mask3d].detach()
    #     normals_sub = self._normal[spectral_id][mask3d].detach()
    #     normals2_sub = self._normal2[spectral_id][mask3d].detach()
    #     roughness_sub = self._roughness[spectral_id][mask3d].detach()
    #     specular_sub = self._specular[spectral_id][mask3d].detach()
        
    #     # Add new points with random initialization
    #     sub_features = {
    #         'xyz': xyz_sub,
    #         'features_dc_'+str(spectral_id): features_dc_sub,
    #         'scaling': scaling_sub,
    #         'objects_dc_'+str(spectral_id): objects_dc_sub,
    #         'features_rest_'+str(spectral_id): features_rest_sub,
    #         'opacity': opacity_sub,
    #         'rotation': rotation_sub,
    #         'roughness_'+str(spectral_id):roughness_sub,
    #         'specular_'+str(spectral_id):specular_sub,
    #         'normal_'+str(spectral_id):normals_sub,
    #         'normal2_'+str(spectral_id):normals2_sub
    #     }

    #     num_new_points = len(mask_xyz_values)
    #     with torch.no_grad():
    #         new_xyz, new_features_dc, new_scaling, new_objects_dc, new_features_rest, new_opacity, new_rotation, new_roughness, new_specular, new_normal, new_normal2 = initialize_new_features(sub_features, num_new_points, mask_xyz_values)

    #     def set_requires_grad(tensor, requires_grad):
    #         """Returns a new tensor with the specified requires_grad setting."""
    #         return tensor.detach().clone().requires_grad_(requires_grad)

    #     # Construct nn.Parameters with specified gradients
    #     self._xyz = nn.Parameter(torch.cat([set_requires_grad(xyz_sub, False), set_requires_grad(new_xyz, True)]))
    #     self._features_dc[spectral_id] = nn.Parameter(torch.cat([set_requires_grad(features_dc_sub, False), set_requires_grad(new_features_dc, True)]))
    #     self._features_rest[spectral_id] = nn.Parameter(torch.cat([set_requires_grad(features_rest_sub, False), set_requires_grad(new_features_rest, True)]))
    #     self._opacity = nn.Parameter(torch.cat([set_requires_grad(opacity_sub, False), set_requires_grad(new_opacity, True)]))
    #     self._scaling = nn.Parameter(torch.cat([set_requires_grad(scaling_sub, False), set_requires_grad(new_scaling, True)]))
    #     self._rotation = nn.Parameter(torch.cat([set_requires_grad(rotation_sub, False), set_requires_grad(new_rotation, True)]))
    #     self._objects_dc[spectral_id] = nn.Parameter(torch.cat([set_requires_grad(objects_dc_sub, False), set_requires_grad(new_objects_dc, True)]))
    #     self._roughness[spectral_id] = nn.Parameter(torch.cat([set_requires_grad(roughness_sub, False), set_requires_grad(new_roughness, True)]))
    #     self._specular[spectral_id] = nn.Parameter(torch.cat([set_requires_grad(specular_sub, False), set_requires_grad(new_specular, True)]))
    #     self._normal[spectral_id] = nn.Parameter(torch.cat([set_requires_grad(normals_sub, False), set_requires_grad(new_normal, True)]))
    #     self._normal2[spectral_id] = nn.Parameter(torch.cat([set_requires_grad(normals2_sub, False), set_requires_grad(new_normal2, True)]))

    #     # for optimize
    #     self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")
    #     self.percent_dense = training_args.percent_dense
    #     self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
    #     self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")

    #     # Setup optimizer. Only the new points will have gradients.
    #     l = [
    #         {'params': [self._xyz], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "xyz"},
    #         {'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"},
    #         {'params': [self._scaling], 'lr': training_args.scaling_lr, "name": "scaling"},
    #         {'params': [self._rotation], 'lr': training_args.rotation_lr, "name": "rotation"},
    #     ]

    #     l.extend([  {'params': [self._features_dc[spectral_id]], 'lr': training_args.feature_lr, "name": "f_dc_"+ str(spectral_id)},
    #                 {'params': [self._features_rest[spectral_id]], 'lr': training_args.feature_lr / 20.0, "name": "f_rest_" + str(spectral_id)},
    #                 {'params': [self._roughness[spectral_id]], 'lr': training_args.roughness_lr, "name": "roughness_" + str(spectral_id)},
    #                 {'params': [self._specular[spectral_id]], 'lr': training_args.specular_lr, "name": "specular_" + str(spectral_id)},
    #                 {'params': [self._normal[spectral_id]], 'lr': training_args.normal_lr, "name": "normal_" + str(spectral_id)},
    #                 {'params': [self._normal2[spectral_id]], 'lr': training_args.normal_lr, "name": "normal2_" + str(spectral_id)},
    #                 {'params': [self._objects_dc[spectral_id]], 'lr': training_args.feature_lr, "name": "obj_dc_"+str(spectral_id)},
    #             ])

    #     self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
