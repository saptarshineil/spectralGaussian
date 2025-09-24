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
from utils.gaussian_utils import inverse_softplus
#     inverse_sigmoid,
#     get_expon_lr_func,
#     build_rotation,
#     inverse_softplus,
#     strip_symmetric,
#     build_scaling_rotation,
# )

from utils.spectralCT_utils import stable_simplex


class GaussianModel:

    def setup_functions(self):
        def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
            L = build_scaling_rotation(scaling_modifier * scaling, rotation)
            actual_covariance = L @ L.transpose(1, 2)
            symm = strip_symmetric(actual_covariance)
            return symm
        
        # -- Spectral --
        # self.scaling_activation = torch.exp
        # self.scaling_inverse_activation = torch.log

        self.covariance_activation = build_covariance_from_scaling_rotation

        # self.opacity_activation = torch.sigmoid
        # self.opacity_inverse_activation = inverse_sigmoid

        # self.bcoeffs_activation = torch.softmax
        # self.bcoeffs_activation = torch.nn.functional.relu 
        # self.bcoeffs_activation = torch.sigmoid 
        # self.bcoeffs_activation = torch.ReLu()
        
        # self.bcoeffs_activation = torch.nn.Softplus()
        # self.bcoeffs_inverse_activation = inverse_softplus
        self.rhos_activation = torch.nn.Softplus()
        self.rhos_inverse_activation = inverse_softplus

        self.rotation_activation = torch.nn.functional.normalize
        
        # -- R2-Gaussian --
        if self.scale_bound is not None:
            scale_min_bound, scale_max_bound = self.scale_bound
            assert (
                scale_min_bound < scale_max_bound
            ), "scale_min must be smaller than scale_max."
            self.scaling_activation = (
                lambda x: torch.sigmoid(x) * (scale_max_bound - scale_min_bound)
                + scale_min_bound
            )
            self.scaling_inverse_activation = lambda x: inverse_sigmoid(
                torch.relu((x - scale_min_bound) / (scale_max_bound - scale_min_bound))
            )
        else:
            self.scaling_activation = torch.exp
            self.scaling_inverse_activation = torch.log

        # self.covariance_activation = build_covariance_from_scaling_rotation

        # self.opacity_activation = torch.nn.Softplus()  # use softplus for [0, +inf]
        # self.opacity_inverse_activation = inverse_softplus

        # self.rotation_activation = torch.nn.functional.normalize

    # def __init__(self, n_spectra : int, n_bcoeffs : int, E : torch.tensor, scale_bound=None):
    def __init__(self, n_spectra : int, n_bcoeffs : int, scale_bound=None):

        # -- General attributes --
        self._xyz = torch.empty(0) # world coordinate
        self._scaling = torch.empty(0) # 3d scale
        self._rotation = torch.empty(0) # rotation expressed in quaternions
        
        # self._bcoeffs = []
        # self._rhos = []
        self._rhos = torch.empty(0)
        self._fracs = []
        self._rho_frozen = False
        
        self.max_radii2D = torch.empty(0)
        self.xyz_gradient_accum = torch.empty(0)
        self.denom = torch.empty(0)
        self.optimizer = None
        self.spatial_lr_scale = 0
        
        # -- Spectral attributes --
        self.percent_dense = 0
        self.num_spectral_bands = n_spectra
        self.num_basis_coeffs = n_bcoeffs
        self.active_spectra = 0
        self.active_bcoeff = 0
        self.num_objects = 16
        self.spectral_ids = list(range(0, self.num_spectral_bands))
        self.bcoeff_ids = list(range(0, self.num_basis_coeffs))

        # -- R2-Gaussians attributes --
        self.scale_bound = scale_bound

        # Init activation functions
        self.setup_functions()

    def set_requires_grad(self, tensor, requires_grad):
            """Returns a new tensor with the specified requires_grad setting."""
            return tensor.detach().clone().requires_grad_(requires_grad)
    
    def freeze_rhos(self):
        """Keep rhos trainable (for autograd) but lr=0, so optimizer won't update."""
        self._rho_frozen = True
        if self.optimizer is not None:
            for g in self.optimizer.param_groups:
                if g["name"] == "rhos":
                    g["lr"] = 0.0

    def unfreeze_rhos(self):
        """Re-enable scheduling for rhos; lr will be set by update_learning_rate()."""
        self._rho_frozen = False

    def rhos_are_frozen(self) -> bool:
        return self._rho_frozen

    def capture(self):
        return (
            self._xyz,
            self._scaling,
            self._rotation,
            self.max_radii2D,
            self.xyz_gradient_accum,
            self.denom,
            self.optimizer.state_dict(),
            self.spatial_lr_scale,
            self.scale_bound,      # CT
        )
    
    # ------------------ General Gaussian functions ------------------
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
    def get_bcoeffs(self):
        rho = self.get_rhos
        frac = self.get_fracs
        return rho * frac
    
    @property
    def get_rhos(self):
        rho = self.rhos_activation(self._rhos)
        return rho
    
    @property
    # def get_fracs(self):
    #     raw = torch.cat(self._fracs, dim=1)
    #     return torch.softmax(raw, dim=1)
    def get_fracs(self):
        raw = torch.cat(self._fracs, dim=1)
        eps = getattr(self, "_frac_eps", 0.20)   # start with some floor
        return stable_simplex(raw, eps=eps)

    def get_covariance(self, scaling_modifier = 1):
        return self.covariance_activation(self.get_scaling, scaling_modifier, self._rotation)
    
    # ------------------ Spectral Gaussian functions ------------------
    def average_params_full_spectra(self):
        self.set_grad_params(self.active_spectra, False)

    def set_grad_params(self, basis_id: int, state: bool):
    #     self.set_requires_grad(self._bcoeffs[bcoeff_id], state)
        # self._rhos[basis_id].requires_grad_(state)
        self._rhos.requires_grad_(state)
        self._fracs[basis_id].requires_grad_(state)
    
    def activate_spectra(self, active_spectral_id):
        # self.active_spectra =  active_spectral_id
        # # self.set_grad_params(self.active_spectra, True)

        # # for spectral_id in list(self.spectral_ids):
        # for bcoeff_id in list(self.bcoeff_ids):
        #     if bcoeff_id != self.active_spectra:
        #         self.set_grad_params(bcoeff_id, False)
        return

    def activate_bcoeff(self, active_bcoeff_id):
        self.active_bcoeff =  active_bcoeff_id
        self.set_grad_params(self.active_bcoeff, True)

        for bcoeff_id in list(self.bcoeff_ids):
            if bcoeff_id != self.active_bcoeff:
                self.set_grad_params(bcoeff_id, False)

    # ------------------ Initialization (loading / storing) ------------------
    def load_ply(self, path):
        plydata = PlyData.read(path)

        xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])),  axis=1)
        self._xyz = nn.Parameter(torch.tensor(xyz, dtype=torch.float, device="cuda").requires_grad_(True))

        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scale_names = sorted(scale_names, key = lambda x: int(x.split('_')[-1]))
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])
        self._scaling = nn.Parameter(torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(True))

        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
        rot_names = sorted(rot_names, key = lambda x: int(x.split('_')[-1]))
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])
        self._rotation = nn.Parameter(torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(True))

        rhos = np.asarray(plydata.elements[0]["rhos"])[..., np.newaxis]
        self._rhos = nn.Parameter(torch.tensor(rhos, dtype=torch.float, device="cuda").requires_grad_(True))
        for bcoeff_id in self.bcoeff_ids:
            # bcoeffs = np.asarray(plydata.elements[0]["bcoeffs_" + str(bcoeff_id)])[..., np.newaxis]
            # self._bcoeffs.append(nn.Parameter(torch.tensor(bcoeffs, dtype=torch.float, device="cuda").requires_grad_(True)))
            
            # rhos = np.asarray(plydata.elements[0]["rhos_" + str(bcoeff_id)])[..., np.newaxis]
            # self._rhos.append(nn.Parameter(torch.tensor(rhos, dtype=torch.float, device="cuda").requires_grad_(True)))
            fracs = np.asarray(plydata.elements[0]["fracs_" + str(bcoeff_id)])[..., np.newaxis]
            self._fracs.append(nn.Parameter(torch.tensor(fracs, dtype=torch.float, device="cuda").requires_grad_(True)))
        

    def save_ply(self, path, viewer_fmt=False, spectral_id = 0):
        mkdir_p(os.path.dirname(path))

        xyz = self._xyz.detach().cpu().numpy()
        
        # bcoeffs = []
        # for bcoeff_id in self.bcoeff_idsp()
                # energy_opt.zero_gra:
        #     bcoeffs.append(self._bcoeffs[bcoeff_id].detach().cpu().numpy())
        # bcoeffs = np.concatenate(bcoeffs, axis=1)
        # rhos = []
        fracs = []
        for bcoeff_id in self.bcoeff_ids:
            # rhos.append(self._rhos[bcoeff_id].detach().cpu().numpy())
            fracs.append(self._fracs[bcoeff_id].detach().cpu().numpy())
        # rhos = np.concatenate(rhos, axis=1)
        fracs = np.concatenate(fracs, axis=1)
        rhos = self._rhos.detach().cpu().numpy()

        scale = self._scaling.detach().cpu().numpy()
        rotation = self._rotation.detach().cpu().numpy()

        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]

        elements = np.empty(xyz.shape[0], dtype=dtype_full)

        # attributes = np.concatenate((xyz, bcoeffs, fracs, scale, rotation), axis=1)
        attributes = np.concatenate((xyz, rhos, fracs, scale, rotation), axis=1)
        
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)

        pcd_o3d = o3d.t.geometry.PointCloud()
        pcd_o3d.point["positions"]  = o3d.core.Tensor(xyz,     dtype=o3d.core.Dtype.Float32)
        # pcd_o3d.point["bcoeffs"]    = o3d.core.Tensor(bcoeffs, dtype=o3d.core.Dtype.Float32)
        pcd_o3d.point["rhos"]    = o3d.core.Tensor(rhos, dtype=o3d.core.Dtype.Float32)
        pcd_o3d.point["fracs"]      = o3d.core.Tensor(fracs, dtype=o3d.core.Dtype.Float32)
        return pcd_o3d
    
    def construct_list_of_attributes(self):
        # l = ['x', 'y', 'z']
        # for i in range(len(self._bcoeffs)):
        #     l.append('bcoeffs_{}'.format(i))
        l = ['x', 'y', 'z', 'rhos']
        # for i in range(len(self._rhos)):
        #     l.append('rhos_{}'.format(i))
        for i in range(len(self._fracs)):
            l.append('fracs_{}'.format(i))
        for i in range(self._scaling.shape[1]):
            l.append('scale_{}'.format(i))
        for i in range(self._rotation.shape[1]):
            l.append('rot_{}'.format(i))
        return l

    def create_from_pcd(self, pcd : BasicPointCloud, spatial_lr_scale : float):
        self.spatial_lr_scale = spatial_lr_scale
        # self.spatial_lr_scale = 1.0

        fused_point_cloud = torch.tensor(np.asarray(pcd.points)).float().cuda()
        print("Number of points at initialisation : ", fused_point_cloud.shape[0])

        dist2 = torch.clamp_min(distCUDA2(torch.from_numpy(np.asarray(pcd.points)).float().cuda()), 0.0000001)
        # scales = torch.log(torch.sqrt(dist2) * 0.5)[...,None].repeat(1, 3)
        scales = torch.log(torch.sqrt(dist2))[...,None].repeat(1, 3)
        rots = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
        rots[:, 0] = 1

        self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
        
        # for bcoeff_id in self.bcoeff_ids:
        #     fused_bcoeffs = torch.tensor(np.asarray(pcd.bcoeffs[:,bcoeff_id])).float().unsqueeze(1).cuda()
        #     self._bcoeffs.append(nn.Parameter(fused_bcoeffs.requires_grad_(True)))
        
        rhos  = torch.tensor(pcd.rhos, dtype=torch.float32, device='cuda')
        fused_rhos = self.rhos_inverse_activation(rhos).float().cuda()
        self._rhos = nn.Parameter(fused_rhos.requires_grad_(True))
       
        fracs = torch.tensor(pcd.fracs, dtype=torch.float32, device='cuda')  # shape (N, C)
        fused_fracs = torch.log(fracs.clamp_min(1e-6))  # shape (N, C)
        for k in self.bcoeff_ids:
            self._fracs.append(nn.Parameter(fused_fracs[:, k:k+1].requires_grad_(True)))

        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(True))
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

    # ------------------ Training / LR ------------------  
    def training_setup(self, training_args):
        # Init accumulators
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")

        # Spectral Gaussian attributes
        self.fix_brdf_lr = training_args.fix_brdf_lr
        self.percent_dense = training_args.percent_dense

        # Define parameter list for basic Gaussian attributes
        # -- Spectral --
        # l = [
        #     {'params': [self._xyz], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "xyz"},       
        #     {'params': [self._scaling], 'lr': training_args.scaling_lr, "name": "scaling"},
        #     {'params': [self._rotation], 'lr': training_args.rotation_lr, "name": "rotation"}
        # ]

        # -- R2-Gaussian --
        l = [
            {
                "params": [self._xyz],
                "lr": training_args.position_lr_init * self.spatial_lr_scale,
                "name": "xyz",
            },
            {
                "params": [self._scaling],
                "lr": training_args.scaling_lr_init * self.spatial_lr_scale,
                "name": "scaling",
            },
            {
                "params": [self._rotation],
                "lr": training_args.rotation_lr_init * self.spatial_lr_scale,
                "name": "rotation",
            },
        ]

        # l.extend([
        #     {'params': [self._bcoeffs[0]], 'lr': training_args.pe_lr_init * self.spatial_lr_scale, "name": "bcoeffs_0"},
        #     {'params': [self._bcoeffs[1]], 'lr': training_args.compton_lr_init * self.spatial_lr_scale, "name": "bcoeffs_1"}
        # ])
        l.extend([
            {'params': [self._rhos], 'lr': training_args.opacity_lr_init * self.spatial_lr_scale, "name": "rhos"},
        ])
        
        # l.extend([
        #     # {'params': [self._rhos[0]], 'lr': training_args.pe_lr_init * self.spatial_lr_scale, "name": "rhos_0"},
        #     # {'params': [self._rhos[1]], 'lr': training_args.compton_lr_init * self.spatial_lr_scale, "name": "rhos_1"},
        #     {'params': [self._fracs[0]], 'lr': training_args.pe_lr_init * self.spatial_lr_scale, "name": "fracs_0"},
        #     {'params': [self._fracs[1]], 'lr': training_args.compton_lr_init * self.spatial_lr_scale, "name": "fracs_1"}
        # ])
        for j in self.bcoeff_ids:                  # [0,1,2]
            lr_init = getattr(training_args, f"basis{j}_lr_init")
            l.append({'params': [self._fracs[j]],
                      'lr': lr_init * self.spatial_lr_scale,
                      'name': f'fracs_{j}'})

        # Init Adam optimizer
        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)

        # Set up learning-rate schedulers
        # -- Spectral --
        # self.xyz_scheduler_args = get_expon_lr_func(
        #     lr_init=training_args.position_lr_init*self.spatial_lr_scale,
        #     lr_final=training_args.position_lr_final*self.spatial_lr_scale,
        #     lr_delay_mult=training_args.position_lr_delay_mult,
        #     max_steps=training_args.position_lr_max_steps
        # )

        # -- R2-Gaussian --
        self.xyz_scheduler_args = get_expon_lr_func(
            lr_init=training_args.position_lr_init * self.spatial_lr_scale,
            lr_final=training_args.position_lr_final * self.spatial_lr_scale,
            max_steps=training_args.position_lr_max_steps,
        )
        self.rhos_scheduler_args = get_expon_lr_func(
            lr_init=training_args.opacity_lr_init * self.spatial_lr_scale,
            lr_final=training_args.opacity_lr_final * self.spatial_lr_scale,
            max_steps=training_args.opacity_lr_max_steps,
        )
        self.scaling_scheduler_args = get_expon_lr_func(
            lr_init=training_args.scaling_lr_init * self.spatial_lr_scale,
            lr_final=training_args.scaling_lr_final * self.spatial_lr_scale,
            max_steps=training_args.scaling_lr_max_steps,
        )
        self.rotation_scheduler_args = get_expon_lr_func(
            lr_init=training_args.rotation_lr_init * self.spatial_lr_scale,
            lr_final=training_args.rotation_lr_final * self.spatial_lr_scale,
            max_steps=training_args.rotation_lr_max_steps,
        )
        
        # self.fracs_0_scheduler_args = get_expon_lr_func(
        #     lr_init=training_args.pe_lr_init * self.spatial_lr_scale,
        #     lr_final=training_args.pe_lr_final * self.spatial_lr_scale,
        #     max_steps=training_args.pe_lr_max_steps,
        # )
        # self.fracs_1_scheduler_args = get_expon_lr_func(
        #     lr_init=training_args.compton_lr_init * self.spatial_lr_scale,
        #     lr_final=training_args.compton_lr_final * self.spatial_lr_scale,
        #     max_steps=training_args.compton_lr_max_steps,
        # )
        for j in self.bcoeff_ids:
            setattr(self, f"fracs_{j}_scheduler_args",
                get_expon_lr_func(
                    lr_init = getattr(training_args, f"basis{j}_lr_init")  * self.spatial_lr_scale,
                    lr_final= getattr(training_args, f"basis{j}_lr_final") * self.spatial_lr_scale,
                    max_steps=getattr(training_args, f"basis{j}_lr_max_steps"),
                ))
            
    # def _update_learning_rate(self, iteration, param):
    #     for param_group in self.optimizer.param_groups:
    #         if param_group["name"] == param:
    #             try:
    #                 if param.endswith(str(self.active_spectra)):
    #                     spectral_id = int(param[-1])
    #                     param_new = param[:-2]
    #                     lr = getattr(self, f"{param_new}_scheduler_args[{spectral_id}]")(iteration)
    #                 else:    
    #                     lr = getattr(self, f"{param}_scheduler_args")(iteration)
    #                 param_group['lr'] = lr
    #                 return lr
    #             except AttributeError:
    #                 pass
    #         if param_group["lr"] == 0.0:           # frozen
    #             return 0.0

    def _update_learning_rate(self, iteration, param):
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == param:
                if param == "rhos" and getattr(self, "_rho_frozen", False):
                    param_group["lr"] = 0.0
                    return 0.0
                try:
                    if param.endswith(str(self.active_spectra)):
                        spectral_id = int(param[-1])
                        param_new = param[:-2]
                        lr = getattr(self, f"{param_new}_scheduler_args[{spectral_id}]")(iteration)
                    else:
                        lr = getattr(self, f"{param}_scheduler_args")(iteration)
                    param_group["lr"] = lr
                    return lr
                except AttributeError:
                    pass
            if param_group["lr"] == 0.0:
                return 0.0

    def update_learning_rate(self, iteration):
        ''' Learning rate scheduling per step '''
        # Basic Gaussian
        self._update_learning_rate(iteration, "xyz")
        
        # self._update_learning_rate(iteration, "opacity") # R2-Gaussian
        self._update_learning_rate(iteration, "scaling") # R2-Gaussian
        self._update_learning_rate(iteration, "rotation") # R2-Gaussian

        # self._update_learning_rate(iteration, "bcoeffs_0")
        # self._update_learning_rate(iteration, "bcoeffs_1")

        self._update_learning_rate(iteration, "rhos")
        # self._update_learning_rate(iteration, "rhos_0")
        # self._update_learning_rate(iteration, "rhos_1")
        # self._update_learning_rate(iteration, "fracs_0")
        # self._update_learning_rate(iteration, "fracs_1")

        for j in self.bcoeff_ids:
            self._update_learning_rate(iteration, f"fracs_{j}")

        # T = 500.0
        # t = min(1.0, iteration / T)
        # self._frac_eps = float(0.10 * (1.0 - t))

        T = 4000.0
        t = min(1.0, iteration / T)
        self._frac_eps = float(0.30 * (1.0 - t)) 

        # T = 3000.0
        # t = min(1.0, iteration / T)
        # self._frac_eps = float(0.20 * (1.0 - t))

        # Spectral Gaussian
        # parameter_list = []
        # for spectral_id in self.spectral_ids:
        #     parameter_list.extend(["brdf_mlp_" + str(spectral_id),"roughness_"+str(spectral_id),"specular_"+str(spectral_id),"normal_"+str(spectral_id),"f_dc_"+str(spectral_id), "f_rest_"+str(spectral_id)])
        # for param in parameter_list:
        #     lr = self._update_learning_rate(iteration, param)


    # ------------------ Reset Opacity ------------------
    def reset_bcoeffs(self, reset_rho = 1.0, min_frac = 0.05):
        with torch.no_grad():
            rho    = self.rhos_activation(self._rhos)
            rho_cl = torch.min(rho, torch.full_like(rho, reset_rho))
            new_raw = self.rhos_inverse_activation(rho_cl)
            name = f"rhos"
            self._rhos = self.replace_tensor_to_optimizer(new_raw, name)[name]

            raw_logits = torch.cat(self._fracs, dim=1)
            probs      = torch.softmax(raw_logits, dim=1)
            probs      = probs.clamp_min(min_frac)
            probs      = probs / probs.sum(dim=1, keepdim=True)
            new_logits = torch.log(probs)

            for k in range(len(self._fracs)):
                name   = f"fracs_{k}"
                tensor = new_logits[:, k:k+1]
                self._fracs[k] = self.replace_tensor_to_optimizer(tensor, name)[name]

            self.decay_optimizer_moments(decay=0.5)

    # def reset_bcoeffs(self, reset_rho = 1.0, min_frac = 1e-4):
    #     with torch.no_grad():
    #         rho    = self.rhos_activation(self._rhos)
    #         rho_cl = torch.min(rho, torch.full_like(rho, reset_rho))
    #         new_raw = self.rhos_inverse_activation(rho_cl)
    #         name = f"rhos"
    #         self._rhos = self.replace_tensor_to_optimizer(new_raw, name)[name]

    #         raw_logits = torch.cat(self._fracs, dim=1)
    #         p = stable_simplex(raw_logits, tau=getattr(self, "_frac_tau", 1.0), eps=0.02, clip=20.0)
    #         p = p.clamp_min(max(1e-6, min_frac))
    #         p = p / p.sum(dim=1, keepdim=True)
    #         new_logits = torch.log(p)  

    #         for k in range(len(self._fracs)):
    #             name   = f"fracs_{k}"
    #             tensor = new_logits[:, k:k+1]
    #             self._fracs[k] = self.replace_tensor_to_optimizer(tensor, name)[name]

    #         self.decay_optimizer_moments(decay=0.5)

    def replace_tensor_to_optimizer(self, tensor, name):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] == name:
                stored_state = self.optimizer.state.get(group['params'][0], None)
                stored_state["exp_avg"] = torch.zeros_like(tensor)
                stored_state["exp_avg_sq"] = torch.zeros_like(tensor)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors
    
    
    # ------------------ Densification / Pruning ------------------
    # def _set_subset_rho_in_activation_space(self, idx, rho_new_act):
    #     """
    #     idx: (M,) LongTensor, indices of parents to modify
    #     rho_new_act: (M,1) tensor in ACTIVATED space (after Softplus)
    #     """
    #     with torch.no_grad():
    #         raw = self._rhos.clone()
    #         raw[idx] = self.rhos_inverse_activation(rho_new_act)
    #     self._rhos = self.replace_tensor_to_optimizer(raw, "rhos")["rhos"]

    def densify_and_prune(self, 
        max_grad, 
        min_opacity, 
        extent, 
        max_screen_size,
        max_scale,
        max_num_gaussians,
        opacity_scale_threshold,
        bbox=None,
        only_active_spectra=False
    ):
        grads = self.xyz_gradient_accum / self.denom
        grads[grads.isnan()] = 0.0

        # -- R2-Gaussian --
        # Densify Gaussians if Gaussians are fewer than threshold
        if opacity_scale_threshold:
            if not max_num_gaussians or (
                max_num_gaussians and grads.shape[0] < max_num_gaussians
            ):
                self.densify_and_clone(grads, max_grad, extent, opacity_scale_threshold, only_active_spectra=only_active_spectra)
                self.densify_and_split(grads, max_grad, extent, opacity_scale_threshold, only_active_spectra=only_active_spectra)

        # -- R2-Gaussian --
        # Prune gaussians with too small density
        # bcoeffs = self.get_rhos
        # prune_mask = ((bcoeffs[:,0] < min_opacity) & (bcoeffs[:,1] < min_opacity))
        # prune_mask = (bcoeffs < min_opacity).all(dim=1)
        
        # prune_mask = (self.get_rhos < min_opacity).squeeze()
        # b = self.get_bcoeffs                                  # (N, C)
        # prune_mask = (b.max(dim=1).values < min_opacity)
        # bsum = self.get_bcoeffs.sum(dim=1, keepdim=True)  # (N,1)
        # prune_mask = (bsum < min_opacity).squeeze()

        # # eff = self.get_rhos.squeeze() * self.get_fracs.max(dim=1).values
        # # prune_mask = (eff < min_opacity)

        prune_mask = torch.zeros(self.get_xyz.shape[0], dtype=torch.bool, device="cuda")
        prune_mask |= (self.get_rhos.squeeze() < min_opacity)
        
        # Prune gaussians outside the bbox
        if bbox is not None:
            xyz = self.get_xyz
            prune_mask_xyz = (
                (xyz[:, 0] < bbox[0, 0])
                | (xyz[:, 0] > bbox[1, 0])
                | (xyz[:, 1] < bbox[0, 1])
                | (xyz[:, 1] > bbox[1, 1])
                | (xyz[:, 2] < bbox[0, 2])
                | (xyz[:, 2] > bbox[1, 2])
            )
            prune_mask = prune_mask | prune_mask_xyz

        if max_screen_size:
            big_points_vs = self.max_radii2D > max_screen_size
            prune_mask = torch.logical_or(prune_mask, big_points_vs)
        if max_scale:
            big_points_ws = self.get_scaling.max(dim=1).values > max_scale
            prune_mask = torch.logical_or(prune_mask, big_points_ws)
        self.prune_points(prune_mask, only_active_spectra=only_active_spectra)

        torch.cuda.empty_cache()

    def densify_and_clone(self, grads, grad_threshold, scene_extent, densify_scale_threshold, only_active_spectra):
        # Extract points that satisfy the gradient condition
        selected_pts_mask = torch.where(
            torch.norm(grads, dim=-1) >= grad_threshold, True, False
        )
        selected_pts_mask = torch.logical_and(
            selected_pts_mask,
            torch.max(self.get_scaling, dim=1).values <= self.percent_dense*scene_extent
        )
        selected_pts_mask = torch.logical_and(
            selected_pts_mask,
            torch.max(self.get_scaling, dim=1).values <= densify_scale_threshold
        )
        
        new_xyz = self._xyz[selected_pts_mask]
        new_scaling = self._scaling[selected_pts_mask]
        new_rotation = self._rotation[selected_pts_mask]
        new_max_radii2D = self.max_radii2D[selected_pts_mask]
        # new_rhos = self._rhos[selected_pts_mask]

        # new_rhos = self.rhos_inverse_activation(
        #     self.get_rhos[selected_pts_mask] * 0.5
        # )
        # self._rhos[selected_pts_mask] = new_rhos

        with torch.no_grad():
            rho_sel     = self.get_rhos[selected_pts_mask]                                # (k,1) positive
            rho_half    = 0.5 * rho_sel
            new_rhos = self.rhos_inverse_activation(rho_half)                 # raw for child
            # also write back the parent's halved ρ (raw)
            self._rhos[selected_pts_mask] = new_rhos

        self.densification_postfix(new_xyz, new_scaling, new_rotation, new_max_radii2D, new_rhos)
        # self.densification_postfix(new_xyz, new_scaling, new_rotation, new_max_radii2D)

        # self._rhos[selected_pts_mask] = new_rhos
        
        # for bcoeff_id in self.bcoeff_ids:
        #     new_bcoeffs = self._bcoeffs[bcoeff_id][selected_pts_mask]
        #     self.densification_postfix_per_coeff(new_bcoeffs, bcoeff_id)

        for bcoeff_id in self.bcoeff_ids:
            # new_rhos = self._rhos[bcoeff_id][selected_pts_mask]
            new_fracs = self._fracs[bcoeff_id][selected_pts_mask]
            # self.densification_postfix_per_coeff(new_rhos, new_fracs, bcoeff_id)
            self.densification_postfix_per_coeff(new_fracs, bcoeff_id)

    # def densify_and_split(self, grads, grad_threshold, scene_extent, densify_scale_threshold, only_active_spectra, N=2):
    #     n_init_points = self.get_xyz.shape[0]
    #     # Extract points that satisfy the gradient condition
    #     padded_grad = torch.zeros((n_init_points), device="cuda")
    #     padded_grad[:grads.shape[0]] = grads.squeeze()
        
    #     selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)
    #     selected_pts_mask = torch.logical_and(
    #         selected_pts_mask,
    #         torch.max(self.get_scaling, dim=1).values > self.percent_dense*scene_extent
    #     )
    #     selected_pts_mask = torch.logical_and(
    #         selected_pts_mask,
    #         torch.max(self.get_scaling, dim=1).values > densify_scale_threshold,
    #     )

    #     stds = self.get_scaling[selected_pts_mask].repeat(N,1)
    #     means =torch.zeros((stds.size(0), 3),device="cuda")
    #     samples = torch.normal(mean=means, std=stds)
    #     rots = build_rotation(self._rotation[selected_pts_mask]).repeat(N,1,1)
        
    #     new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[selected_pts_mask].repeat(N, 1)
    #     new_scaling = self.scaling_inverse_activation(self.get_scaling[selected_pts_mask].repeat(N,1) / (0.8*N))
    #     new_rotation = self._rotation[selected_pts_mask].repeat(N,1)
    #     new_max_radii2D = self.max_radii2D[selected_pts_mask].repeat(N)
        
    #     # new_rhos = self._rhos[selected_pts_mask].repeat(N,1)
    #     # new_rhos = self.rhos_inverse_activation(
    #     #     self.get_rhos[selected_pts_mask].repeat(N, 1) * (1 / N)
    #     # )

    #     with torch.no_grad():
    #         rho_parent   = self.get_rhos[selected_pts_mask]                 # (k,1)
    #         rho_child    = rho_parent.repeat(N,1)* (1 / N)    # (kN,1)
    #         new_rhos = self.rhos_inverse_activation(rho_child)
    #         # new_rhos  = self._rhos[selected_pts_mask].repeat(N, 1)

    #     self.densification_postfix(new_xyz, new_scaling, new_rotation, new_max_radii2D, new_rhos)
    #     # self.densification_postfix(new_xyz, new_scaling, new_rotation, new_max_radii2D)
        
    #     # for bcoeff_id in self.bcoeff_ids:
    #     #     new_opacity = self._bcoeffs[bcoeff_id][selected_pts_mask].repeat(N,1)
    #     #     self.densification_postfix_per_coeff(new_opacity, bcoeff_id)

    #     for bcoeff_id in self.bcoeff_ids:
    #         # new_rhos = self._rhos[bcoeff_id][selected_pts_mask].repeat(N,1)
    #         new_fracs = self._fracs[bcoeff_id][selected_pts_mask].repeat(N,1)
    #         # self.densification_postfix_per_coeff(new_rhos, new_fracs, bcoeff_id)
    #         self.densification_postfix_per_coeff(new_fracs, bcoeff_id)

    #     prune_filter = torch.cat(
    #         (
    #             selected_pts_mask,
    #             torch.zeros(N * selected_pts_mask.sum(), device="cuda", dtype=bool),
    #         )
    #     )
    #     self.prune_points(prune_filter, only_active_spectra=only_active_spectra)
    

    # def densify_and_prune(
    #     self,
    #     max_grad,
    #     min_opacity,
    #     extent,
    #     max_screen_size,
    #     max_scale,
    #     max_num_gaussians,
    #     opacity_scale_threshold,
    #     bbox=None,
    #     only_active_spectra: bool = False,
    # ):
    #     """
    #     Adaptive control:
    #     • clone  → high-grad & small (refine edges)
    #     • split  → low-grad  & large (subdivide flats)
    #     • prune  → by raw ρ (avoid basis-biased pruning)
    #     """
    #     grads = self.xyz_gradient_accum / self.denom
    #     grads[grads.isnan()] = 0.0

    #     # densify if allowed by the global cap
    #     if opacity_scale_threshold and (
    #         not max_num_gaussians or self.get_xyz.shape[0] < max_num_gaussians
    #     ):
    #         # high-grad + small → clone
    #         self.densify_and_clone(
    #             grads,
    #             max_grad,
    #             extent,
    #             opacity_scale_threshold,
    #             only_active_spectra=only_active_spectra,
    #         )
    #         # low-grad + large → split
    #         self.densify_and_split(
    #             grads,
    #             max_grad,
    #             extent,
    #             opacity_scale_threshold,
    #             only_active_spectra=only_active_spectra,
    #             N=2,
    #         )

    #     # # ---------- pruning (safe) ----------
    #     prune_mask = torch.zeros(self.get_xyz.shape[0], dtype=torch.bool, device="cuda")

    #     # prune by *raw* density only (do NOT multiply by fractions)
    #     prune_mask |= (self.get_rhos.squeeze() < min_opacity)

    #     if bbox is not None:
    #         xyz = self.get_xyz
    #         prune_mask |= (
    #             (xyz[:, 0] < bbox[0, 0]) | (xyz[:, 0] > bbox[1, 0]) |
    #             (xyz[:, 1] < bbox[0, 1]) | (xyz[:, 1] > bbox[1, 1]) |
    #             (xyz[:, 2] < bbox[0, 2]) | (xyz[:, 2] > bbox[1, 2])
    #         )

    #     if max_screen_size:
    #         prune_mask |= (self.max_radii2D > max_screen_size)

    #     if max_scale:
    #         prune_mask |= (self.get_scaling.max(dim=1).values > max_scale)

    #     if prune_mask.any():
    #         self.prune_points(prune_mask, only_active_spectra=only_active_spectra)

    
    # def densify_and_clone(
    #     self,
    #     grads,
    #     grad_threshold,
    #     scene_extent,
    #     densify_scale_threshold,
    #     only_active_spectra: bool,
    # ):
    #     """
    #     Clone high-gradient, *small* Gaussians.
    #     Halve the parent's ρ; child receives the other half.
    #     Fractions are copied 1:1 (no noise).
    #     """
    #     g = torch.norm(grads, dim=-1) if grads.dim() > 1 else grads.flatten().abs()

    #     smax = self.get_scaling.max(dim=1).values
    #     small = (smax <= self.percent_dense * scene_extent) & (smax <= densify_scale_threshold)
    #     selected = (g >= grad_threshold) & small
    #     if selected.sum() == 0:
    #         return

    #     new_xyz         = self._xyz[selected]
    #     new_scaling     = self._scaling[selected]       # raw copy
    #     new_rotation    = self._rotation[selected]
    #     new_max_radii2D = self.max_radii2D[selected]

    #     with torch.no_grad():
    #         rho_sel  = self.get_rhos[selected]          # activated
    #         rho_half = 0.5 * rho_sel
    #         child_rho_raw = self.rhos_inverse_activation(rho_half)
    #         # halve parent as well (in-place under no_grad)
    #         self._rhos[selected] = child_rho_raw

    #     self.densification_postfix(
    #         new_xyz, new_scaling, new_rotation, new_max_radii2D, child_rho_raw
    #     )

    #     # copy logits exactly so mixture is preserved
    #     for j in self.bcoeff_ids:
    #         self.densification_postfix_per_coeff(self._fracs[j][selected], j)

    def densify_and_split(
        self,
        grads,
        grad_threshold,
        scene_extent,
        densify_scale_threshold,
        only_active_spectra: bool,
        N: int = 2,
        offset_std_scale: float = 0.35,   # child offset in units of parent std
        scale_shrink: float = 0.90,       # extra shrink (× N^{-1/3} overall)
    ):
        """
        Split low-gradient, *large* Gaussians into N children.

        Positions: x_child = x_parent + R · 𝒩(0, (offset_std_scale·diag(scale)) )
        Scales:    s_child = s_parent · (scale_shrink / N^(1/3))   (≈ volume-preserving)
        ρ:         equally divided across children (in activated space)
        Fractions: copy logits exactly (no noise)
        Parents:   removed after children are appended.
        """
        n0 = self.get_xyz.shape[0]

        # robust grad vector with padding for newly created points
        g = grads.flatten() if grads.dim() > 1 else grads
        if g.numel() < n0:
            pad = torch.zeros((n0,), device="cuda", dtype=g.dtype)
            pad[: g.numel()] = g
            g = pad

        s = self.get_scaling
        smax = s.max(dim=1).values
        large = (smax > self.percent_dense * scene_extent) & (smax > densify_scale_threshold)
        selected = (g <= grad_threshold) & large
        M = int(selected.sum())
        if M == 0:
            return

        # parents
        x_parent = self._xyz[selected]         # raw
        s_parent = s[selected]                 # activated
        r_parent = self._rotation[selected]    # raw
        R_parent = build_rotation(r_parent)    # (M,3,3)
        rho_act  = self.get_rhos[selected]     # activated

        # offsets in local coords → world
        std_local = offset_std_scale * s_parent                     # (M,3)
        eps = torch.randn((M * N, 3), device="cuda") * std_local.repeat(N, 1)
        offs = torch.bmm(R_parent.repeat(N, 1, 1), eps.unsqueeze(-1)).squeeze(-1)

        new_xyz         = x_parent.repeat(N, 1) + offs
        shrink          = scale_shrink / (N ** (1.0 / 3.0))
        new_scaling     = self.scaling_inverse_activation(s_parent.repeat(N, 1) * shrink)
        new_rotation    = r_parent.repeat(N, 1)
        new_max_radii2D = self.max_radii2D[selected].repeat(N)

        # split the "mass" equally
        child_rho_act = rho_act.repeat(N, 1) / float(N)
        child_rho_raw = self.rhos_inverse_activation(child_rho_act)

        self.densification_postfix(
            new_xyz, new_scaling, new_rotation, new_max_radii2D, child_rho_raw
        )

        # exact mixture copy (no noise to avoid biasing rare bases like metal)
        for j in self.bcoeff_ids:
            self.densification_postfix_per_coeff(self._fracs[j][selected].repeat(N, 1), j)

        # prune the parents we just split
        prune_mask = torch.zeros(n0 + M * N, dtype=torch.bool, device="cuda")
        prune_mask[:n0] = selected
        self.prune_points(prune_mask, only_active_spectra=only_active_spectra)


    def densification_postfix(self, new_xyz, new_scaling, new_rotation, new_max_radii2D, new_rhos):
        """
        Append new points to all optimizer-managed tensors and reset running stats
        for the newly created entries.
        """
        d = {
            "xyz":      new_xyz,
            "scaling":  new_scaling,   # raw
            "rotation": new_rotation,  # raw
            "rhos":     new_rhos,      # raw
        }
        optimizable_tensors = self.cat_tensors_to_optimizer(d, -1)

        self._xyz      = optimizable_tensors["xyz"]
        self._scaling  = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]
        self._rhos     = optimizable_tensors["rhos"]

        # reset/extend accumulators
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom              = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.max_radii2D        = torch.cat([self.max_radii2D, new_max_radii2D], dim=0)

        # damp optimizer momentum so the new entries can adapt
        self.decay_optimizer_moments(decay=0.5)

    def densification_postfix_per_coeff(self, new_fracs, bcoeff_id):
        d = {}
        d.update({"fracs_"+ str(bcoeff_id): new_fracs})

        optimizable_tensors = self.cat_tensors_to_optimizer(d, bcoeff_id)
        self._fracs[bcoeff_id] = optimizable_tensors["fracs_"+ str(bcoeff_id)]

    def decay_optimizer_moments(self, decay: float = 0.5):
        for state in self.optimizer.state.values():          # one dict per param-tensor
            if "exp_avg" in state:       # guard against empty states
                state["exp_avg"]    *= decay
                state["exp_avg_sq"] *= decay

    def cat_tensors_to_optimizer(self, tensors_dict, bcoeff_id):
        optimizable_tensors = {} 
        for group in self.optimizer.param_groups:
            if group["name"][-1].isdigit():
                key = group["name"]
                if int(key[key.rindex("_") + 1:]) != bcoeff_id:
                    continue
            elif not group["name"][-1].isdigit() and bcoeff_id > -1:
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

    def prune_points(self, mask, only_active_spectra):
        valid_points_mask = ~mask
        optimizable_tensors = self._prune_optimizer(valid_points_mask)

        self._xyz = optimizable_tensors["xyz"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]
        
        # for bcoeff_id in self.bcoeff_ids:
        #     self._bcoeffs[bcoeff_id] = optimizable_tensors["bcoeffs_"+ str(bcoeff_id)]
        
        self._rhos = optimizable_tensors["rhos"]
        for bcoeff_id in self.bcoeff_ids:
            # self._rhos[bcoeff_id] = optimizable_tensors["rhos_"+ str(bcoeff_id)]
            self._fracs[bcoeff_id] = optimizable_tensors["fracs_"+ str(bcoeff_id)]

        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]
        self.denom = self.denom[valid_points_mask]
        self.max_radii2D = self.max_radii2D[valid_points_mask]

    def _prune_optimizer(self, mask):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
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
