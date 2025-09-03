# Copyright (C) 2023, Gaussian-Grouping
# Gaussian-Grouping research group, https://github.com/lkeab/gaussian-grouping
# All rights reserved.
#
# ------------------------------------------------------------------------
# Modified from codes in Gaussian-Splatting 
# GRAPHDECO research group, https://team.inria.fr/graphdeco

from argparse import ArgumentParser, Namespace
import sys
import os

class GroupParams:
    pass

class ParamGroup:
    def __init__(self, parser: ArgumentParser, name : str, fill_none = False):
        group = parser.add_argument_group(name)
        for key, value in vars(self).items():
            shorthand = False
            if key.startswith("_"):
                shorthand = True
                key = key[1:]
            t = type(value)
            value = value if not fill_none else None 
            if shorthand:
                if t == bool:
                    group.add_argument("--" + key, ("-" + key[0:1]), default=value, action="store_true")
                else:
                    group.add_argument("--" + key, ("-" + key[0:1]), default=value, type=t)
            else:
                if t == bool:
                    group.add_argument("--" + key, default=value, action="store_true")
                else:
                    group.add_argument("--" + key, default=value, type=t)

    def extract(self, args):
        group = GroupParams()
        for arg in vars(args).items():
            if arg[0] in vars(self) or ("_" + arg[0]) in vars(self):
                setattr(group, arg[0], arg[1])
        return group

class ModelParams(ParamGroup): 
    def __init__(self, parser, sentinel=False):
        self.sh_degree = -1
        self._source_path = ""
        self._model_path = ""
        self._images = "images"
        self._resolution = -1
        self._white_background = True
        self.transparentbg = False
        self.data_device = "cuda"
        self.eval = False
        self.n_views = 100 
        self.random_init = False
        self.uniform_init = False
        self.train_split = False
        self._object_path = "object_mask"
        self.num_classes = 200
        self.brdf_dim = 0
        self.brdf_mode = "envmap"
        self.brdf_envmap_res = 64

        # -- R2-Gaussian --
        self.scale_min = 0.0005  # percent of volume size
        self.scale_max = 0.5  # percent of volume size

        super().__init__(parser, "Loading Parameters", sentinel)

    def extract(self, args):
        g = super().extract(args)
        g.source_path = os.path.abspath(g.source_path)
        g.brdf = g.brdf_dim>=0
        return g

class PipelineParams(ParamGroup):
    def __init__(self, parser):
        self.convert_SHs_python = False
        self.compute_cov3D_python = False
        self.debug = False
        super().__init__(parser, "Pipeline Parameters")
    

    def extract(self, args):
        g = super().extract(args)
        g.brdf = args.brdf_dim>=0
        if g.brdf:
            g.convert_SHs_python = True
        g.brdf_mode = args.brdf_mode
        return g

class OptimizationParams(ParamGroup):
    def __init__(self, parser):
        # self.iterations = 30_000
        self.iterations = 5_000
        # self.iterations = 1000
        # self.iterations = 10
        
        # -- Spectral --> Gaussian basics --
        # self.position_lr_init = 0.00016
        # self.position_lr_final = 0.0000016
        # self.position_lr_delay_mult = 0.01
        # self.position_lr_max_steps = 30_000
        # self.opacity_lr = 0.05
        # self.scaling_lr = 0.005
        # self.rotation_lr = 0.001

        # -- R2-Gaussian --> Gaussian basics --
        self.position_lr_init = 0.0002
        self.position_lr_final = 0.00002
        self.position_lr_max_steps = 30_000
        self.scaling_lr_init = 0.005
        self.scaling_lr_final = 0.0005
        self.scaling_lr_max_steps = 30_000
        self.rotation_lr_init = 0.001
        self.rotation_lr_final = 0.0001
        self.rotation_lr_max_steps = 30_000
        
        # -- Spectral --> LR --
        self.feature_lr = 0.0025
        self.percent_dense = 0.01
        # self.opacity_reset_interval = 3000
        self.opacity_reset_interval = 60000
        # self.lambda_dssim = 0.005                 # spectral
        self.lambda_dssim = 0.25              # R2
        # self.lambda_dssim = 0.80              # Test
        
        # -- Spectral --> densification --
        self.densification_interval = 100      # spectral + R2 
        # self.densify_from_iter = 500           # spectral + R2
        self.densify_from_iter = 50000           # spectral + R2
        self.densify_until_iter = 15_000       # spectral + R2
        # self.densify_grad_threshold = 0.00050   # R2
        # self.densify_grad_threshold = 0.00015
        # self.densify_grad_threshold = 0.00025
        self.densify_grad_threshold = 0.0001
        # self.densify_grad_threshold = 0.00017
        # self.densify_grad_threshold = 0.005
        self.switch_iter = 1_500

        self.densify_scale_threshold = 0.1     # R2 (percent of volume size)
        self.density_min_threshold = 0.00001   # R2
        # self.density_min_threshold = 0.000001   # R2

        # -- Spectral --> segmentation --
        self.reg3d_interval = 2
        self.reg3d_k = 5
        self.reg3d_lambda_val = 2
        self.reg3d_max_points = 300000
        self.reg3d_sample_size = 1000

        # -- Spectral --> extra --
        self.brdf_mlp_lr_init = 1.6e-2
        self.brdf_mlp_lr_final = 1.6e-3
        self.brdf_mlp_lr_delay_mult = 0.01
        self.brdf_mlp_lr_max_steps = 30_000
        self.normal_lr = 0.0002
        self.specular_lr = 0.0002
        self.roughness_lr = 0.0002
        self.normal_reg_from_iter = 0
        self.normal_reg_util_iter = 30_000
        self.lambda_zero_one = 1e-3
        self.lambda_predicted_normal = 2e-1
        self.lambda_delta_reg = 1e-3
        self.fix_brdf_lr = 0

        # -- R2-Gaussian --
        self.lambda_tv = 0.005
        self.tv_vol_size = 32
        self.max_scale = None
        self.max_num_gaussians = 500_000
        self.max_screen_size = None

        # -- Test --
        # self.pe_lr_init = 0.005  # PE initial learning rate (higher)
        # self.pe_lr_final = 0.0005
        # self.pe_lr_max_steps = 30000

        # self.compton_lr_init = 0.001  # Compton initial learning rate (lower)
        # self.compton_lr_final = 0.0001
        # self.compton_lr_max_steps = 30000

        # Density
        self.opacity_lr_init = 0.0001
        self.opacity_lr_final = 0.00001
        # self.opacity_lr_init = 0.0005
        # self.opacity_lr_final = 0.00005
        self.opacity_lr_max_steps = 30_000
        # PE
        self.basis0_lr_init = 0.001
        self.basis0_lr_final = 0.0001
        # self.basis0_lr_init = 0.005     # Lower PSNR, but better segmentation
        # self.basis0_lr_final = 0.0005
        self.basis0_lr_max_steps = 30000
        # Compton
        self.basis1_lr_init = 0.001
        self.basis1_lr_final = 0.0001
        # self.basis1_lr_init = 0.005     # Lower PSNR, but better segmentation
        # self.basis1_lr_final = 0.0005
        self.basis1_lr_max_steps = 30000
        # K-edge
        self.basis2_lr_init = 0.001
        self.basis2_lr_final = 0.0001
        # self.basis2_lr_init = 0.005     # Lower PSNR, but better segmentation
        # self.basis2_lr_final = 0.0005
        self.basis2_lr_max_steps = 30000

        self.frac_warmup_iters = 1000
        self.energy_warmup_iters  = 6000
        self.energy_lr_after_warmup = 1e-4

        # self.lambda_tv_basis  = 1e-3
        self.lambda_tv_basis_pe   = 1e-3
        self.lambda_tv_basis_comp = 1e-3
        self.lambda_tv_basis_K = 1e-3
        
        self.lambda_dirichlet = 1e-4
        self.lambda_ortho     = 1e-3

        super().__init__(parser, "Optimization Parameters")

def get_combined_args(parser : ArgumentParser):
    cmdlne_string = sys.argv[1:]
    cfgfile_string = "Namespace()"
    args_cmdline = parser.parse_args(cmdlne_string)

    try:
        cfgfilepath = os.path.join(args_cmdline.model_path, "cfg_args")
        print("Looking for config file in", cfgfilepath)
        with open(cfgfilepath) as cfg_file:
            print("Config file found: {}".format(cfgfilepath))
            cfgfile_string = cfg_file.read()
    except TypeError:
        print("Config file not found at")
        pass
    args_cfgfile = eval(cfgfile_string)

    merged_dict = vars(args_cfgfile).copy()
    for k,v in vars(args_cmdline).items():
        if v != None:
            merged_dict[k] = v
    return Namespace(**merged_dict)
