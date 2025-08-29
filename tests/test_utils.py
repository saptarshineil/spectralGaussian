import os, json, math, argparse
import numpy as np
import torch
from pathlib import Path
from utils.spectralCT_utils import (
    energy_quadrature, get_scaled_spectrum, get_B_E, bfunc_PE, bfunc_Compton
)

DEFAULTS = dict(
    E_min=20.0, E_max=140.0, N=64,
    kvp_low=50, kvp_high=120,
    filt_mat_high="Cu", filt_thick_high=0.5,
    I0_low=1136.8784583333334, I0_high=2026.8482152777776,
)

def dev():
    return "cuda" if torch.cuda.is_available() else "cpu"

def ensure_dir(p):
    Path(p).mkdir(parents=True, exist_ok=True)
    return str(p)

def build_grid_and_spectra(N=None, E_min=None, E_max=None,
                           kvp_low=None, kvp_high=None,
                           filt_mat_high=None, filt_thick_high=None,    
                           I0_low=None, I0_high=None):
    """Mirror train.py defaults so tests see the exact same setup."""
    cfg = DEFAULTS.copy()
    for k, v in dict(N=N, E_min=E_min, E_max=E_max, kvp_low=kvp_low, kvp_high=kvp_high,
                     filt_mat_high=filt_mat_high, filt_thick_high=filt_thick_high,
                     I0_low=I0_low, I0_high=I0_high).items():
        if v is not None: cfg[k] = v
    device = dev()
    E_grid, w_k = energy_quadrature(cfg["E_min"], cfg["E_max"], cfg["N"])
    E_np = E_grid.detach().cpu().numpy()
    S_low  = get_scaled_spectrum(cfg["kvp_low"],  None,              0.0, cfg["I0_low"],  E_np)
    S_high = get_scaled_spectrum(cfg["kvp_high"], cfg["filt_mat_high"], cfg["filt_thick_high"],
                                 cfg["I0_high"], E_np)
    S = torch.stack([S_low, S_high])        # (2, K)
    I0 = torch.sum(S * w_k, dim=1)          # (2,)
    B_E = get_B_E(E_grid, bfunc_Kedge=None) # (K, C)
    return E_grid, w_k, S, I0, B_E

def beer_lambert(I0_scalar, S_spec, w_k, mu_E):
    """Return I_pred and line integral y=-log(I/I0)."""
    I_pred = torch.sum(S_spec * torch.exp(-mu_E) * w_k, dim=0)
    I_pred = torch.clamp(I_pred, min=1e-9 * I0_scalar)
    y = -torch.log(I_pred / I0_scalar)
    return I_pred, y

def save_npz(path, **arrays):
    ensure_dir(Path(path).parent)
    np.savez_compressed(path, **{k: (v.detach().cpu().numpy() if torch.is_tensor(v) else v)
                                 for k, v in arrays.items()})
    return path