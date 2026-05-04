#!/usr/bin/env python3
"""
Generate a synthetic 3-basis spectral CT dataset for testing the 3-basis
coefficient model (PE + Compton + K-edge).

Output structure matches data/ct_spec_new/quail3 exactly:
  data/ct_spec_new/<name>/
    images/                     0001..0720.npy  (= spectral_0, for loading)
    images_train/               75 training views
    images_test/                100 test views
    spectra/
      spectral_0/               0001..0720.npy  50 kVp polychromatic
      spectral_1/               0001..0720.npy  80 kVp polychromatic
      spectral_2/               0001..0720.npy  120 kVp polychromatic
    vol_gt.npy                  (128,128,128) reference LAC at 70 keV
    init_pcd.npy                (50000, 4) sparse point cloud (x,y,z,density)
    init_pcd.ply                (PLY format point cloud)
    meta_data.json              scanner params + 75 training camera entries

Phantom design (3 materials for unambiguous 3-basis testing):
  Material A – soft tissue shell:  high Compton, low PE,  no  K-edge
  Material B – bone core:           high PE,     high Compton, no  K-edge
  Material C – K-edge absorber:     moderate PE+Compton,  strong K-edge (E_K=63 keV)
"""

import os
import sys
import json
import numpy as np
import torch
import torch.nn.functional as Fn
from pathlib import Path
from plyfile import PlyData, PlyElement

# ---------------------------------------------------------------------------
# Make sure the project root is on the path so we can import spectralCT_utils
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from utils.spectralCT_utils import (
    get_spectral_constants,
    bfunc_PE,
    bfunc_Compton,
    get_B_E,
    interp1d,
)

# ===========================================================================
# Configuration
# ===========================================================================
OUTPUT_NAME  = "synthetic3basis"
OUTPUT_DIR   = ROOT / "data" / "ct_spec_new" / OUTPUT_NAME

N_ANGLES  = 720
N_TRAIN   = 75       # cameras in meta_data.json  (training views)
N_TEST    = 100      # views saved to images_test/
N_COARSE  = 100      # test views come from 100 non-training angles
N_INIT_PCD = 50_000  # points in init_pcd.npy

DEVICE  = "cuda" if torch.cuda.is_available() else "cpu"
N_STEPS = 256        # ray-marching steps through the volume
ROW_BATCH = 48       # detector rows processed per GPU batch (memory limit)

# Scanner parameters – identical to quail3
SCANNER = dict(
    mode        = "cone",
    DSD         = 4.1999963379,
    DSO         = 2.5199943541999996,
    nDetector   = [576, 552],
    sDetector   = [4.6080000000000005, 4.416],
    nVoxel      = [128, 128, 128],
    sVoxel      = [2.0,  2.0,  2.0],
    offOrigin   = [0.0, 0.0, 0.0],
    offDetector = [0.0, 0.0],
    accuracy    = 0.5,
    totalAngle  = 359.5,
    startAngle  = 0.0,
    noise       = False,
    filter      = None,
)

# K-edge energy for the synthetic absorber (keV).
# Placed between channel-0 (50 kVp, spectrum up to 50 keV) and channel-1
# (80 kVp, spectrum up to 80 keV).  With alpha >> 1 the sigmoid approximates
# a true sharp step at E_K so that the 50 kVp channel sees ZERO K-edge
# contribution while both 80 kVp and 120 kVp channels are affected.
E_KEDGE      = 63.0   # keV
KEDGE_ALPHA  = 100.0  # large → sharp step ≈ physical K-edge

# ===========================================================================
# 1.  Synthetic K-edge basis function (smooth step above E_KEDGE)
# ===========================================================================

class BfuncKedgeSynthetic(torch.nn.Module):
    """Smooth positive step function rising above E_KEDGE."""

    def __init__(self, E_grid, E_K=E_KEDGE, alpha=KEDGE_ALPHA):
        super().__init__()
        self.E_grid = E_grid
        self.E_K    = E_K
        self.alpha  = alpha

    def forward(self):
        """Return basis values on E_grid."""
        return torch.sigmoid(
            self.alpha * (self.E_grid - self.E_K) / self.E_K
        )


# ===========================================================================
# 2.  Phantom: 128³ volume with three distinct material regions
# ===========================================================================

def make_phantom(N=128):
    """
    Returns three (N,N,N) float32 arrays: c_PE, c_Compton, c_Kedge.

    Storage convention: array[iz, iy, ix].

    Geometry (in scene coordinates [-1, 1]):
      Outer ellipsoid (soft tissue shell, A): semi-axes (0.75, 0.75, 0.70)
      Bone core sphere      (material B):    radius 0.22 at origin
      K-edge sphere 1       (material C):    radius 0.22 at (+0.40, +0.30, 0)
      K-edge sphere 2       (material C):    radius 0.18 at (-0.33, -0.25, +0.15)

    The K-edge spheres are fully outside the bone core (nearest edges are
    0.28 and 0.27 scene units from the bone surface respectively), so every
    ray passing through a K-edge voxel travels only through tissue – giving a
    PURE K-edge spectral signature with no bone dilution.

    Material basis coefficients (ρ = c_PE + c_Compton + c_Kedge = LAC at 70 keV,
    because B_j_norm(70) = 1 for all j in the normalised basis):
      A – soft tissue:  c_PE=0.028, c_Compton=0.322, c_Kedge=0.000  → ρ=0.35
      B – bone:         c_PE=0.300, c_Compton=0.700, c_Kedge=0.000  → ρ=1.00
      C – K-edge mat.:  c_PE=0.028, c_Compton=0.172, c_Kedge=0.400  → ρ=0.60

    Material C is isoelectric (same PE as tissue) so its spectral anomaly is
    PURELY from the K-edge term.  Since the 50 kVp spectrum (ch0) has virtually
    no photons above E_K=63 keV, while the 120 kVp spectrum (ch2) has ~58% of
    its photons above E_K, the K-edge material shows anomalously HIGH ch2
    attenuation compared to what PE+Compton alone would predict.  This is the
    ideal signal for unambiguously testing the 3-basis model.
    """
    # Voxel-centre coordinates in [-1, 1]
    grid = (np.arange(N, dtype=np.float32) + 0.5) / N * 2.0 - 1.0  # (N,)
    iz, iy, ix = np.meshgrid(grid, grid, grid, indexing="ij")        # (N,N,N) each

    r2_center = ix**2 + iy**2 + iz**2

    r2_outer = (ix/0.75)**2 + (iy/0.75)**2 + (iz/0.70)**2

    # Bone core
    R_BONE = 0.22

    # K-edge sphere 1: centre (0.40, 0.30, 0) – distance from bone centre = 0.500
    CX1, CY1, CZ1, R_K1 = 0.40, 0.30, 0.0, 0.22
    r2_k1 = (ix - CX1)**2 + (iy - CY1)**2 + (iz - CZ1)**2

    # K-edge sphere 2: centre (-0.33, -0.25, 0.15) – distance from bone centre ≈ 0.453
    CX2, CY2, CZ2, R_K2 = -0.33, -0.25, 0.15, 0.18
    r2_k2 = (ix - CX2)**2 + (iy - CY2)**2 + (iz - CZ2)**2

    in_outer  = r2_outer  <= 1.0
    in_bone   = r2_center <= R_BONE**2
    in_kedge  = ((r2_k1 <= R_K1**2) | (r2_k2 <= R_K2**2)) & in_outer

    tissue_mask = in_outer & ~in_bone & ~in_kedge
    bone_mask   = in_bone  & ~in_kedge
    kedge_mask  = in_kedge

    c_PE      = np.zeros((N, N, N), dtype=np.float32)
    c_Compton = np.zeros((N, N, N), dtype=np.float32)
    c_Kedge   = np.zeros((N, N, N), dtype=np.float32)

    # Material A – soft tissue (water-like, mostly Compton)
    c_PE[tissue_mask]      = 0.028
    c_Compton[tissue_mask] = 0.322

    # Material B – bone (high PE fraction, dense)
    c_PE[bone_mask]      = 0.300
    c_Compton[bone_mask] = 0.700

    # Material C – K-edge absorber (isoelectric: same PE as tissue so spectral
    # contrast is PURELY from the K-edge component, not from PE difference).
    # c_Kedge fraction = 0.667 → strongly K-edge dominated at 120 kVp.
    # At 50 kVp (all photons < 63 keV = E_K): K-edge contribution ≈ 0.
    # At 120 kVp (57.9% of spectrum > 63 keV): large K-edge contribution.
    # This creates an unambiguous anomalous ch2 attenuation that can only be
    # explained by a K-edge basis function – ideal for testing the 3-basis model.
    c_PE[kedge_mask]      = 0.028   # same as tissue → no PE-induced spectral bias
    c_Compton[kedge_mask] = 0.172   # reduced to keep ρ=0.60
    c_Kedge[kedge_mask]   = 0.400   # dominant K-edge component

    return c_PE, c_Compton, c_Kedge


# ===========================================================================
# 3.  Spectral model setup
# ===========================================================================

def build_spectral_model(device=DEVICE):
    """
    Returns the same spectral constants used in spectralCT_utils.get_spectral_constants().
    E, E_grid, w_k, S, W, I0_scalar  – all on `device`.
    Also returns B_E: (K, 3) normalised basis evaluated on E_grid.
    """
    E, E_grid, w_k, S, W, I0_scalar = get_spectral_constants()
    bfunc_kedge = BfuncKedgeSynthetic(E_grid).to(device)

    # Build B_E (K, 3) normalised basis: PE, Compton, Kedge
    # Uses the same normalisation convention as get_B_E (pivot at 70 keV)
    B_E = get_B_E(E_grid, bfunc_kedge, normalize=True, E_ref=70.0)  # (K, 3)

    return E, E_grid, w_k, S, W, I0_scalar, B_E


# ===========================================================================
# 4.  GPU cone-beam forward projector (ray-marching + trilinear grid_sample)
# ===========================================================================

def _build_ray_coords(angle: float, scanner: dict, device: str,
                      row_start: int, row_end: int):
    """
    Construct sample-point coordinates for one angle and a row slice.

    Coordinate conventions (match angle2pose in dataset_readers.py):
      - Source rotates in the xy-plane; z is the rotation axis.
      - At angle θ: source position = [DSO·cos θ, DSO·sin θ, 0]
      - Detector u-axis (columns): [-sin θ, cos θ, 0]
      - Detector v-axis (rows):    [0, 0, 1]

    Returns
    -------
    pts : (1, n_steps, n_rows, nW, 3) tensor  – sample pts in scene [-1,1]^3
    dt  : (n_rows, nW) tensor                 – step length per ray
    """
    DSO = float(scanner["DSO"])
    DSD = float(scanner["DSD"])
    nH, nW = scanner["nDetector"]
    sH, sW = float(scanner["sDetector"][0]), float(scanner["sDetector"][1])

    n_rows = row_end - row_start

    cos_a = float(np.cos(angle))
    sin_a = float(np.sin(angle))

    src = torch.tensor([DSO * cos_a, DSO * sin_a, 0.0],
                       device=device, dtype=torch.float32)

    ODD = DSD - DSO   # source-to-detector distance minus DSO = origin-to-detector
    det_centre = torch.tensor([-ODD * cos_a, -ODD * sin_a, 0.0],
                               device=device, dtype=torch.float32)

    det_u = torch.tensor([-sin_a,  cos_a, 0.0], device=device, dtype=torch.float32)
    det_v = torch.tensor([ 0.0,    0.0,   1.0], device=device, dtype=torch.float32)

    du = sW / nW   # physical pixel spacing
    dv = sH / nH

    col_idx = torch.arange(nW,          device=device, dtype=torch.float32)
    row_idx = torch.arange(row_start, row_end, device=device, dtype=torch.float32)

    u_phys = (col_idx - (nW - 1) * 0.5) * du   # (nW,)
    v_phys = (row_idx - (nH - 1) * 0.5) * dv   # (n_rows,)

    VV, UU = torch.meshgrid(v_phys, u_phys, indexing="ij")   # (n_rows, nW) each
    det_pts = (det_centre[None, None, :]
               + UU[:, :, None] * det_u[None, None, :]
               + VV[:, :, None] * det_v[None, None, :])      # (n_rows, nW, 3)

    ray_d = det_pts - src[None, None, :]   # (n_rows, nW, 3)

    # ---------- AABB slab intersection with [-1, 1]^3 --------------------
    # Avoid division by zero: perturb tiny directions away from zero
    TINY = 1e-8
    safe_d = torch.where(ray_d.abs() < TINY,
                         torch.full_like(ray_d, TINY),
                         ray_d)

    t0 = (-1.0 - src[None, None, :]) / safe_d   # (n_rows, nW, 3)
    t1 = ( 1.0 - src[None, None, :]) / safe_d

    t_near = torch.max(torch.minimum(t0, t1), dim=-1)[0].clamp(min=0.0)   # (n_rows, nW)
    t_far  = torch.min(torch.maximum(t0, t1), dim=-1)[0]

    t_len = (t_far - t_near).clamp(min=0.0)   # zero outside the bbox

    # ---------- Sample positions along each ray ---------------------------
    t_frac   = torch.linspace(0.0, 1.0, N_STEPS, device=device)           # (N_STEPS,)
    t_sample = t_near[:, :, None] + t_frac[None, None, :] * t_len[:, :, None]
    # t_sample: (n_rows, nW, N_STEPS)

    # 3-D world points  (n_rows, nW, N_STEPS, 3)
    pts = src[None, None, None, :] + t_sample[:, :, :, None] * ray_d[:, :, None, :]

    # Reshape for grid_sample: need (N=1, D_out=N_STEPS, H_out=n_rows, W_out=nW, 3)
    pts = pts.permute(2, 0, 1, 3).unsqueeze(0)   # (1, N_STEPS, n_rows, nW, 3)

    dt = t_len / N_STEPS   # (n_rows, nW) – integration step size

    return pts, dt


def cone_beam_project_3ch(vols_torch, angle: float, scanner: dict,
                           device: str = DEVICE, row_batch: int = ROW_BATCH):
    """
    Project a 3-channel volume for one angle using GPU ray-marching.

    Parameters
    ----------
    vols_torch : (1, 3, Nz, Ny, Nx) float32 tensor on device
    angle      : acquisition angle in radians
    scanner    : scanner configuration dict (physical units)
    device     : torch device string

    Returns
    -------
    line_ints : (3, nH, nW) numpy float32 array
                – line integrals of c_PE, c_Compton, c_Kedge
    """
    nH, nW = scanner["nDetector"]
    result = np.zeros((3, nH, nW), dtype=np.float32)

    for r_start in range(0, nH, row_batch):
        r_end = min(r_start + row_batch, nH)

        # Build ray coordinates for this row batch
        pts, dt = _build_ray_coords(angle, scanner, device, r_start, r_end)
        # pts: (1, N_STEPS, n_rows, nW, 3)
        # dt:  (n_rows, nW)

        # grid_sample: input (N, C, D, H, W), grid (N, D_out, H_out, W_out, 3)
        # We map: D_out=N_STEPS, H_out=n_rows, W_out=nW
        # grid coords (x,y,z) map to W=Nx, H=Ny, D=Nz of volume
        sampled = Fn.grid_sample(
            vols_torch,       # (1, 3, Nz, Ny, Nx)
            pts,              # (1, N_STEPS, n_rows, nW, 3)
            mode="bilinear",
            padding_mode="zeros",
            align_corners=True,
        )
        # sampled: (1, 3, N_STEPS, n_rows, nW)
        sampled = sampled.squeeze(0)          # (3, N_STEPS, n_rows, nW)
        sampled = sampled.permute(0, 2, 3, 1) # (3, n_rows, nW, N_STEPS)

        # Riemann integration: sum(f) * dt
        dt_t = dt.unsqueeze(0).unsqueeze(-1)  # (1, n_rows, nW, 1)
        proj_batch = (sampled * dt_t).sum(dim=-1)  # (3, n_rows, nW)

        result[:, r_start:r_end, :] = proj_batch.cpu().numpy()

    return result  # (3, nH, nW)


# ===========================================================================
# 5.  Polychromatic Beer-Lambert rendering
# ===========================================================================

def polychromatic_projection(
    line_ints_np: np.ndarray,   # (3, nH, nW): L_PE, L_Compton, L_Kedge
    B_E,                        # (K, 3) basis on energy grid
    S,                          # (C, K) spectra for each channel
    w_k,                        # (K,)  quadrature weights
    I0_scalar,                  # (C,)  incident intensities
    channel: int,               # spectral channel index
    device: str = DEVICE,
):
    """
    Compute polychromatic log-attenuation projection for one spectral channel.

    p_c = -log(I_c / I0_c)
    I_c = sum_k S_c(E_k) * exp(-mu(E_k)) * w_k
    mu(E_k) = sum_j L_j * B_j(E_k)   -- (nH, nW) map

    Returns (nH, nW) float32 numpy array.
    """
    K = B_E.shape[0]
    L = torch.from_numpy(line_ints_np).to(device)  # (3, nH, nW)

    # mu(E_k, pixel) = sum_j L_j(pixel) * B_j(E_k)
    # B_E: (K, 3)  →  reshape for broadcast
    B = B_E.to(device)     # (K, 3)
    # (K, 3) x (3, nH*nW) → (K, nH*nW)  →  (K, nH, nW)
    nH, nW = line_ints_np.shape[1], line_ints_np.shape[2]
    mu_E = torch.einsum("kc,cnm->knm", B, L)  # (K, nH, nW)

    # Polychromatic transmission
    S_c  = S[channel]         # (K,)
    w    = w_k                # (K,)
    I0_c = I0_scalar[channel] # scalar

    # I = sum_k S(E_k) * exp(-mu(E_k)) * w_k   (nH, nW)
    integrand = (S_c[:, None, None] * torch.exp(-mu_E) * w[:, None, None])
    I_pred = integrand.sum(dim=0)                      # (nH, nW)
    I_pred = I_pred.clamp(min=1e-6 * I0_c)

    p = -torch.log(I_pred / I0_c)                     # (nH, nW)
    return p.cpu().numpy().astype(np.float32)


# ===========================================================================
# 6.  Save PLY helper (matches R2-Gaussian / spectral-Gaussian conventions)
# ===========================================================================

def save_ply(path: Path, xyz: np.ndarray, density: np.ndarray):
    """Save point cloud as PLY with fields (x, y, z, density)."""
    assert xyz.shape[0] == density.shape[0]
    verts = np.zeros(xyz.shape[0],
                     dtype=[("x", "f4"), ("y", "f4"), ("z", "f4"),
                            ("density", "f4")])
    verts["x"]       = xyz[:, 0]
    verts["y"]       = xyz[:, 1]
    verts["z"]       = xyz[:, 2]
    verts["density"] = density.ravel()
    el = PlyElement.describe(verts, "vertex")
    PlyData([el], text=False).write(str(path))


# ===========================================================================
# 7.  Main generation routine
# ===========================================================================

def main():
    print(f"Device: {DEVICE}")
    print(f"Output: {OUTPUT_DIR}")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # ---- Create output sub-directories ------------------------------------
    (OUTPUT_DIR / "images").mkdir(exist_ok=True)
    (OUTPUT_DIR / "images_train").mkdir(exist_ok=True)
    (OUTPUT_DIR / "images_test").mkdir(exist_ok=True)
    for c in range(3):
        (OUTPUT_DIR / "spectra" / f"spectral_{c}").mkdir(parents=True, exist_ok=True)

    # ---- Build phantom ----------------------------------------------------
    print("Building phantom …")
    c_PE_np, c_Compton_np, c_Kedge_np = make_phantom(N=128)

    # Reference LAC at 70 keV: rho = c_PE + c_Compton + c_Kedge
    # (all normalised basis values = 1.0 at E_ref=70 keV)
    vol_gt = (c_PE_np + c_Compton_np + c_Kedge_np).astype(np.float32)
    np.save(str(OUTPUT_DIR / "vol_gt.npy"), vol_gt)
    # Also save per-basis ground-truth volumes for quantitative evaluation
    np.save(str(OUTPUT_DIR / "vol_gt_cPE.npy"),      c_PE_np)
    np.save(str(OUTPUT_DIR / "vol_gt_cCompton.npy"), c_Compton_np)
    np.save(str(OUTPUT_DIR / "vol_gt_cKedge.npy"),   c_Kedge_np)
    print(f"  vol_gt  shape={vol_gt.shape}  min={vol_gt.min():.4f}  "
          f"max={vol_gt.max():.4f}  mean={vol_gt.mean():.4f}")

    # ---- Build spectral model --------------------------------------------
    print("Building spectral model …")
    E, E_grid, w_k, S, W, I0_scalar, B_E = build_spectral_model(DEVICE)
    print(f"  E channels: {E.cpu().numpy()}")
    print(f"  Energy grid: {E_grid.shape[0]} bins  [{E_grid.min().item():.1f}, "
          f"{E_grid.max().item():.1f}] keV")
    print(f"  I0_scalar: {I0_scalar.cpu().numpy()}")

    # ---- Prepare volume tensor for GPU grid_sample -----------------------
    # Storage: c[iz, iy, ix]  → torch (1, C, Nz, Ny, Nx) = (N, C, D, H, W)
    vols_np = np.stack([c_PE_np, c_Compton_np, c_Kedge_np], axis=0)  # (3, Nz, Ny, Nx)
    vols_t  = torch.from_numpy(vols_np).unsqueeze(0).float().to(DEVICE)  # (1,3,Nz,Ny,Nx)

    # ---- Acquisition angles ----------------------------------------------
    total_angle_rad = np.deg2rad(SCANNER["totalAngle"])
    # Angles identical to quail3: np.linspace(0, total_angle_rad, N_ANGLES)
    all_angles = np.linspace(0.0, total_angle_rad, N_ANGLES, dtype=np.float64)

    # Training cameras: 75 evenly spaced indices into the 720 views
    train_idx = np.round(np.linspace(0, N_ANGLES - 1, N_TRAIN)).astype(int)

    # Test views: 100 views NOT in train set, evenly spaced from the remaining
    non_train_idx = np.array([i for i in range(N_ANGLES) if i not in set(train_idx)])
    test_idx = non_train_idx[
        np.round(np.linspace(0, len(non_train_idx) - 1, N_TEST)).astype(int)
    ]

    train_set = set(train_idx.tolist())
    test_set  = set(test_idx.tolist())

    print(f"\nAcquisition angles: {N_ANGLES}  |  train: {N_TRAIN}  |  test: {N_TEST}")

    # ---- Project all angles ---------------------------------------------
    print("\nGenerating projections …")
    max_proj = [0.0] * 3  # track per-channel max for sanity check

    for a_idx in range(N_ANGLES):
        angle = all_angles[a_idx]
        fname = f"{a_idx + 1:04d}.npy"

        if a_idx % 72 == 0:
            print(f"  angle {a_idx + 1}/{N_ANGLES}  ({np.rad2deg(angle):.1f}°)")

        # Compute line integrals for all 3 basis channels simultaneously
        line_ints = cone_beam_project_3ch(vols_t, float(angle), SCANNER, DEVICE, ROW_BATCH)
        # line_ints: (3, nH, nW) = [L_PE, L_Compton, L_Kedge]

        # Polychromatic projection per spectral channel
        projs = []
        for c in range(3):
            p = polychromatic_projection(line_ints, B_E, S, w_k, I0_scalar, c, DEVICE)
            projs.append(p)
            max_proj[c] = max(max_proj[c], float(p.max()))
            np.save(str(OUTPUT_DIR / "spectra" / f"spectral_{c}" / fname), p)

        # images/ = spectral_0 (matches quail3 convention)
        np.save(str(OUTPUT_DIR / "images" / fname), projs[0])

        # images_train / images_test
        if a_idx in train_set:
            np.save(str(OUTPUT_DIR / "images_train" / fname), projs[0])
        elif a_idx in test_set:
            np.save(str(OUTPUT_DIR / "images_test" / fname), projs[0])

    print(f"\nPeak log-attenuation per channel:")
    for c in range(3):
        print(f"  channel {c}: {max_proj[c]:.4f}")

    # ---- Sample init_pcd from the volume ---------------------------------
    print("\nSampling init_pcd …")
    Nz, Ny, Nx = vol_gt.shape
    rho_flat = vol_gt.ravel()
    nz, ny, nx = np.meshgrid(
        np.arange(Nz, dtype=np.float32),
        np.arange(Ny, dtype=np.float32),
        np.arange(Nx, dtype=np.float32),
        indexing="ij",
    )
    # Scene coordinates: centre of voxel (iz, iy, ix) → (x, y, z) in [-1, 1]
    x_coords = ( (nx.ravel() + 0.5) / Nx * 2.0 - 1.0 ).astype(np.float32)
    y_coords = ( (ny.ravel() + 0.5) / Ny * 2.0 - 1.0 ).astype(np.float32)
    z_coords = ( (nz.ravel() + 0.5) / Nz * 2.0 - 1.0 ).astype(np.float32)

    # Sample with probability proportional to rho (denser sampling in
    # high-attenuation regions, same as R2-Gaussian strategy)
    probs = rho_flat.copy()
    probs_sum = probs.sum()
    if probs_sum > 0:
        probs = probs / probs_sum
    else:
        probs = np.ones_like(probs) / len(probs)

    chosen = np.random.choice(len(probs), size=N_INIT_PCD, replace=True, p=probs)

    xyz  = np.stack([x_coords[chosen], y_coords[chosen], z_coords[chosen]], axis=1)
    dens = rho_flat[chosen][:, None]

    pcd_data = np.concatenate([xyz, dens], axis=1).astype(np.float64)  # (50000, 4)
    np.save(str(OUTPUT_DIR / "init_pcd.npy"), pcd_data)
    print(f"  init_pcd  shape={pcd_data.shape}  "
          f"density range: [{dens.min():.4f}, {dens.max():.4f}]")

    # Save PLY
    save_ply(OUTPUT_DIR / "init_pcd.ply", xyz, dens.ravel())
    print(f"  init_pcd.ply saved.")

    # ---- Write meta_data.json -------------------------------------------
    print("\nWriting meta_data.json …")
    cameras = []
    for idx in train_idx:
        cameras.append({
            "file_path": f"{idx + 1:04d}.npy",
            "angle":     float(all_angles[idx]),
        })

    meta = {
        "scanner": SCANNER,
        "ct":      "vol_gt.npy",
        "radius":  1.0,
        "bbox":    [[-1.0, -1.0, -1.0], [1.0, 1.0, 1.0]],
        "cameras": cameras,
    }
    with open(str(OUTPUT_DIR / "meta_data.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"  {len(cameras)} training cameras written.")

    # ---- Final summary ---------------------------------------------------
    n_train_files = len(list((OUTPUT_DIR / "images_train").iterdir()))
    n_test_files  = len(list((OUTPUT_DIR / "images_test").iterdir()))
    n_spec0_files = len(list((OUTPUT_DIR / "spectra" / "spectral_0").iterdir()))

    print("\n" + "=" * 60)
    print(f"Dataset written to: {OUTPUT_DIR}")
    print(f"  images/        : {len(list((OUTPUT_DIR/'images').iterdir()))} files")
    print(f"  images_train/  : {n_train_files} files")
    print(f"  images_test/   : {n_test_files} files")
    print(f"  spectra/spectral_0/ : {n_spec0_files} files")
    print(f"  vol_gt.npy     : {vol_gt.shape}")
    print(f"  init_pcd.npy   : {pcd_data.shape}")
    print("=" * 60)
    print("Done.")


if __name__ == "__main__":
    main()
