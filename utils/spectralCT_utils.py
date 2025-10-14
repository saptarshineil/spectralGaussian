import numpy as np
import torch
import spekpy
from utils.general_utils import interp1d


C = 510.999
EPSILON = 1e-12


# ------------------------ Fraction initialization ------------------------
def pe_compton_from_density(rho, q_lo=0.01, q_hi=0.99, eps=1e-3):
    """
    Monotone logistic prior: high rho -> high PE fraction.
    q_lo,q_hi set where f_PE hits ~0.1 and ~0.9, respectively.
    """
    rho = np.asarray(rho, np.float32).reshape(-1)
    # pick where the transition lives, from dataset statistics
    rho_l = np.quantile(rho, q_lo)
    rho_h = np.quantile(rho, q_hi)
    rho_l = max(rho_l, eps)
    rho_h = max(rho_h, rho_l + 1e-6)

    # logistic params in log-space: f = sigmoid(alpha*(log(rho)-log(rho0)))
    log_rho0 = 0.5*(np.log(rho_l) + np.log(rho_h))
    alpha = 2.0*np.log(9.0) / (np.log(rho_h) - np.log(rho_l) + 1e-12)  # 10%->90% width

    t = alpha*(np.log(np.clip(rho, eps, None)) - log_rho0)
    f_pe = 1.0 / (1.0 + np.exp(-t))
    f_pe = np.clip(f_pe, eps, 1.0 - eps)

    f = np.stack([f_pe, 1.0 - f_pe], axis=-1).astype(np.float32)
    rho0 = float(np.exp(log_rho0))
    return f, rho0, float(alpha)

# def pe_comp_kedge_from_density(rho, q_lo=0.20, q_hi=0.80, rho0_edge=2.4, alpha_edge=10.0, k_frac_max=0.60, eps=1e-3):
# # def pe_comp_kedge_from_density(rho, q_lo=0.01, q_hi=0.99, rho0_edge=2.4, alpha_edge=10.0, k_frac_max=0.60, eps=1e-3):
#     """
#     Monotone logistic prior: high rho -> high PE fraction.
#     q_lo,q_hi set where f_PE hits ~0.1 and ~0.9, respectively.
#     """
#     rho = np.asarray(rho, np.float32).reshape(-1)
#     # pick where the transition lives, from dataset statistics
#     rho_l = np.quantile(rho, q_lo)
#     rho_h = np.quantile(rho, q_hi)
#     rho_l = max(rho_l, eps)
#     rho_h = max(rho_h, rho_l + 1e-6)

#     # logistic params in log-space: f = sigmoid(alpha*(log(rho)-log(rho0)))
#     log_rho0 = 0.5*(np.log(rho_l) + np.log(rho_h))
#     alpha = 2.0*np.log(9.0) / (np.log(rho_h) - np.log(rho_l) + 1e-12)  # 10%->90% width

#     t = alpha*(np.log(np.clip(rho, eps, None)) - log_rho0)
#     f_pe_tot = 1.0 / (1.0 + np.exp(-t))
#     f_pe_tot = np.clip(f_pe_tot, eps, 1.0 - eps)

#     f_comp    = 1. - f_pe_tot                                   # rest → Compton

#     # ── 2.  decide how much of that PE budget is actually an edge ─
#     gate_edge = 1. / (1. + np.exp(-alpha_edge * (rho - rho0_edge)))
#     f_k       = k_frac_max * gate_edge * f_pe_tot               # ≤ k_frac_max·f_pe_tot
#     f_pe      = f_pe_tot - f_k

#     # ── 3.  stack & renormalise (paranoia)  ─────────────────────
#     frac = np.stack([f_pe, f_comp, f_k], axis=-1)
#     frac = np.clip(frac, eps, None)
#     frac /= frac.sum(axis=-1, keepdims=True)

#     rho0 = float(np.exp(log_rho0))
#     return frac, rho0, float(alpha)

# def pe_comp_kedge_from_density(
#     rho,
#     q_lo=0.20,
#     q_hi=0.80,
#     rho0_edge="q95",        # data-driven threshold for K gate (string "qXX" or float)
#     alpha_edge=12.0,        # steeper gate than before
#     # k_frac_max=0.99,        # allow more K (but keep <1.0)
#     k_frac_max=0.60,        # allow more K (but keep <1.0)
#     pe_shift=1.25,          # shift logistic toward higher PE fraction
#     pe_slope=10.15,          # steepen logistic slope for PE
#     eps=1e-3
# ):
#     """
#     Initialize fractions for (PE, Compton, K-edge) from density.

#     - Monotone logistic prior: high rho -> high PE fraction.
#     - K-edge fraction carved out of PE fraction, with data-driven gating.
#     - Parameters (pe_shift, pe_slope, k_frac_max) let you bias toward more K.
#     """

#     rho = np.asarray(rho, np.float32).reshape(-1)

#     # --- 1. logistic for PE vs Compton ----------------------------
#     rho_l = max(np.quantile(rho, q_lo), eps)
#     rho_h = max(np.quantile(rho, q_hi), rho_l + 1e-6)

#     log_rho0 = 0.5 * (np.log(rho_l) + np.log(rho_h))
#     alpha = 2.0 * np.log(9.0) / (np.log(rho_h) - np.log(rho_l) + 1e-12)

#     # bias PE upward: earlier transition and a bit steeper
#     log_rho0 -= np.log(pe_shift)
#     alpha *= pe_slope

#     t = alpha * (np.log(np.clip(rho, eps, None)) - log_rho0)
#     f_pe_tot = 1.0 / (1.0 + np.exp(-t))
#     f_pe_tot = np.clip(f_pe_tot, eps, 1.0 - eps)

#     f_comp = 1.0 - f_pe_tot

#     # --- 2. carve out a K-edge component from PE ------------------
#     # resolve rho0_edge if it's a string like "q95"
#     if isinstance(rho0_edge, str) and rho0_edge.startswith("q"):
#         q = float(rho0_edge[1:]) / 100.0
#         rho0_edge = float(np.quantile(rho, q))

#     gate_edge = 1.0 / (1.0 + np.exp(-alpha_edge * (rho - rho0_edge)))

#     f_k = k_frac_max * gate_edge * f_pe_tot
#     f_pe = f_pe_tot - f_k

#     # --- 3. stack & renormalise ----------------------------------
#     frac = np.stack([f_pe, f_comp, f_k], axis=-1)
#     frac = np.clip(frac, eps, None)
#     frac /= frac.sum(axis=-1, keepdims=True)

#     rho0 = float(np.exp(log_rho0))
#     return frac, rho0, float(alpha)


# 1
def pe_comp_kedge_from_density(
    rho,
    *,
    xyz,                        # (N,3) REQUIRED: used in a generic, dataset-agnostic way
    # --- PE vs Compton (logistic on log-rho) ---
    q_lo=0.01,
    q_hi=0.99,
    pe_shift=1.15,
    pe_slope=1.10,
    # --- K (outer shell) gate, geometry-agnostic ---
    shell_q=0.90,              # global Mahalanobis-radius quantile for “outer shell”
    # shell_q=0.70,              # global Mahalanobis-radius quantile for “outer shell”
    alpha_shell=14.0,          # steepness of the shell gate
    # optional local equalization to avoid one-sided yellow
    use_local_rank=True,
    local_k=48,
    local_q=0.90,
    local_alpha=10.0,
    # --- safety / regularizers ---
    rho0_edge="q95",
    alpha_edge=12.0,
    top_p=0.0,                 # 0 disables hard cap
    k_frac_max=0.60,
    eps=1e-3
):
    """
    Initialize (PE, Compton, K-edge) from density and position without any
    dataset-specific plane/cylinder assumptions.

    Idea:
      • PE/Compton: logistic on log(rho).
      • K-edge: part of PE but only where the Mahalanobis radius is large
        (outer shell); optional kNN local equalization for uniformity.
    Returns: frac (N,3) in (PE, Compton, K), rho0, alpha
    """
    rho = np.asarray(rho, np.float32).reshape(-1)
    pts = np.asarray(xyz, np.float32).reshape(-1, 3)
    N   = rho.size

    # ---------- 1) PE vs Compton on log-ρ ----------
    rho_l = np.quantile(rho, q_lo)
    rho_h = np.quantile(rho, q_hi)
    rho_l = max(rho_l, eps)
    rho_h = max(rho_h, rho_l + 1e-6)

    # logistic params in log-space: f = sigmoid(alpha*(log(rho)-log(rho0)))
    log_rho0 = 0.5*(np.log(rho_l) + np.log(rho_h))
    alpha = 2.0*np.log(9.0) / (np.log(rho_h) - np.log(rho_l) + 1e-12)  # 10%->90% width

    t = alpha*(np.log(np.clip(rho, eps, None)) - log_rho0)
    f_pe_tot = 1.0 / (1.0 + np.exp(-t))
    f_pe_tot = np.clip(f_pe_tot, eps, 1.0 - eps)
    f_comp   = 1.0 - f_pe_tot

    # ---------- 2) Robust center + whitening (Mahalanobis radius) ----------
    # weight center/cov by rho so the soft/bone band drives the fit
    r50, r90 = np.quantile(rho, [0.50, 0.90])
    w = np.clip((rho - r50) / (r90 - r50 + 1e-12), 0.0, 1.0)  # robust, in [0,1]
    w = (w + 1e-6)  # avoid zeros

    c = (w[:, None] * pts).sum(axis=0) / (w.sum() + 1e-12)
    X = pts - c[None, :]

    # weighted covariance with shrinkage for stability
    Xw = X * np.sqrt(w[:, None])
    cov = (Xw.T @ Xw) / (w.sum() + 1e-12)
    tr  = np.trace(cov)
    cov += 1e-3 * (tr / 3.0 + 1e-9) * np.eye(3, dtype=np.float32)

    # whiten: Y = X @ W, with W = U diag(1/sqrt(s)) U^T
    s, U = np.linalg.eigh(cov)
    s = np.clip(s, 1e-9, None)
    Winv_sqrt = (U / np.sqrt(s))[..., :] @ U.T     # (3,3)
    Y = X @ Winv_sqrt.T
    rM = np.linalg.norm(Y, axis=1)                 # Mahalanobis radius

    # global outer-shell gate
    r_thr = np.quantile(rM, shell_q)
    g_shell = 1.0 / (1.0 + np.exp(-alpha_shell * ((rM - r_thr) / (r_thr + 1e-9))))

    # ---------- 3) Local rank equalization (optional) ----------
    # if use_local_rank:
    #     try:
    #         from scipy.spatial import cKDTree
    #         tree = cKDTree(Y.astype(np.float32))
    #         # include self in k, then ignore first column when computing quantile
    #         k = min(local_k + 1, max(2, N))
    #         _, nn_idx = tree.query(Y, k=k)
    #         # gather neighbors’ rM and take per-row quantile (ignoring self at [:,0])
    #         neigh_r = rM[nn_idx]
    #         if neigh_r.ndim == 1:  # N==1 edge case
    #             qloc = np.full_like(rM, r_thr, dtype=np.float32)
    #         else:
    #             qloc = np.quantile(neigh_r[:, 1:], local_q, axis=1)
    #         g_loc = 1.0 / (1.0 + np.exp(-local_alpha * ((rM - qloc) / (qloc + 1e-9))))
    #     except Exception:
    #         # fallback: no local equalization
    #         g_loc = 1.0
    # else:
    #     g_loc = 1.0

    # small density safety gate to suppress obvious air/background
    if isinstance(rho0_edge, str) and rho0_edge.startswith("q"):
        q = float(rho0_edge[1:]) / 100.0
        rho0_edge = float(np.quantile(rho, q))
    g_rho = 1.0 / (1.0 + np.exp(-alpha_edge * (rho - float(rho0_edge))))

    # g = g_shell * g_loc * g_rho
    g = g_shell * g_rho

    # # optional hard cap by largest radii
    # if top_p and top_p > 0.0:
    #     kk = max(1, int(top_p * N))
    #     idx = np.argpartition(rM, -kk)[-kk:]
    #     hard = np.zeros_like(rM, dtype=np.float32); hard[idx] = 1.0
    #     g *= hard

    # ---------- 4) Carve K out of PE and normalize ----------
    f_k  = k_frac_max * g * f_pe_tot        # K is a part of PE
    f_pe = f_pe_tot - f_k

    frac = np.stack([f_pe, f_comp, f_k], axis=-1).astype(np.float32)
    frac = np.clip(frac, eps, None)
    frac /= frac.sum(axis=-1, keepdims=True)
    return frac, float(np.exp(log_rho0)), float(alpha)


# 2
# def pe_comp_kedge_from_density(
#     rho,
#     q_lo=0.02,
#     q_hi=0.98,
#     rho0_edge="q95",
#     alpha_edge=12.0,
#     k_frac_max=0.90,
#     pe_shift=1.15,
#     pe_slope=1.10,
#     eps=1e-3,
#     *,
#     xyz=None,
#     shell_q=0.90,
#     shell_alpha=18.0,
#     top_p=0.01
# ):
#     """
#     Initialize (PE, Compton, K-edge) from rho (+ geometry).
#       1) PE vs Compton: logistic on log(rho).
#       2) K-edge: carved from PE_total but gated by an OUTER-SHELL prior
#          (large radius from robust center). Optional ρ gate/top-p retained
#          only as safeguards.
#     """

#     rho = np.asarray(rho, np.float32).reshape(-1)
#     N   = rho.size
#     lr  = np.log(np.clip(rho, eps, None))

#     # ---- 1) PE vs Compton on log-ρ (well-conditioned) -------------------------
#     r_lo = max(np.quantile(rho, q_lo), eps)
#     r_hi = max(np.quantile(rho, q_hi), r_lo + 1e-6)
#     log_rho0 = 0.5 * (np.log(r_lo) + np.log(r_hi))
#     alpha    = 2.0 * np.log(9.0) / (np.log(r_hi) - np.log(r_lo) + 1e-12)
#     log_rho0 -= np.log(pe_shift)
#     alpha    *= pe_slope

#     t        = alpha * (lr - log_rho0)
#     f_pe_tot = 1.0 / (1.0 + np.exp(-t))
#     f_pe_tot = np.clip(f_pe_tot, eps, 1.0 - eps)
#     f_comp   = 1.0 - f_pe_tot

#     # ---- 2) OUTER-SHELL gate (swaps yellow to the outside) --------------------
#     # Robust center (median) and radius
#     if xyz is None:
#         raise ValueError("pe_comp_kedge_from_density(...): xyz is required for shell gating.")
#     pts   = np.asarray(xyz, np.float32).reshape(N, 3)
#     c_med = np.median(pts, axis=0)
#     r     = np.linalg.norm(pts - c_med[None, :], axis=1)        # Euclidean radius
#     r_thr = np.quantile(r, shell_q)
#     # steep logistic that turns on only near/after r_thr
#     g_shell = 1.0 / (1.0 + np.exp(-shell_alpha * ((r - r_thr) / (r_thr + 1e-9))))

#     # Optional small ρ gate to prevent K in obvious air/background
#     if isinstance(rho0_edge, str) and rho0_edge.startswith("q"):
#         q = float(rho0_edge[1:]) / 100.0
#         rho0_edge = float(np.quantile(rho, q))
#     g_rho = 1.0 / (1.0 + np.exp(-alpha_edge * (rho - float(rho0_edge))))

#     g = g_shell * g_rho

#     # Optional hard cap: only top-p by radius can carry K at init
#     if top_p and top_p > 0.0:
#         k = max(1, int(top_p * N))
#         idx = np.argpartition(r, -k)[-k:]
#         hard = np.zeros_like(r, dtype=np.float32)
#         hard[idx] = 1.0
#         g *= hard

#     # ---- 3) carve K from PE and renormalize -----------------------------------
#     f_k  = k_frac_max * g * f_pe_tot        # physics: K is a part of PE
#     f_pe = f_pe_tot - f_k

#     frac = np.stack([f_pe, f_comp, f_k], axis=-1).astype(np.float32)
#     frac = np.clip(frac, eps, None)
#     frac /= frac.sum(axis=-1, keepdims=True)
#     return frac, float(np.exp(log_rho0)), float(alpha)


    


# ------------------------ Photoelectric + Compton basis functions ------------------------
def bfunc_PE(E_keV: torch.Tensor) -> torch.Tensor:
    E_keV = torch.as_tensor(E_keV, dtype=torch.float32, device=E_keV.device)
    # return torch.pow(E_keV, -3.5)
    return torch.pow(E_keV, -3.0)

def bfunc_Compton(E_keV: torch.Tensor,
                  eps: float = EPSILON) -> torch.Tensor:
    E_keV = torch.as_tensor(E_keV, dtype=torch.float32, device=E_keV.device)
    
    # Energy ratio
    a = torch.clamp(E_keV / C, min=eps)

    denom_a      = a + eps
    denom_a2     = a.pow(2) + eps
    denom_2a     = 2.0 * a + eps
    denom_1_2a   = 1.0 + denom_2a
    denom_1_2a2  = denom_1_2a.pow(2) + eps

    # Low‑frequency component
    term1 = (1.0 + a) / denom_a2

    # High‑frequency
    term2 = 2.0 * (1.0 + a) / denom_1_2a \
            - torch.log1p(denom_2a - eps) / denom_a

    term3 = torch.log1p(denom_2a - eps) / denom_2a
    term4 = (1.0 + 3.0 * a) / denom_1_2a2

    return (term1 * term2 + term3 - term4)

# def _col_scales_at_pivot(E_grid: torch.Tensor, bfunc_Kedge, E_ref: float):
#     eref = torch.as_tensor(E_ref, dtype=E_grid.dtype, device=E_grid.device)
#     s_pe = bfunc_PE(eref).clamp(min=1e-30)
#     s_co = bfunc_Compton(eref).clamp(min=1e-30)
#     scales = [s_pe, s_co]
#     if bfunc_Kedge is not None:
#         scales.append(torch.tensor(1.0, device=E_grid.device, dtype=E_grid.dtype))
#     return torch.stack(scales)

def _col_scales_at_pivot(E_grid: torch.Tensor, bfunc_Kedge, E_ref: float):
    eref = torch.as_tensor(E_ref, dtype=E_grid.dtype, device=E_grid.device)
    s_pe = bfunc_PE(eref).clamp(min=1e-30)
    s_co = bfunc_Compton(eref).clamp(min=1e-30)
    scales = [s_pe, s_co]
    if bfunc_Kedge is not None:
        # evaluate K-edge at E_ref by interpolating its E_grid output
        s_k = interp1d(E_grid, bfunc_Kedge(), eref).clamp(min=1e-30)
        scales.append(s_k)
    return torch.stack(scales)


def get_B_E(E_grid, bfunc_Kedge, *, normalize=True, E_ref: float = 70.0):
    B = (torch.stack([bfunc_PE(E_grid), bfunc_Compton(E_grid)], dim=1)
         if bfunc_Kedge is None else
         torch.stack([bfunc_PE(E_grid), bfunc_Compton(E_grid), bfunc_Kedge()], dim=1))
    if normalize:
        s = _col_scales_at_pivot(E_grid, bfunc_Kedge, E_ref)  # (C,)
        B = B / s[None, :]
    return B

def get_B_E_at_energy(E_grid, bfunc_Kedge, E, *, normalize=True, E_ref: float = 70.0):
    B1 = (torch.stack([bfunc_PE(E), bfunc_Compton(E)], dim=-1)
          if bfunc_Kedge is None else
          torch.stack([bfunc_PE(E), bfunc_Compton(E), interp1d(E_grid, bfunc_Kedge(), E)], dim=-1))
    if normalize:
        s = _col_scales_at_pivot(E_grid, bfunc_Kedge, E_ref)
        B1 = B1 / s
    return B1


# ------------------------------------------------------------------------------------------------
def get_spectral_constants(E_grid=None, w_k=None):
    # E = torch.tensor([50.0, 70.0], dtype=torch.float32).cuda()
    # E = torch.tensor([50.0, 120.0], device='cuda', dtype=torch.float32)
    E = torch.tensor([50.0, 80.0, 120.0], device='cuda', dtype=torch.float32)
    # E = torch.tensor([50.0, 80.0], device='cuda', dtype=torch.float32)

    I0_low = 1136.8784583333334
    I0_mid = 1594.836
    I0_high = 2026.8482152777776

    if E_grid is None and w_k is None:
        N_samples = 64
        # If dataset is EID (no bins) --> sample full spectrum
        E_grid, w_k = energy_quadrature(20.0, 140.0, N_samples)
    E_grid_np = E_grid.cpu().numpy()
    w_k_np = w_k.cpu().numpy()

    S_low  = get_scaled_spectrum(50, None, 0.0, I0_low, E_grid_np, w_k_np)
    S_mid  = get_scaled_spectrum(80, "Al", 1.0, I0_mid, E_grid_np, w_k_np)
    S_high = get_scaled_spectrum(120, "Cu", 0.5, I0_high, E_grid_np, w_k_np)

    # S         = torch.stack([S_low, S_high])      # (C=2, K)
    S         = torch.stack([S_low, S_mid, S_high])      # (C=2, K)
    W = (w_k * S.sum(dim=0)).clamp_min(1e-12)
    I0_scalar = torch.sum(S * w_k, dim=1)

    return E, E_grid, w_k, S, W, I0_scalar

def energy_quadrature(E_min, E_max, N, *, device="cuda", dtype=torch.float32):
    dE   = (E_max - E_min) / N                       # scalar
    E_k  = torch.linspace(E_min + 0.5*dE,            # mid-points
        E_max - 0.5*dE,
        steps=N,
        device=device,
        dtype=dtype
    )               # (K,)
    w_k  = torch.full((N,), dE, device=device, dtype=dtype)  # (K,)
    return E_k, w_k

# def energy_quadrature(E_min, E_max, N, *, device="cuda", dtype=torch.float32):
#     e0 = torch.tensor(E_min, device=device, dtype=dtype)
#     e1 = torch.tensor(E_max, device=device, dtype=dtype)
#     edges = torch.logspace(torch.log10(e0), torch.log10(e1), steps=N+1, device=device, dtype=dtype)
#     E_k = 0.5*(edges[:-1] + edges[1:])
#     w_k = (edges[1:] - edges[:-1])
#     return E_k, w_k

def get_scaled_spectrum(kvp, filt_mat, filt_thick, I0, E_grid_np, w_k_np):
    spek = spekpy.Spek(kvp=kvp, th=12)
    if filt_mat is not None:
        spek.filter(filt_mat, filt_thick)
    E_native, S_native = spek.get_spectrum()
    S = np.interp(E_grid_np, E_native, S_native)
    S /= np.sum(S * w_k_np) + 1e-12
    return torch.tensor(S*I0, device='cuda', dtype=torch.float32)


# def render_image(bcoeff_image, bfunc_Kedge, E, E_grid, w_k, S, W, I0_scalar, n_bcoeffs, spectral_id, log_tilt):
# def render_image(bcoeff_image, bfunc_Kedge, E, E_grid, w_k, S, W, I0_scalar, n_bcoeffs, spectral_id, scatter_frac):
def render_image(bcoeff_image, bfunc_Kedge, E, E_grid, w_k, S, W, I0_scalar, n_bcoeffs, spectral_id):
    # B_E_unnorm = get_B_E(E_grid, bfunc_Kedge if n_bcoeffs == 3 else None, normalize=False) 
    # if n_bcoeffs == 3: 
    #     S_sum = S.sum(dim=0) # (K,) 
    #     W = (w_k * S_sum).clamp_min(1e-12)
    #     B12 = B_E_unnorm[:, :2] # (K,2) 
    #     b3 = B_E_unnorm[:, 2:3] # (K,1) 
    #     BW = B12 * W[:, None] 
    #     M = BW.T @ B12 + 1e-6 * torch.eye(2, device=B12.device, dtype=B12.dtype) 
    #     rhs = BW.T @ b3 
    #     c = torch.cholesky_solve(rhs, torch.linalg.cholesky(M)) # (2,1) # Orthogonalized columns (still UNNORMALIZED) 
    #     B_E_ortho = torch.cat([B12, b3 - B12 @ c], dim=1) # Now normalize *all* columns at the same pivot 
    #     # s_pe = renormalize_col_at_pivot(E_grid, B_E_ortho, 0, E_ref=70.0) 
    #     # s_co = renormalize_col_at_pivot(E_grid, B_E_ortho, 1, E_ref=70.0) 
    #     # s_ke = renormalize_col_at_pivot(E_grid, B_E_ortho, 2, E_ref=70.0) 
    #     B_E, s = renormalize_col_at_pivot(E_grid, B_E_ortho, E_ref=70.0)
    #     s_pe, s_co, s_ke = s.unbind()
    #     # B_E = B_E_ortho 
    # else: # 2-basis: just normalize at the pivot 
    #     B_E = get_B_E(E_grid, None, normalize=True, E_ref=70.0)

    # if n_bcoeffs == 3:
    #     B1 = torch.stack([
    #         bfunc_PE(E[spectral_id]),
    #         bfunc_Compton(E[spectral_id]),
    #         interp1d(E_grid, bfunc_Kedge(), E[spectral_id])
    #     ], dim=0)                                          # (3,)

    #     # project K column like we did for B_E:
    #     B12_E = B1[:2]                                     # (2,)
    #     b3_E  = B1[2:3]                                    # (1,)
    #     b3o_E = b3_E - (B12_E @ c).view(1)                 # (1,)

    #     # normalize by the same pivot scales we captured above
    #     bfuncs = torch.stack([B1[0]/s_pe, B1[1]/s_co, b3o_E.squeeze()/s_ke], dim=0)
    # else:
    #     bfuncs = get_B_E_at_energy(E_grid, None, E[spectral_id])

    if n_bcoeffs == 3:
        B_raw = get_B_E(E_grid, bfunc_Kedge, normalize=False)
        B_raw = orthogonalize_k_to_pe_compton(B_raw, W)
        B_E, _ = renormalize_col_at_pivot(E_grid, B_raw, E_ref=70.0)
        # bfuncs = get_B_E_at_energy(E_grid, bfunc_Kedge, E[spectral_id], normalize=True)
        bfuncs = torch.stack([interp1d(E_grid, B_E[:, j], E[spectral_id]) for j in range(B_E.shape[1])], dim=0)
    else:
        B_E = get_B_E(E_grid, None, normalize=True)
        bfuncs = get_B_E_at_energy(E_grid, None, E[spectral_id], normalize=True)

    μ_E = (bcoeff_image.unsqueeze(0) * B_E[..., None, None]).sum(1)

    I_pred = torch.sum(S[spectral_id,:,None,None] *
                    torch.exp(-μ_E) * w_k[:,None,None],
                    dim=0)
    # I_pred = torch.sum(S[spectral_id,:,None,None].double() * torch.exp(-μ_E) * w_k[:,None,None].double(), dim=0).float()

    # E0 = 70.0
    # tilt = torch.exp(log_tilt[spectral_id]).double()         # ≈ 1 initially
    # tilt_vec = torch.exp( (E_grid.double()-E0) * (tilt-1.0) / 120.0 )
    # I_pred = torch.sum((S[spectral_id]*tilt_vec).unsqueeze(-1).unsqueeze(-1) *
    #                    torch.exp(-μ_E) * w_k[:,None,None].double(), dim=0).float()

    # I_pred = torch.sum(S[spectral_id,:,None,None].double() * torch.exp(-μ_E) * w_k[:,None,None], dim=0)
    # I_pred = I_pred + torch.nn.functional.softplus(scatter_frac[spectral_id]) * I0_scalar[spectral_id]

    I_pred = torch.clamp_min(I_pred, 1e-6 * I0_scalar[spectral_id])
    image = -torch.log(I_pred / I0_scalar[spectral_id]).unsqueeze(0)
    # image = (bcoeff_image * bfuncs[:, None, None]).sum(dim=0, keepdim=True)
    return image, bfuncs

def orthogonalize_k_to_pe_compton(B_raw: torch.Tensor, W: torch.Tensor) -> torch.Tensor:
    if B_raw.shape[1] < 3:
        return B_raw

    B12 = B_raw[:, :2]         # (K, 2)
    b3  = B_raw[:, 2:3]        # (K, 1)

    # M = B12^T W B12, rhs = B12^T W b3
    BW  = B12 * W[:, None]
    M   = BW.T @ B12 + 1e-6 * torch.eye(2, device=B_raw.device, dtype=B_raw.dtype)
    rhs = BW.T @ b3

    # Solve M c = rhs, project out: b3_orth = b3 - B12 c
    c   = torch.cholesky_solve(rhs, torch.linalg.cholesky(M))
    b3o = b3 - B12 @ c

    # optional: renormalize at pivot (handled by get_B_E via _col_scales_at_pivot)
    Bout = torch.cat([B12, b3o], dim=1)
    return Bout

def renormalize_col_at_pivot(E_grid, B, E_ref=70.0, eps=1e-12):
    eref = torch.as_tensor(E_ref, device=B.device, dtype=B.dtype)
    scales = []
    cols = []
    for j in range(B.shape[1]):
        col  = B[:, j]
        val  = interp1d(E_grid, col, eref).clamp_min(eps)  # depends on col
        cols.append(col / val)                             # out-of-place
        scales.append(val)
    B_out = torch.stack(cols, dim=1)
    s = torch.stack(scales)                                # (C,)
    return B_out, s


# ------------------------ Softmax ------------------------
def stable_simplex(
    logits: torch.Tensor,
    tau: float = 1.0,
    eps: float = 0.60,
    clip: float = 30.0,
    target: float = 0.5,     # kept for C>=3 path
    norm: str = "std",       # kept for C>=3 path
    s_ref_binary: float = 10.0,  # NEW: fixed scale for C=2
) -> torch.Tensor:
    """
    Map unconstrained logits -> probabilities with:
      • centering; 
      • for C==2: fixed scaling (avoids zero-gradient trap);
      • for C>=3: gentle row-wise scaling to avoid saturation;
      • temperature tau;
      • clipping;
      • epsilon floor via convex mix with uniform.

    Returns a tensor with last-dim summing to 1.
    """
    z = logits - logits.mean(dim=-1, keepdim=True)
    k = z.shape[-1]

    if k == 2:
        # ---- binary-safe path: constant scale -> nonzero ∂p/∂logits ----
        z = z / (max(tau, 1e-6) * s_ref_binary)
    else:
        # ---- multi-class path: mild normalization to avoid saturation ----
        if norm == "std":
            s = z.std(dim=-1, keepdim=True, unbiased=True).clamp_min(1e-6)
        elif norm == "max":
            s = z.abs().amax(dim=-1, keepdim=True).clamp_min(1e-6)
        else:
            s = torch.ones_like(z[..., :1])

        scale = torch.maximum(s / target, torch.ones_like(s))
        z = z / (scale * max(tau, 1e-6))

    z = z.clamp(min=-clip, max=clip)
    p_soft = torch.softmax(z, dim=-1)

    if eps > 0.0:
        u = p_soft.new_full(p_soft.shape, 1.0 / k)
        p = (1.0 - eps) * p_soft + eps * u
    else:
        p = p_soft
    return p







# ------------------------ Learned K-edge basis function ------------------------
class BFuncKedge(torch.nn.Module):
    """
    Smooth, positive K-edge basis:
        Bθ(E) = α · b_PE(E) · σ(γ·(E − EK))
    where b_PE(E)=E^{-3} and σ is the logistic sigmoid.
    Columns are normalised at the pivot in get_B_E()/get_B_E_at_energy().
    """
    def __init__(self, E_grid, E_margin_keV: float = 1.0):
        super().__init__()
        E_grid = E_grid.detach().float()
        self.register_buffer('E_grid',   E_grid)
        self.register_buffer('E_min',    torch.tensor(float(E_grid.min())))
        self.register_buffer('E_max',    torch.tensor(float(E_grid.max())))
        self.register_buffer('E_span',   self.E_max - self.E_min)
        self.register_buffer('E_margin', torch.tensor(E_margin_keV))

        # unconstrained params -> constrained inside forward
        self.logit_EK  = torch.nn.Parameter(torch.zeros(()))           # EK in [min+margin, max−margin]
        self.log_alpha = torch.nn.Parameter(torch.log(torch.tensor(0.1)))  # α > 0
        self.log_gamma = torch.nn.Parameter(torch.log(torch.tensor(0.5)))  # γ > 0 (1/keV)

    def EK(self):
        s = torch.sigmoid(self.logit_EK)
        return self.E_min + self.E_margin + s * (self.E_span - 2.0 * self.E_margin)

    def forward(self):
        E     = self.E_grid
        EK    = self.EK()
        alpha = self.log_alpha.exp()
        gamma = self.log_gamma.exp()
        step  = torch.sigmoid(gamma * (E - EK))
        spec  = alpha * bfunc_PE(E) * step   # >= 0, step-up across EK
        return spec

    @torch.no_grad()
    def set_gamma(self, g: float):
        self.log_gamma.copy_(torch.log(torch.tensor(float(g), device=self.log_gamma.device)))

    @torch.no_grad()
    def set_EK_init(self, ek_keV: float):
        ek = torch.tensor(float(ek_keV), device=self.logit_EK.device)
        ek = ek.clamp(self.E_min + self.E_margin, self.E_max - self.E_margin)
        s  = (ek - (self.E_min + self.E_margin)) / (self.E_span - 2.0 * self.E_margin)
        s  = s.clamp(1e-6, 1 - 1e-6)
        self.logit_EK.data = torch.log(s) - torch.log(1 - s)

# def renormalize_col_at_pivot(E_grid, B, j_col, E_ref=70.0, eps=1e-12):
#     """Divide column j_col so that its value at E_ref equals 1."""
#     val = interp1d(E_grid, B[:, j_col], torch.as_tensor(E_ref, device=B.device, dtype=B.dtype))
#     val = val.clamp_min(eps)
#     B[:, j_col] = B[:, j_col] / val
#     return val

