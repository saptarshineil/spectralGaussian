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

def pe_comp_kedge_from_density(rho, q_lo=0.01, q_hi=0.99, rho0_edge_q=0.95, alpha_edge=20.0, k_frac_max=0.60, eps=1e-3):
    """
    Monotone logistic prior: high rho -> high PE fraction.
    q_lo,q_hi set where f_PE hits ~0.1 and ~0.9, respectively.
    rho0_edge_q: quantile of the rho distribution to use as K-edge gate threshold.
      Using a data-adaptive quantile instead of a hardcoded value ensures the gate
      correctly selects the top fraction of high-density Gaussians regardless of scale.
      e.g. rho0_edge_q=0.85 → top 15% of Gaussians by density get non-zero f_ke.
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
    f_pe_tot = 1.0 / (1.0 + np.exp(-t))
    f_pe_tot = np.clip(f_pe_tot, eps, 1.0 - eps)

    f_comp    = 1. - f_pe_tot                                   # rest → Compton

    # ── 2.  decide how much of that PE budget is actually an edge ─
    # Use data-adaptive threshold: top rho0_edge_q fraction of Gaussians get K budget.
    # Gate uses scale-invariant form: sigmoid(alpha * (rho/rho0 - 1)).
    # With rho in [0.001, 0.015], the OLD form sigmoid(10*(rho-rho0)) had transition width
    # 4/alpha=0.4 >> density range 0.014, so ALL Gaussians got gate≈0.5 → K mean≈0.267 everywhere.
    # The scale-invariant form has transition width ≈ 2*rho0_edge/alpha regardless of scale.
    rho0_edge = float(np.quantile(rho, rho0_edge_q))
    gate_edge = 1. / (1. + np.exp(-alpha_edge * (rho / (rho0_edge + 1e-12) - 1.0)))
    f_k       = k_frac_max * gate_edge * f_pe_tot               # ≤ k_frac_max·f_pe_tot
    f_pe      = f_pe_tot - f_k

    # ── 3.  stack & renormalise (paranoia)  ─────────────────────
    frac = np.stack([f_pe, f_comp, f_k], axis=-1)
    frac = np.clip(frac, eps, None)
    frac /= frac.sum(axis=-1, keepdims=True)

    rho0 = float(np.exp(log_rho0))
    return frac, rho0, float(alpha)


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
def _default_spectral_cfg():
    return {
        "n_channels": 3,
        "I0": [1136.8784583333334, 1594.836, 2026.8482152777776],
        "E": [50.0, 80.0, 120.0],
        "S": [
            {"material": None, "thickness": 0.0},
            {"material": "Al", "thickness": 1.0},
            {"material": "Cu", "thickness": 0.5},
        ],
    }


def _parse_filter_cfg(filter_cfg):
    if filter_cfg is None:
        return None, 0.0
    if isinstance(filter_cfg, dict):
        return filter_cfg.get("material"), float(filter_cfg.get("thickness", 0.0))
    if isinstance(filter_cfg, (list, tuple)) and len(filter_cfg) == 2:
        return filter_cfg[0], float(filter_cfg[1])
    raise ValueError(f"Unsupported spectral filter specification: {filter_cfg!r}")


def _channel_spectrum_type(channel_cfg):
    if not isinstance(channel_cfg, dict):
        return "tube"
    return str(
        channel_cfg.get("type", channel_cfg.get("spectrum", channel_cfg.get("kind", "tube")))
    ).lower()


def _discrete_mono_spectrum(energy_keV, I0, E_grid_np, w_k_np, *, device="cuda", dtype=torch.float32):
    idx = int(np.argmin(np.abs(E_grid_np - float(energy_keV))))
    S = np.zeros_like(E_grid_np, dtype=np.float32)
    S[idx] = float(I0) / (float(w_k_np[idx]) + 1e-12)
    return torch.tensor(S, device=device, dtype=dtype)


def _discrete_energy_bin_spectrum(channel_cfg, energy_keV, I0, E_grid_np, w_k_np, *, device="cuda", dtype=torch.float32):
    e_min = channel_cfg.get("energy_min", channel_cfg.get("E_min", channel_cfg.get("lower", None)))
    e_max = channel_cfg.get("energy_max", channel_cfg.get("E_max", channel_cfg.get("upper", None)))
    if e_min is None or e_max is None:
        width = float(channel_cfg.get("width", channel_cfg.get("width_keV", 0.0)))
        if width <= 0.0:
            return _discrete_mono_spectrum(energy_keV, I0, E_grid_np, w_k_np, device=device, dtype=dtype)
        e_min = float(energy_keV) - 0.5 * width
        e_max = float(energy_keV) + 0.5 * width
    e_min = float(e_min)
    e_max = float(e_max)
    if e_max <= e_min:
        raise ValueError(f"Invalid energy-bin bounds: energy_min={e_min}, energy_max={e_max}")

    S = np.zeros_like(E_grid_np, dtype=np.float32)
    mask = (E_grid_np >= e_min) & (E_grid_np < e_max)
    if not np.any(mask):
        center = 0.5 * (e_min + e_max)
        mask[int(np.argmin(np.abs(E_grid_np - center)))] = True

    S[mask] = 1.0
    S /= np.sum(S * w_k_np) + 1e-12
    return torch.tensor(S * float(I0), device=device, dtype=dtype)


def get_channel_spectrum(energy_keV, channel_cfg, I0, E_grid_np, w_k_np=None, *, device="cuda", dtype=torch.float32):
    """
    Build a discrete source/detector spectrum on ``E_grid_np`` for one channel.

    Backward-compatible default: an entry without an explicit ``type`` is treated as
    a SpekPy tube spectrum with optional material filtration.  Hyperspectral
    detector data should instead use ``{"type": "mono"}`` or
    ``{"type": "energy_bin", "energy_min": ..., "energy_max": ...}`` so the
    channel energy in metadata is not misinterpreted as tube kVp.
    """
    if w_k_np is None:
        dE = float(np.median(np.diff(E_grid_np))) if len(E_grid_np) > 1 else 1.0
        w_k_np = np.full_like(E_grid_np, dE, dtype=np.float32)

    spectrum_type = _channel_spectrum_type(channel_cfg)
    if spectrum_type in ("mono", "monochromatic", "delta", "energy"):
        return _discrete_mono_spectrum(energy_keV, I0, E_grid_np, w_k_np, device=device, dtype=dtype)
    if spectrum_type in ("bin", "energy_bin", "detector_bin", "window"):
        return _discrete_energy_bin_spectrum(channel_cfg, energy_keV, I0, E_grid_np, w_k_np, device=device, dtype=dtype)
    if spectrum_type not in ("tube", "spekpy", "poly", "polychromatic"):
        raise ValueError(f"Unsupported spectral channel type: {spectrum_type!r}")

    filt_mat, filt_thick = _parse_filter_cfg(channel_cfg)
    return get_scaled_spectrum(energy_keV, filt_mat, filt_thick, I0, E_grid_np, w_k_np, device=device, dtype=dtype)


def _normalize_spectral_cfg(spectral_cfg):
    spectral_cfg = _default_spectral_cfg() if spectral_cfg is None else spectral_cfg

    energies = [float(e) for e in spectral_cfg.get("E", [])]
    i0_values = [float(i0) for i0 in spectral_cfg.get("I0", [])]
    filters = spectral_cfg.get("S", [])

    if not energies:
        raise ValueError("Spectral metadata must contain non-empty 'E' energy values.")
    if len(i0_values) != len(energies):
        raise ValueError(
            f"Spectral metadata length mismatch: got {len(i0_values)} I0 values for {len(energies)} energies."
        )
    if len(filters) != len(energies):
        raise ValueError(
            f"Spectral metadata length mismatch: got {len(filters)} filter entries for {len(energies)} energies."
        )

    n_channels = int(spectral_cfg.get("n_channels", len(energies)))
    if n_channels != len(energies):
        raise ValueError(
            f"Spectral metadata length mismatch: n_channels={n_channels}, but {len(energies)} energies were provided."
        )

    return energies, i0_values, filters


def get_spectral_constants(E_grid=None, w_k=None, spectral_cfg=None, *, device="cuda", dtype=torch.float32):
    energies, i0_values, filters = _normalize_spectral_cfg(spectral_cfg)
    E = torch.tensor(energies, device=device, dtype=dtype)

    if E_grid is None and w_k is None:
        N_samples = 64
        # If dataset is EID (no bins) --> sample full spectrum
        E_grid, w_k = energy_quadrature(20.0, 140.0, N_samples, device=device, dtype=dtype)
    elif E_grid is None or w_k is None:
        raise ValueError("E_grid and w_k must either both be provided or both be omitted.")
    else:
        E_grid = E_grid.to(device=device, dtype=dtype)
        w_k = w_k.to(device=device, dtype=dtype)

    E_grid_np = E_grid.cpu().numpy()
    w_k_np = w_k.cpu().numpy()

    spectra = []
    for energy_keV, i0, channel_cfg in zip(energies, i0_values, filters):
        spectra.append(get_channel_spectrum(energy_keV, channel_cfg, i0, E_grid_np, w_k_np, device=device, dtype=dtype))

    S = torch.stack(spectra)
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

def get_scaled_spectrum(kvp, filt_mat, filt_thick, I0, E_grid_np, w_k_np=None, *, device="cuda", dtype=torch.float32):
    spek = spekpy.Spek(kvp=kvp, th=12)
    if filt_mat is not None:
        spek.filter(filt_mat, filt_thick)
    E_native, S_native = spek.get_spectrum()
    S = np.interp(E_grid_np, E_native, S_native)
    if w_k_np is None:
        dE = float(np.median(np.diff(E_grid_np))) if len(E_grid_np) > 1 else 1.0
        w_k_np = np.full_like(E_grid_np, dE, dtype=np.float32)
    S /= np.sum(S * w_k_np) + 1e-12
    return torch.tensor(S*I0, device=device, dtype=dtype)


# def render_image(bcoeff_image, bfunc_Kedge, E, E_grid, w_k, S, W, I0_scalar, n_bcoeffs, spectral_id, log_tilt):
# def render_image(bcoeff_image, bfunc_Kedge, E, E_grid, w_k, S, W, I0_scalar, n_bcoeffs, spectral_id, scatter_frac):
def render_image(bcoeff_image, bfunc_Kedge, E, E_grid, w_k, S, W, I0_scalar, n_bcoeffs, spectral_id, B_E=None):
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

    # B_E can be pre-computed once per iteration and passed in;
    # if absent, compute it here (backward-compatible).
    if B_E is None:
        if n_bcoeffs == 3:
            B_raw = get_B_E(E_grid, bfunc_Kedge, normalize=False)
            B_raw = orthogonalize_k_to_pe_compton(B_raw, W)
            B_E, _ = renormalize_col_at_pivot(E_grid, B_raw, E_ref=70.0)
        else:
            B_E = get_B_E(E_grid, None, normalize=True)

    if n_bcoeffs == 3:
        bfuncs = torch.stack([interp1d(E_grid, B_E[:, j], E[spectral_id]) for j in range(B_E.shape[1])], dim=0)
    else:
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
        Bθ(E) = α · σ(γ·(E − EK))
    where σ is the logistic sigmoid.  The basis is flat (~α) above EK and
    ~0 below — a step function in the mass-attenuation coefficient.

    NOTE: the previous model α·E⁻³·σ(·) was nearly proportional to bfunc_PE
    for EK well below the pivot (70 keV), causing the orthogonalized column
    b3o to vanish at the pivot and renormalize_col_at_pivot to blow up (NaN).
    Using a plain step (no E⁻³ factor) makes the K-edge column genuinely
    different in shape from PE (decreasing E⁻³) and Compton, yielding a
    well-conditioned orthogonalized basis.

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
        # spec  = alpha * bfunc_PE(E) * step   # was: alpha*E^-3*step ≈ PE → collinear after ortho → tiny K grad
        spec  = alpha * step                    # pure step: genuinely different from PE/Compton → large K grad
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

