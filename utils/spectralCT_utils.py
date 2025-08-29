import numpy as np
import torch
import spekpy
from utils.general_utils import interp1d


C = 510.999
EPSILON = 1e-12


# ------------------------ Fraction initialization ------------------------
def pe_compton_from_density(rho, q_lo=0.20, q_hi=0.80, eps=1e-3):
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


def analytic_fraction_2(rho, rho0=1.55, alpha=3.1, eps=1e-3):
    f_pe = 1.0 / (1.0 + (rho0 / rho)**alpha)
    f_pe = np.clip(f_pe, eps, 1.0-eps)
    return np.stack([f_pe, 1.0 - f_pe], axis=-1).astype(np.float32)


def analytic_fraction_3(
        rho,
        rho0_pe   = 1.55,     # centre of transition  [g cm⁻³]
        alpha_pe  = 3.1,      # slope
        rho0_edge = 2.4,      # edge weight
        alpha_edge= 10.0,     # slope
        k_frac_max= 0.70,     # max PE share
        eps       = 1e-3
    ):
    """
    Returns initial fractions [f_PE, f_Comp, f_K] for a 3-basis model.
    `rho` can be a scalar, a 1-D or an (N,1) array.
    """

    rho = np.asarray(rho, np.float32).squeeze()

    # ── 1.  classic PE vs Comp split  ───────────────────────────
    f_pe_tot  = 1. / (1. + (rho0_pe / rho)**alpha_pe)           # PE budget
    f_pe_tot  = np.clip(f_pe_tot, eps, 1 - eps)

    f_comp    = 1. - f_pe_tot                                   # rest → Compton

    # ── 2.  decide how much of that PE budget is actually an edge ─
    gate_edge = 1. / (1. + np.exp(-alpha_edge * (rho - rho0_edge)))
    f_k       = k_frac_max * gate_edge * f_pe_tot               # ≤ k_frac_max·f_pe_tot
    f_pe      = f_pe_tot - f_k

    # ── 3.  stack & renormalise (paranoia)  ─────────────────────
    frac = np.stack([f_pe, f_comp, f_k], axis=-1)
    frac = np.clip(frac, eps, None)
    frac /= frac.sum(axis=-1, keepdims=True)

    return frac.astype(np.float32)


# ------------------------ Photoelectric + Compton basis functions ------------------------
def bfunc_PE(E_keV: torch.Tensor) -> torch.Tensor:
    E_keV = torch.as_tensor(E_keV, dtype=torch.float32, device=E_keV.device)
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

def _col_scales_at_pivot(E_grid: torch.Tensor, bfunc_Kedge, E_ref: float):
    eref = torch.as_tensor(E_ref, dtype=E_grid.dtype, device=E_grid.device)
    s_pe = bfunc_PE(eref).clamp(min=1e-30)
    s_co = bfunc_Compton(eref).clamp(min=1e-30)
    scales = [s_pe, s_co]
    if bfunc_Kedge is not None:
        scales.append(torch.tensor(1.0, device=E_grid.device, dtype=E_grid.dtype))
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


# ------------------------ Discretized energy grid ------------------------
def energy_quadrature(E_min, E_max, N, *, device="cuda", dtype=torch.float32):
    e0 = torch.tensor(E_min, device=device, dtype=dtype)
    e1 = torch.tensor(E_max, device=device, dtype=dtype)
    edges = torch.logspace(torch.log10(e0), torch.log10(e1), steps=N+1, device=device, dtype=dtype)
    E_k = 0.5*(edges[:-1] + edges[1:])
    w_k = (edges[1:] - edges[:-1])
    return E_k, w_k

def get_scaled_spectrum(kvp, filt_mat, filt_thick, I0, E_grid_np):
    spek = spekpy.Spek(kvp=kvp, th=12)
    if filt_mat is not None:
        spek.filter(filt_mat, filt_thick)
    E_native, S_native = spek.get_spectrum()         # photons / keV / pixel
    S = np.interp(E_grid_np, E_native, S_native)     # resample
    S /= np.trapz(S, E_grid_np)                      # normalise
    return torch.tensor(S*I0, device='cuda', dtype=torch.float32)


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
    Two–parameter analytic K-edge basis.
    Bθ(E) = [(E0/E) + α·H(E-EK)]^{-3}  ,   rescaled to max = 1.
    """
    def __init__(self, E_grid):
        super().__init__()
        E_grid = E_grid.detach().float()
        self.register_buffer('E_grid', E_grid)

        # --- parameters -------------------------------
        self.log_EK = torch.nn.Parameter(torch.log(E_grid.mean()))   # ln E_K
        self.log_alpha = torch.nn.Parameter(torch.tensor(0.0))       # ln α
        self.E0 = E_grid.max()                                       # constant

    def forward(self):
        EK = self.log_EK.exp()
        alpha = self.log_alpha.exp()

        tail  = self.E0 / self.E_grid
        step  = alpha * (self.E_grid >= EK).float()
        spec  = (tail + step).pow(-3)

        return spec / spec.max()