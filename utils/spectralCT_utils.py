import numpy as np
import torch
import spekpy
from utils.general_utils import interp1d


C = 510.999
EPSILON = 1e-8


# ------------------------ Fraction initialization ------------------------ 
def analytic_fraction(
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
    """
    Photo‑electric basis  ~  E^{-3}.
    """
    E_keV = torch.as_tensor(E_keV, dtype=torch.float32, device=E_keV.device)
    return torch.pow(E_keV, -3.0)         


def bfunc_Compton(E_keV: torch.Tensor,
                  eps: float = EPSILON) -> torch.Tensor:
    """
    Klein‑Nishina–based Compton basis with ε‑stabilised denominators.
    """
    E_keV = torch.as_tensor(E_keV, dtype=torch.float32, device=E_keV.device)
    
    # Dimensionless energy ratio  a = E / mec²
    a = torch.clamp(E_keV / C, min=eps)              # avoids a→0 divergences

    # Pre‑compute safe denominators
    denom_a      = a + eps
    denom_a2     = a.pow(2) + eps
    denom_2a     = 2.0 * a + eps
    denom_1_2a   = 1.0 + denom_2a                    # already contains +eps
    denom_1_2a2  = denom_1_2a.pow(2) + eps

    # Low‑frequency component
    term1 = (1.0 + a) / denom_a2

    # High‑frequency corrections (original Klein–Nishina algebra, eps‑safe)
    term2 = 2.0 * (1.0 + a) / denom_1_2a \
            - torch.log1p(denom_2a - eps) / denom_a   # subtract eps to keep log argument exact

    term3 = torch.log1p(denom_2a - eps) / denom_2a
    term4 = (1.0 + 3.0 * a) / denom_1_2a2

    return term1 * term2 + term3 - term4


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


# ------------------------ Discretized energy grid ------------------------
def energy_quadrature(E_min, E_max, N, *, device="cuda", dtype=torch.float32):
    dE   = (E_max - E_min) / N                       # scalar
    E_k  = torch.linspace(E_min + 0.5*dE,            # mid-points
                        E_max - 0.5*dE,
                        steps=N,
                        device=device,
                        dtype=dtype)               # (K,)
    w_k  = torch.full((N,), dE, device=device, dtype=dtype)  # (K,)
    return E_k, w_k

def get_scaled_spectrum(kvp, filt_mat, filt_thick, I0, E_grid_np):
    spek = spekpy.Spek(kvp=kvp, th=12)
    if filt_mat is not None:
        spek.filter(filt_mat, filt_thick)
    E_native, S_native = spek.get_spectrum()         # photons / keV / pixel
    S = np.interp(E_grid_np, E_native, S_native)     # resample
    S /= np.trapz(S, E_grid_np)                      # normalise
    return torch.tensor(S*I0, device='cuda', dtype=torch.float32)

def get_B_E(E_grid, bfunc_Kedge):
        return torch.stack([bfunc_PE(E_grid), bfunc_Compton(E_grid), bfunc_Kedge()], dim=1)
        # return torch.stack([bfunc_PE(E_grid), bfunc_Compton(E_grid)], dim=1)

def get_B_E_at_energy(E_grid, bfunc_Kedge, E):
    Kedge = interp1d(E_grid, bfunc_Kedge(), E)
    return torch.stack([bfunc_PE(E), bfunc_Compton(E), Kedge], dim=-1)
