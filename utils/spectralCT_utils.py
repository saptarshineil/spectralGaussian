import numpy as np
import torch

C = 510.999               # mec²  [keV]
EPSILON = 1e-8            # smaller eps gives better accuracy but still guards denom ≈ 0

def bfunc_PE(E_keV: torch.Tensor) -> torch.Tensor:
    """
    Photo‑electric basis  ~  E^{-3}.
    """
    E_keV = torch.as_tensor(E_keV, dtype=torch.float32, device=E_keV.device)
    # return 511999.9688 * torch.pow(E_keV, -3.0)          # (+eps) protects E≈0
    return torch.pow(E_keV, -3.0)          # (+eps) protects E≈0

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

    # return (term1 * term2 + term3 - term4) * 0.9645
    return term1 * term2 + term3 - term4

# C = 510.999
# EPSILON = 1e-5

# def bfunc_PE(E_keV):
#     return E_keV.pow(-3.0)

# def bfunc_Compton(E_keV):
#     E_keV = torch.as_tensor(E_keV, dtype=torch.float32, device=E_keV.device)

#     a = E_keV / C

#     # TODO: add epsilon for stability
#     term1 = (1.0 + a) / ((a ** 2) + EPSILON) # low freq
#     term1 = term1

#     # high freq:
#     # prioritize high freq
#     term2 = 2.0 * (1.0 + a) / ((1.0 + 2.0 * a) - (1.0 / a) * torch.log1p(2.0 * a) + EPSILON)
#     term3 = (1.0 / ((2.0 * a) + EPSILON)) * torch.log1p(2.0 * a)  
#     term4 = (1.0 + 3.0 * a) / ((1.0 + 2.0 * a) ** 2 + EPSILON)

#     return term1 * term2 + term3 - term4

# def bfunc_Compton_per_electron(E_keV):
#     a = E_keV / C
#     t1 = (1+a)/a**2
#     t2 = 2*(1+a)/(1+2*a) - (1/a)*torch.log1p(2*a)
#     t3 = 0.5/a * torch.log1p(2*a)
#     t4 = (1+3*a)/(1+2*a)**2
#     return t1*t2 + t3 - t4      

# rho, Z_over_A = 1.0, 0.555                   # water
# rho_e = rho * Z_over_A * 6.022e23 * 1e-24    # → cm² g⁻¹ units
# def bfunc_Compton(E): return rho_e * bfunc_Compton_per_electron(E)

def analytic_fraction(rho, rho0=1.55, alpha=3.1, eps=1e-3):
    f_pe = 1.0 / (1.0 + (rho0 / rho)**alpha)
    f_pe = np.clip(f_pe, eps, 1.0-eps)
    return np.stack([f_pe, 1.0 - f_pe], axis=-1).astype(np.float32)


# MATERIAL_DB = {
#     "water":     dict(rho=1.00, mu_pe=0.0403, mu_co=0.1347),
#     "bone":      dict(rho=1.85, mu_pe=0.1650, mu_co=0.1720),
#     "adipose":   dict(rho=0.95, mu_pe=0.0340, mu_co=0.1190),
#     "iodine15":  dict(rho=1.03, mu_pe=1.1600, mu_co=0.3500),
# }

# def closest_material(pe_to_c_ratio: torch.Tensor):
#     """
#     Map an observed PE/C ratio (scalar tensor) to the material
#     whose μ_PE / μ_C is closest.  Returns (material_name, ratio).
#     """
#     names, ratios = zip(*[(k, v["mu_pe"] / v["mu_co"]) for k, v in MATERIAL_DB.items()])
#     ratios = torch.tensor(ratios, device=pe_to_c_ratio.device)
#     idx = torch.argmin(torch.abs(ratios - pe_to_c_ratio))
#     return names[idx], ratios[idx]

