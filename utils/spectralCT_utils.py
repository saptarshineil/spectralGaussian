import torch

C = 510.999


def bfunc_PE(E_keV):
    return E_keV.pow(-3.0)

def bfunc_Compton(E_keV):
    E_keV = torch.as_tensor(E_keV, dtype=torch.float32, device=E_keV.device)

    a = E_keV / C

    # TODO: add epsilon for stability
    term1 = (1.0 + a) / (a ** 2) 
    term2 = 2.0 * (1.0 + a) / (1.0 + 2.0 * a) - (1.0 / a) * torch.log1p(2.0 * a)
    term3 = (1.0 / (2.0 * a)) * torch.log1p(2.0 * a)  
    term4 = (1.0 + 3.0 * a) / (1.0 + 2.0 * a) ** 2

    return term1 * term2 + term3 - term4

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
