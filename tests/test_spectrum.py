import torch, numpy as np
from tests.test_utils import build_grid_and_spectra, beer_lambert, save_npz, ensure_dir, DEFAULTS

def main():
    E, w, S, I0, B = build_grid_and_spectra()
    arts = ensure_dir("artifacts/spectrum")
    ok_all = True

    # ---------- B1: I0 check ----------
    I0_sum = torch.sum(S*w, dim=1)
    rel = torch.abs(I0_sum - I0) / (I0 + 1e-12)
    print(f"[B1] rel error I0 vs sum(S·w): {rel.tolist()}")
    if not torch.all(rel < 1e-6): ok_all=False

    # ---------- build a simple mu(E) for tests ----------
    coeff_true = torch.tensor([0.03, 0.05], device=E.device)[:B.shape[1]]
    mu_E = (B * coeff_true[None,:]).sum(dim=1)

    # ---------- B2: scale invariance ----------
    for c in (0,1):
        I_pred, y = beer_lambert(I0[c], S[c], w, mu_E)
        alpha = 3.7
        I_pred2, y2 = beer_lambert(alpha*I0[c], alpha*S[c], w, mu_E)
        rel = float((torch.abs(y2-y)/(torch.abs(y)+1e-12)).max().item())
        print(f"[B2] channel {c}: |y2-y|/|y| max = {rel:.2e}")
        if rel > 1e-6: ok_all=False

    # ---------- B3: quadrature convergence 64→128 ----------
    from utils.spectralCT_utils import energy_quadrature, get_scaled_spectrum, get_B_E
    E2, w2, S2, I0_2, B2 = build_grid_and_spectra(N=128)[0:5]
    mu2 = (B2 * coeff_true[None,:]).sum(dim=1)
    y64 = torch.stack([beer_lambert(I0[i],   S[i],   w,  mu_E)[1] for i in (0,1)])
    y128= torch.stack([beer_lambert(I0_2[i], S2[i], w2, mu2  )[1] for i in (0,1)])
    rel = float((torch.abs(y128 - y64)/(torch.abs(y128)+1e-12)).max().item())
    print(f"[B3] max rel change (128 vs 64 bins) = {rel:.3e}")
    if rel > 1e-3: ok_all=False

    # ---------- B4: spectra sanity ----------
    for i, kvp in enumerate([DEFAULTS['kvp_low'], DEFAULTS['kvp_high']]):
        S_i = S[i]; nonneg = (S_i>=0).all().item()
        p = (S_i * w) / (torch.sum(S_i*w)+1e-12)
        eff_bins = 1.0 / float((p**2).sum().item())
        e_peak = float(E[torch.argmax(S_i)])
        print(f"[B4] ch{i}: nonneg={nonneg}, eff_bins≈{eff_bins:.1f}, E_peak≈{e_peak:.1f} keV")
        if not nonneg or eff_bins < 10: ok_all=False

    save_npz(f"{arts}/spectra_and_mu.npz", E=E, w=w, S=S, I0=I0, B=B)
    print(f"\nB SUMMARY: {'PASS' if ok_all else 'FAIL'}")
    return 0 if ok_all else 4

if __name__ == "__main__":
    raise SystemExit(main())