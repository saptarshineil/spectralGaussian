import torch
import numpy as np
import pytest
import os, sys

sys.path.append('./')

from utils.spectralCT_utils import energy_quadrature, get_B_E
from train import orthogonalize_k_to_pe_compton
from utils.spectralCT_utils import BFuncKedge as BFuncKedgeImpl

DEV = "cuda" if torch.cuda.is_available() else "cpu"

def make_kedge(E_min=20., E_max=140., N=128, ek=50., alpha=0.4, gamma=2.0):
    E_grid, w = energy_quadrature(E_min, E_max, N, device=DEV)
    b = BFuncKedgeImpl(E_grid).to(DEV)
    with torch.no_grad():
        b.set_EK_init(ek)
        # DO NOT use [:] on scalars; copy_ into the 0-dim parameters.
        b.log_alpha.copy_(torch.log(torch.tensor(alpha, device=b.log_alpha.device)))
        b.log_gamma.copy_(torch.log(torch.tensor(gamma, device=b.log_gamma.device)))
    return b, E_grid, w

# ---------- 1) EK bounds ----------
def test_ek_within_bounds():
    b, E, _ = make_kedge()
    ek = b.EK()
    assert float(ek) >= float(b.E_min + b.E_margin) - 1e-6
    assert float(ek) <= float(b.E_max - b.E_margin) + 1e-6

# ---------- 2) monotonic away from edge ----------
@torch.no_grad()
def test_monotonic_away_from_edge():
    b, E, _ = make_kedge(ek=50., alpha=0.5, gamma=3.0)
    spec = b().detach().cpu().numpy()  # α * E^{-3} * step
    E_np = E.cpu().numpy()

    mask_low  = E_np < 40.0
    mask_high = E_np > 60.0

    # finite differences
    def is_strictly_decreasing(x, e):
        dif = np.diff(x[e])
        return np.all(dif < 1e-8)  # allow tiny numerical noise

    assert is_strictly_decreasing(spec, mask_high), "spec should decrease with E well above EK"
    # below EK spec ~= 0 → no trend to test, but ensure small:
    assert (spec[mask_low] < 1e-6).all(), "spec should be near zero well below EK"

# ---------- 3) step behavior ----------
@torch.no_grad()
def test_step_ratio_increases_with_alpha():
    b1, E, _ = make_kedge(ek=50., alpha=0.2, gamma=5.0)
    b2, _, _ = make_kedge(ek=50., alpha=0.6, gamma=5.0)
    s1 = b1().detach()
    s2 = b2().detach()
    # pick points just below/above edge
    e_lo = torch.argmin((E-48).abs())
    e_hi = torch.argmin((E-52).abs())
    r1 = (s1[e_hi] + 1e-12) / (s1[e_lo] + 1e-12)
    r2 = (s2[e_hi] + 1e-12) / (s2[e_lo] + 1e-12)
    assert r2 > r1, "bigger alpha should produce a larger jump across EK"

# ---------- 4) pivot normalization correctness ----------
@torch.no_grad()
def test_pivot_normalization():
    b, E, _ = make_kedge(ek=50.)
    B = get_B_E(E, b, normalize=True, E_ref=70.0)  # (K,3) if PE+Comp+Kedge
    # columns should equal 1 at E_ref (within interpolation error)
    from utils.general_utils import interp1d
    Eref = torch.tensor(70.0, device="cuda")
    vals_at_ref = interp1d(E, B, Eref)  # (3,)
    assert torch.allclose(vals_at_ref, torch.ones_like(vals_at_ref), atol=5e-3)

# ---------- 5) photon-weighted orthogonality ----------
@torch.no_grad()
def test_weighted_orthogonality():
    b, E, w = make_kedge(ek=50.)
    B_raw = get_B_E(E, b, normalize=True, E_ref=70.0)  # (K,3)
    # photon-weighted inner product with average spectrum
    # simple flat spectrum surrogate (ok for unit test)
    S = torch.ones_like(E)
    W = (w * S).clamp_min(1e-12)
    B = orthogonalize_k_to_pe_compton(B_raw, W)
    G = (B * W[:, None]).T @ B
    # off-diagonals in first two rows vs third should be ~0
    assert abs(G[0,2].item()) < 1e-4 and abs(G[1,2].item()) < 1e-4

# ---------- 6) gradient flows & learns EK on tiny synthetic ----------
def test_learns_edge_on_synthetic():
    torch.manual_seed(0)
    # synthetic target basis (ground-truth)
    ek_gt, alpha_gt, gamma_gt = 55.0, 0.4, 4.0
    b_gt, E, w = make_kedge(ek=ek_gt, alpha=alpha_gt, gamma=gamma_gt)
    y = b_gt().detach()  # pretend we "measured" this spectral shape (one pixel)

    # learn EK, alpha, gamma from scratch
    b, _, _ = make_kedge(ek=60.0, alpha=0.2, gamma=1.0)  # poor init
    opt = torch.optim.Adam(b.parameters(), lr=5e-2)
    for _ in range(200):
        opt.zero_grad()
        loss = torch.nn.functional.mse_loss(b(), y)
        loss.backward(); opt.step()
    ek_est = float(b.EK().detach().cpu())
    assert abs(ek_est - ek_gt) < 2.0, f"EK should be learned (~{ek_gt} keV); got {ek_est:.2f}"
