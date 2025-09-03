import math
import numpy as np
import torch
import pytest
import importlib

import sys
sys.path.append('./')

import segmentation.render_materials_tri as rm
from utils.spectralCT_utils import BFuncKedge, energy_quadrature, get_B_E_at_energy

torch.backends.cudnn.benchmark = False
DEV = "cuda" if torch.cuda.is_available() else "cpu"

# ----------------- tiny helpers -----------------
class FakeGaussians:
    """
    Minimal stand-in for GaussianModel exposing just the tensors
    your render code touches: _rhos and _fracs (+ the properties).
    We'll use 3 "global" gaussians (metal, bone, tissue).
    """
    def __init__(self, rhos, fracs):
        self._rhos  = torch.as_tensor(rhos, device=DEV, dtype=torch.float32).view(-1, 1)
        self._fracs = torch.as_tensor(fracs, device=DEV, dtype=torch.float32)

    @property
    def get_rhos(self):
        # mimic positivity; your code often softplus() internally
        return torch.nn.functional.softplus(self._rhos)

    @property
    def get_fracs(self):
        return self._fracs


class DummyView: pass
class DummyPipe: pass


def make_spatial_templates(H=64, W=64, sigma=10.0):
    """
    Three spatial templates W_metal, W_bone, W_tiss over the image,
    smoothly overlapping and summing to 1 per pixel.
    """
    yy, xx = torch.meshgrid(torch.arange(H, device=DEV), torch.arange(W, device=DEV), indexing="ij")
    def g(cx, cy, s):
        return torch.exp(-((xx-cx)**2 + (yy-cy)**2) / (2*s*s))

    W_m = g(16, 16, sigma)
    W_b = g(48, 16, sigma)
    W_t = g(32, 48, sigma)

    S = W_m + W_b + W_t + 1e-8
    return (W_m/S, W_b/S, W_t/S)  # each (H,W)


def fake_render_mu_factory(Ws, coeffs_per_gaussian):
    """
    Build a fake render_mu that returns "bcoeffs" by mixing
    N gaussians over space using templates W_n(y,x):

        bcoeffs(c,y,x) = sum_n  (rho_n * frac_n[c]) * W_n(y,x)

    coeffs_per_gaussian: (N,C) tensor with rho*frac already multiplied.
    """
    W = torch.stack(Ws, dim=0)  # (N,H,W)
    C = coeffs_per_gaussian.shape[1]
    mix = (coeffs_per_gaussian[:, :, None, None] * W[:, None, :, :]).sum(dim=0)  # (C,H,W)

    def fake_render_mu(view, gaussians, pipe):
        # respect _render_cluster_mu's masking of gaussians via _rhos:
        # recompute with current _rhos (after masking) each call
        r = torch.nn.functional.softplus(gaussians._rhos).view(-1)  # (N,)
        # original fractions per gaussian:
        fr = gaussians._fracs  # (N,C)
        coeffs = (r[:, None] * fr)                                  # (N,C)
        bcoeffs = (coeffs[:, :, None, None] * W[:, None]).sum(0)    # (C,H,W)
        return {"bcoeffs": bcoeffs}

    return fake_render_mu, mix  # mix is only for debug


def make_kedge(E_min=20., E_max=140., N=128, ek=50., alpha=0.5, gamma=6.0):
    E_grid, _ = energy_quadrature(E_min, E_max, N, device=DEV)
    b = BFuncKedge(E_grid).to(DEV)
    with torch.no_grad():
        b.set_EK_init(ek)
        # α can be 0.0 in parametrized test -> avoid log(0)
        a = max(float(alpha), 1e-12)
        g = max(float(gamma), 1e-12)
        b.log_alpha.copy_(torch.log(torch.tensor(a, device=b.log_alpha.device)))
        b.log_gamma.copy_(torch.log(torch.tensor(g, device=b.log_gamma.device)))
    return b, E_grid


# ----------------- 1) Metal μ elevated above edge -----------------
@pytest.mark.parametrize("alpha_edge", [0.0, 0.5])
def test_metal_mu_elevated_above_edge(alpha_edge, monkeypatch):
    H, W = 64, 64
    W_m, W_b, W_t = make_spatial_templates(H, W)
    N = 3  # gaussians: metal, bone, tissue

    # per-class rho and fractions [PE, Comp, K]
    rho_m, rho_b, rho_t = 3.0, 1.6, 1.0
    f_m = torch.tensor([0.45, 0.35, 0.20], device=DEV)
    f_b = torch.tensor([0.80, 0.20, 0.00], device=DEV)
    f_t = torch.tensor([0.20, 0.80, 0.00], device=DEV)
    fracs = torch.stack([f_m, f_b, f_t], dim=0)                 # (N,3)
    rhos  = torch.tensor([rho_m, rho_b, rho_t], device=DEV)     # (N,)

    g = FakeGaussians(rhos, fracs)

    fake_render_mu, _ = fake_render_mu_factory((W_m, W_b, W_t), (rhos[:, None]*fracs))
    monkeypatch.setattr(rm, "render_mu", fake_render_mu)

    bfunc_K, E_grid = make_kedge(alpha=alpha_edge)
    energy_keV = 70.0  # > EK

    # build masks per class (metal/bone/tiss)
    metal_mask = torch.tensor([True, False, False], device=DEV)
    bone_mask  = torch.tensor([False, True, False], device=DEV)
    tiss_mask  = torch.tensor([False, False, True], device=DEV)

    mu_metal = rm._render_cluster_mu(DummyView(), g, DummyPipe(), metal_mask, energy_keV, bfunc_K, E_grid)  # (1,H,W)
    mu_bone  = rm._render_cluster_mu(DummyView(), g, DummyPipe(), bone_mask,  energy_keV, bfunc_K, E_grid)

    # Compare means over "mostly metal" vs "mostly bone" regions:
    gt_is_metal = (W_m > (W_b).maximum(W_t))
    gt_is_bone  = (W_b > (W_m).maximum(W_t))

    m_metal = float(mu_metal[0, gt_is_metal].mean().cpu())
    m_bone  = float(mu_bone[0,  gt_is_bone].mean().cpu())
    # compute separation metric
    sep = (m_metal - m_bone) / (m_bone + 1e-6)

    if alpha_edge == 0.0:
        # Without edge you should NOT get strong extra separation
        assert sep < 0.35, f"Unexpectedly large metal/bone separation without K-edge: {sep:.3f}"
    else:
        # With edge you SHOULD get clear separation
        assert sep > 0.35, f"Metal/bone separation too small with K-edge: {sep:.3f}"


# ----------------- 2) Hard segmentation accuracy improves with edge -----------------
def _hard_labels_from_three(mu_metal, mu_bone, mu_tiss):
    S = torch.stack([mu_metal, mu_bone, mu_tiss], dim=0)  # (3,1,H,W)
    return torch.argmax(S, dim=0).squeeze(0)              # (H,W) in {0,1,2}

def segmentation_accuracy(alpha_edge, monkeypatch):
    H, W = 64, 64
    W_m, W_b, W_t = make_spatial_templates(H, W)
    N = 3

    rho_m, rho_b, rho_t = 3.0, 1.6, 1.0
    f_m = torch.tensor([0.45, 0.35, 0.20], device=DEV)    # metal has some K-edge fraction
    f_b = torch.tensor([0.80, 0.20, 0.00], device=DEV)
    f_t = torch.tensor([0.20, 0.80, 0.00], device=DEV)
    fracs = torch.stack([f_m, f_b, f_t], dim=0)
    rhos  = torch.tensor([rho_m, rho_b, rho_t], device=DEV)

    g = FakeGaussians(rhos, fracs)
    fake_render_mu, _ = fake_render_mu_factory((W_m, W_b, W_t), (rhos[:, None]*fracs))
    monkeypatch.setattr(rm, "render_mu", fake_render_mu)

    bfunc_K, E_grid = make_kedge(alpha=alpha_edge)
    energy_keV = 70.0

    metal_mask = torch.tensor([True, False, False], device=DEV)
    bone_mask  = torch.tensor([False, True, False], device=DEV)
    tiss_mask  = torch.tensor([False, False, True], device=DEV)

    mu_metal = rm._render_cluster_mu(DummyView(), g, DummyPipe(), metal_mask, energy_keV, bfunc_K, E_grid)
    mu_bone  = rm._render_cluster_mu(DummyView(), g, DummyPipe(), bone_mask,  energy_keV, bfunc_K, E_grid)
    mu_tiss  = rm._render_cluster_mu(DummyView(), g, DummyPipe(), tiss_mask,  energy_keV, bfunc_K, E_grid)

    pred = _hard_labels_from_three(mu_metal, mu_bone, mu_tiss)  # 0=metal,1=bone,2=tissue

    gt = torch.where((W_m > (W_b).maximum(W_t)), 0,
         torch.where((W_b > (W_m).maximum(W_t)), 1, 2)).to(pred.device)

    acc = float((pred == gt).float().mean().cpu())
    return acc

def test_segmentation_benefits_from_kedge(monkeypatch):
    acc_noedge = segmentation_accuracy(alpha_edge=0.0, monkeypatch=monkeypatch)
    acc_edge   = segmentation_accuracy(alpha_edge=0.5, monkeypatch=monkeypatch)

    # Expect a noticeable improvement when the K-edge contributes.
    assert acc_edge > acc_noedge + 0.05, f"Acc did not improve with K-edge: {acc_noedge:.3f} -> {acc_edge:.3f}"
    assert acc_edge > 0.90, f"Segmentation accuracy with K-edge too low: {acc_edge:.3f}"
