import numpy as np
import torch
import pytest

# Import pure helpers from our module
from render_materials import (
    extract_material_features,
    _scale_features_minmax,
    cluster_materials_mean_shift,
    label_clusters_bone_vs_tissue,
    default_cluster_colors,
    compose_blend,
    compose_hard,
    run_material_clustering,  # smoke with dummy gaussians (no rendering)
)

class DummyGaussians:
    """Minimal stub to test feature extraction on CPU."""
    def __init__(self, rho, fracs):  # rho: (N,1), fracs: (N,2)
        self._rho = torch.as_tensor(rho, dtype=torch.float32)
        self._fr = torch.as_tensor(fracs, dtype=torch.float32)
    @property
    def get_rhos(self):
        return self._rho
    @property
    def get_fracs(self):
        # already softmax-like; mimic actual code path
        return self._fr

def test_extract_features_basic():
    # rho: 2 points; fracs: PE then Comp
    rho = [[2.0],[0.5]]
    fr  = [[0.8,0.2],[0.1,0.9]]
    g = DummyGaussians(rho, fr)
    feats = extract_material_features(g, pe_is_channel0=True)
    # photo & comp coefficients
    assert np.allclose(feats["pe_coeff"],   np.array([1.6, 0.05]), atol=1e-6)
    assert np.allclose(feats["comp_coeff"], np.array([0.4, 0.45]), atol=1e-6)
    # ratios
    r = feats["ratio"]
    assert r[0] > 1.0 and r[1] < 1.0
    # 2D features
    X = feats["feats2d"]
    assert X.shape == (2,2)
    assert np.allclose(X[:,0], np.array([2.0, 0.5]))

def test_minmax_scaling_is_robust():
    X = np.array([[0.0, 0.1],
                  [1.0, 10.0],
                  [5.0, 2.0],
                  [1000.0, 999.0]], dtype=np.float32)
    Xs, stats = _scale_features_minmax(X)
    assert Xs.min() >= 0.0 - 1e-6 and Xs.max() <= 1.0 + 1e-6
    assert "col0_lo" in stats and "col1_hi" in stats

def test_clustering_and_labels_kmeans_fallback():
    # clearly separable in ratio dimension
    rng = np.random.RandomState(0)
    rho_a = rng.uniform(0.5, 1.0, size=(50,1))
    rho_b = rng.uniform(1.0, 2.0, size=(50,1))
    ratio_a = rng.normal(0.2, 0.02, size=(50,1))
    ratio_b = rng.normal(1.5, 0.05, size=(50,1))
    X = np.vstack([np.hstack([rho_a, ratio_a]), np.hstack([rho_b, ratio_b])])
    labels, centers = cluster_materials_mean_shift(X, bandwidth=0.1)  # if sklearn present, still fine
    mapping = label_clusters_bone_vs_tissue(X, labels)
    # ensure high-ratio cluster is mapped to Bone
    bone_ids = [k for k,v in mapping.items() if "Bone" in v]
    assert len(bone_ids) == 1
    assert centers.shape[1] == 2

def test_default_colors_and_composite():
    bone_rgb = (0, 110, 255)
    tiss_rgb = (230, 60, 60)
    # simple images
    H, W = 8, 8
    bone = torch.ones(1, H, W)
    tiss = torch.zeros(1, H, W)
    blend = compose_blend(bone, tiss, bone_rgb, tiss_rgb)
    assert blend.shape == (3, H, W)
    # should be "near" bone color after normalization (within tolerance)
    mean_px = (blend*255).mean(dim=(1,2)).cpu().numpy()
    assert np.all(mean_px > np.array([0,80,180]))  # blue-ish

    # hard segmentation: left half bone, right half tissue
    bone2 = torch.zeros(1, H, W); bone2[:, :, :W//2] = 2.0
    tiss2 = torch.zeros(1, H, W); tiss2[:, :, W//2:] = 3.0
    hard = compose_hard(bone2, tiss2, bone_rgb, tiss_rgb)
    # left column == bone color, right == tissue color
    left = (hard[:, :, 0]*255).to(torch.int32).cpu().numpy()
    right= (hard[:, :, -1]*255).to(torch.int32).cpu().numpy()
    assert np.allclose(left.mean(axis=1), np.array(bone_rgb), atol=5)
    assert np.allclose(right.mean(axis=1), np.array(tiss_rgb), atol=5)
