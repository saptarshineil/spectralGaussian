import torch
import torchvision

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

# Your palette
TISSUE_RGB = (230, 60, 60)   # red
BONE_RGB   = (0, 110, 255)   # blue

def make_halfhalf_phantom(H=256, W=256,
                          comp_soft=1.0,  pe_over_comp_soft=0.15,
                          comp_bone=1.0,  pe_over_comp_bone=0.55):
    """
    Returns bcoeffs of shape (2, H, W): [Compton, Photoelectric].
    """
    b = torch.zeros(2, H, W, dtype=torch.float32)
    mid = W // 2
    b[0, :, :mid] = comp_soft
    b[1, :, :mid] = comp_soft * pe_over_comp_soft
    b[0, :, mid:] = comp_bone
    b[1, :, mid:] = comp_bone * pe_over_comp_bone
    b += 0.02 * torch.randn_like(b)  # optional noise
    return b

def render_halfhalf_colored_mask_only(out_png="phantom_rgb.png", H=256, W=256):
    # 1) Build phantom (not actually needed for this minimal path, but kept for parity with your setup)
    _ = make_halfhalf_phantom(H, W)

    # 2) Intensity maps: 1 on class pixels, 0 elsewhere (shape: 1xHxW floats)
    tissue_img = torch.zeros(1, H, W, dtype=torch.float32)
    bone_img   = torch.zeros(1, H, W, dtype=torch.float32)
    mid = W // 2
    tissue_img[:, :, :mid] = 1.0
    bone_img[:, :,  mid:]  = 1.0

    # 3) Colorize (winner-take-all) and save
    seg = compose_blend(bone_img, tissue_img, BONE_RGB, TISSUE_RGB)  # (3,H,W) in [0,1]
    torchvision.utils.save_image(seg, out_png)  # saves as RGB PNG
    print(f"Saved {out_png}")


if __name__ == "__main__":
    render_halfhalf_colored_mask_only()