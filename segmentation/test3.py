# test_tri_phantom.py
import torch
import torchvision

# Import the multi-class composers from your tri-material module
from render_materials_tri import compose_blend_multi, compose_hard_multi

# Your palette (matches render_materials_tri.py defaults)
TISSUE_RGB = (230, 60, 60)    # red
BONE_RGB   = (0, 110, 255)    # blue
METAL_RGB  = (255, 215, 0)    # yellow

def render_thirds_colored_mask_only(
    out_prefix: str = "phantom_tri",
    H: int = 256,
    W: int = 360,
    blend_mode: str = "independent",  # "independent" or "proportional"
):
    """
    Creates a (H x W) image split into three vertical regions:
      left=Soft Tissue, middle=Bone, right=Metal
    Then colorizes with either blended or hard labels.
    """
    # 1) Build simple 1-channel "intensity" maps per material (shape: 1xHxW)
    third = W // 3
    tiss = torch.zeros(1, H, W, dtype=torch.float32)   # Soft Tissue
    bone = torch.zeros(1, H, W, dtype=torch.float32)   # Bone
    metal= torch.zeros(1, H, W, dtype=torch.float32)   # Metal

    tiss[:, :, :third]           = 1.0
    bone[:, :, third:2*third]    = 1.0
    metal[:, :, 2*third:]        = 1.0

    # Optional tiny noise so normalization paths get exercised a bit
    noise_level = 0.02
    for m in (tiss, bone, metal):
        m += noise_level * torch.randn_like(m)

    # 2) Blend and Hard colorizations
    seg_blend = compose_blend_multi(
        [metal, bone, tiss],           # order matters: must match colors
        [METAL_RGB, BONE_RGB, TISSUE_RGB],
        mode=blend_mode
    )
    seg_hard = compose_hard_multi(
        [metal, bone, tiss],
        [METAL_RGB, BONE_RGB, TISSUE_RGB]
    )

    # 3) Save PNGs
    out_blend = f"{out_prefix}_blend_{blend_mode}.png"
    out_hard  = f"{out_prefix}_hard.png"
    torchvision.utils.save_image(seg_blend, out_blend)
    torchvision.utils.save_image(seg_hard,  out_hard)
    print(f"Saved {out_blend}")
    print(f"Saved {out_hard}")

if __name__ == "__main__":
    # Default run: independent blending + hard
    render_thirds_colored_mask_only()
    # Also try proportional blending (weights sum to 1 across classes per pixel)
    render_thirds_colored_mask_only(out_prefix="phantom_tri_prop", blend_mode="proportional")
