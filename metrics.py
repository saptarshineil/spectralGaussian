# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.

# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.

# For inquiries contact  george.drettakis@inria.fr


from pathlib import Path
import os
from PIL import Image
import torch
import torchvision.transforms.functional as tf
from utils.loss_utils import ssim
from lpipsPyTorch import lpips
import json
from tqdm import tqdm
from utils.image_utils import psnr, apply_colormap_viridis
from argparse import ArgumentParser


def readImages(renders_dir, gt_dir):
    renders = []
    gts = []
    image_names = []
    for fname in os.listdir(renders_dir):
        render = Image.open(renders_dir / fname)
        gt = Image.open(gt_dir / fname)
        renders.append(tf.to_tensor(render).unsqueeze(0)[:, :3, :, :].cuda())
        gts.append(tf.to_tensor(gt).unsqueeze(0)[:, :3, :, :].cuda())
        image_names.append(fname)
    return renders, gts, image_names

def find_channel_indices(method_dir):
    """Find all channel indices by intersecting gt_* and renders_* folders."""
    gt_idxs = {int(p.name.split('_')[-1]) for p in method_dir.iterdir()
               if p.is_dir() and p.name.startswith('gt_') and p.name.split('_')[-1].isdigit()}
    r_idxs  = {int(p.name.split('_')[-1]) for p in method_dir.iterdir()
               if p.is_dir() and p.name.startswith('renders_') and p.name.split('_')[-1].isdigit()}
    idxs = sorted(gt_idxs.intersection(r_idxs))
    return idxs

def evaluate(model_paths):
    full_dict = {}
    per_view_dict = {}
    print("")

    for scene_dir in model_paths:
        try:
            print("Scene:", scene_dir)
            full_dict.setdefault(scene_dir, {})
            per_view_dict.setdefault(scene_dir, {})

            test_dir = Path(scene_dir) / "train"
            if not test_dir.exists():
                print(f"  Missing {test_dir}, skipping.")
                continue

            for method in sorted(os.listdir(test_dir)):
                method_dir = test_dir / method
                if not method_dir.is_dir():
                    continue

                print("Method:", method)
                full_dict[scene_dir].setdefault(method, {})
                per_view_dict[scene_dir].setdefault(method, {})

                # Find channels that have both gt_* and renders_*
                ch_idxs = find_channel_indices(method_dir)
                if not ch_idxs:
                    print(f"  No channel folders found in {method_dir} (expect gt_*/renders_*). Skipping.")
                    continue

                # Accumulators across *all* channels for overall means
                ssims_all, psnrs_all, lpipss_all = [], [], []

                # Per-channel block
                full_dict[scene_dir][method].setdefault("per_channel", {})
                per_view_dict[scene_dir][method].setdefault("per_channel", {})

                for ch in ch_idxs:
                    gt_dir = method_dir / f"gt_{ch}"
                    renders_dir = method_dir / f"renders_{ch}"
                    if not gt_dir.exists() or not renders_dir.exists():
                        print(f"  Missing {gt_dir} or {renders_dir}, skipping channel {ch}.")
                        continue

                    renders, gts, image_names = readImages(renders_dir, gt_dir)

                    ssims, psnrs, lpipss = [], [], []
                    for idx in tqdm(range(len(renders)), desc=f"Ch {ch} metric eval"):
                        render = renders[idx]
                        gt = gts[idx]

                        # render = apply_colormap_viridis(renders[idx])
                        # gt = apply_colormap_viridis(gts[idx])

                        ssims.append(ssim(render, gt))
                        psnrs.append(psnr(render, gt))
                        lpipss.append(lpips(render, gt, net_type='vgg'))

                    # Means for this channel
                    ssim_mean  = torch.tensor(ssims).mean()
                    psnr_mean  = torch.tensor(psnrs).mean()
                    lpips_mean = torch.tensor(lpipss).mean()

                    print(f"Channel {ch}:")
                    print("  SSIM : {:>12.7f}".format(ssim_mean))
                    print("  PSNR : {:>12.7f}".format(psnr_mean))
                    print("  LPIPS: {:>12.7f}".format(lpips_mean))
                    print("")

                    # Save per-channel means
                    full_dict[scene_dir][method]["per_channel"][str(ch)] = {
                        "SSIM":  float(ssim_mean.item()),
                        "PSNR":  float(psnr_mean.item()),
                        "LPIPS": float(lpips_mean.item()),
                    }

                    # Save per-view, per-channel metrics
                    per_view_dict[scene_dir][method]["per_channel"][str(ch)] = {
                        "SSIM":  {name: float(val) for name, val in zip(image_names, torch.tensor(ssims).tolist())},
                        "PSNR":  {name: float(val) for name, val in zip(image_names, torch.tensor(psnrs).tolist())},
                        "LPIPS": {name: float(val) for name, val in zip(image_names, torch.tensor(lpipss).tolist())},
                    }

                    # Contribute to overall accumulators
                    ssims_all.extend(ssims)
                    psnrs_all.extend(psnrs)
                    lpipss_all.extend(lpipss)

                # Overall means across all channels
                if ssims_all:
                    ssim_overall  = torch.tensor(ssims_all).mean()
                    psnr_overall  = torch.tensor(psnrs_all).mean()
                    lpips_overall = torch.tensor(lpipss_all).mean()

                    print("Total (all channels):")
                    print("  SSIM : {:>12.7f}".format(ssim_overall))
                    print("  PSNR : {:>12.7f}".format(psnr_overall))
                    print("  LPIPS: {:>12.7f}".format(lpips_overall))
                    print("")

                    full_dict[scene_dir][method].update({
                        "SSIM":  float(ssim_overall.item()),
                        "PSNR":  float(psnr_overall.item()),
                        "LPIPS": float(lpips_overall.item()),
                    })
                else:
                    print("  No valid channel data found for overall metrics.")

            # Write results for this scene
            with open(os.path.join(scene_dir, "results.json"), "w") as fp:
                json.dump(full_dict[scene_dir], fp, indent=True)
            with open(os.path.join(scene_dir, "per_view.json"), "w") as fp:
                json.dump(per_view_dict[scene_dir], fp, indent=True)

        except Exception as e:
            print(f"Unable to compute metrics for model {scene_dir}: {e}")


if __name__ == "__main__":
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)

    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    parser.add_argument('--model_paths', '-m', required=True, nargs="+", type=str, default=[])
    args = parser.parse_args()
    evaluate(args.model_paths)


# from pathlib import Path
# import os
# from PIL import Image
# import torch
# import torchvision.transforms.functional as tf
# from utils.loss_utils import ssim
# from lpipsPyTorch import lpips
# import json
# from tqdm import tqdm
# from utils.image_utils import psnr, apply_colormap_viridis
# from argparse import ArgumentParser


# def readImages(renders_dir, gt_dir):
#     renders = []
#     gts = []
#     image_names = []
#     for fname in os.listdir(renders_dir):
#         render = Image.open(renders_dir / fname)
#         gt = Image.open(gt_dir / fname)
#         renders.append(tf.to_tensor(render).unsqueeze(0)[:, :3, :, :].cuda())
#         gts.append(tf.to_tensor(gt).unsqueeze(0)[:, :3, :, :].cuda())
#         image_names.append(fname)
#     return renders, gts, image_names

# def evaluate(model_paths):

#     full_dict = {}
#     per_view_dict = {}
#     full_dict_polytopeonly = {}
#     per_view_dict_polytopeonly = {}
#     print("")

#     for scene_dir in model_paths:
#         try:
#             print("Scene:", scene_dir)
#             full_dict[scene_dir] = {}
#             per_view_dict[scene_dir] = {}
#             full_dict_polytopeonly[scene_dir] = {}
#             per_view_dict_polytopeonly[scene_dir] = {}

#             test_dir = Path(scene_dir) / "train"

#             for method in os.listdir(test_dir):
#                 print("Method:", method)

#                 full_dict[scene_dir][method] = {}
#                 per_view_dict[scene_dir][method] = {}
#                 full_dict_polytopeonly[scene_dir][method] = {}
#                 per_view_dict_polytopeonly[scene_dir][method] = {}

#                 method_dir = test_dir / method
#                 gt_dir = method_dir/ "gt_2"
#                 renders_dir = method_dir / "renders_2"
#                 renders, gts, image_names = readImages(renders_dir, gt_dir)

#                 ssims = []
#                 psnrs = []
#                 lpipss = []

#                 for idx in tqdm(range(len(renders)), desc="Metric evaluation progress"):
#                     render = apply_colormap_viridis(renders[idx])
#                     gt = apply_colormap_viridis(gts[idx])
                    
#                     ssims.append(ssim(render, gt))
#                     psnrs.append(psnr(render, gt))
#                     lpipss.append(lpips(render, gt, net_type='vgg'))

#                 print("  SSIM : {:>12.7f}".format(torch.tensor(ssims).mean(), ".5"))
#                 print("  PSNR : {:>12.7f}".format(torch.tensor(psnrs).mean(), ".5"))
#                 print("  LPIPS: {:>12.7f}".format(torch.tensor(lpipss).mean(), ".5"))
#                 print("")

#                 full_dict[scene_dir][method].update({"SSIM": torch.tensor(ssims).mean().item(),
#                                                         "PSNR": torch.tensor(psnrs).mean().item(),
#                                                         "LPIPS": torch.tensor(lpipss).mean().item()})
#                 per_view_dict[scene_dir][method].update({"SSIM": {name: ssim for ssim, name in zip(torch.tensor(ssims).tolist(), image_names)},
#                                                             "PSNR": {name: psnr for psnr, name in zip(torch.tensor(psnrs).tolist(), image_names)},
#                                                             "LPIPS": {name: lp for lp, name in zip(torch.tensor(lpipss).tolist(), image_names)}})

#             with open(scene_dir + "/results.json", 'w') as fp:
#                 json.dump(full_dict[scene_dir], fp, indent=True)
#             with open(scene_dir + "/per_view.json", 'w') as fp:
#                 json.dump(per_view_dict[scene_dir], fp, indent=True)
#         except:
#             print("Unable to compute metrics for model", scene_dir)

# if __name__ == "__main__":
#     device = torch.device("cuda:0")
#     torch.cuda.set_device(device)

#     # Set up command line argument parser
#     parser = ArgumentParser(description="Training script parameters")
#     parser.add_argument('--model_paths', '-m', required=True, nargs="+", type=str, default=[])
#     args = parser.parse_args()
#     evaluate(args.model_paths)


# import os
# from pathlib import Path
# import json
# from tqdm import tqdm
# import torch
# import torchvision
# import torchvision.transforms.functional as tf
# from argparse import ArgumentParser
# from utils.loss_utils import ssim
# from utils.image_utils import psnr  # works on GPU if tensors are on cuda
# from lpipsPyTorch import lpips

# # ---------- fast GPU viridis (no NumPy/mpl, no big LUT) ----------
# @torch.no_grad()
# def viridis_gpu_unit(x: torch.Tensor) -> torch.Tensor:
#     """
#     Lightweight, LUT-free, differentiable viridis approximation.
#     Assumes x in [0,1]. Accepts (B,1,H,W) or (B,3,H,W) (returns the latter).
#     """
#     if x.ndim == 3:  # (C,H,W)
#         x = x.unsqueeze(0)
#     B, C, H, W = x.shape
#     if C == 3:
#         return x
#     assert C == 1, "Expect grayscale or RGB"
#     t = x.clamp(0, 1).squeeze(1)  # (B,H,W)

#     # 6 anchor colors from viridis (approx)
#     pos = torch.tensor([0.00, 0.25, 0.50, 0.75, 0.90, 1.00], device=x.device, dtype=x.dtype)
#     cols = torch.tensor([
#         [0.267, 0.005, 0.329],
#         [0.283, 0.141, 0.458],
#         [0.254, 0.265, 0.530],
#         [0.207, 0.372, 0.553],
#         [0.275, 0.633, 0.437],
#         [0.993, 0.906, 0.144],
#     ], device=x.device, dtype=x.dtype)  # (6,3)

#     idx = torch.bucketize(t, pos) - 1
#     idx = idx.clamp(0, pos.numel() - 2)
#     p0, p1 = pos[idx], pos[idx + 1]
#     w = (t - p0) / (p1 - p0 + 1e-8)

#     c0, c1 = cols[idx], cols[idx + 1]      # (B,H,W,3)
#     rgb = (1 - w.unsqueeze(-1)) * c0 + w.unsqueeze(-1) * c1
#     return rgb.permute(0, 3, 1, 2).contiguous()  # (B,3,H,W)


# def read_dir_as_tensor_batch(renders_dir: Path, gt_dir: Path, device: torch.device):
#     """Load a folder of PNGs as tensors on GPU. Keeps original order by sorting filenames."""
#     fnames = sorted([f for f in os.listdir(renders_dir) if f.lower().endswith(('.png','.jpg','.jpeg','.tif','.tiff'))])
#     renders, gts = [], []
#     for fname in fnames:
#         r = torchvision.io.read_image(str(renders_dir / fname))  # (C,H,W), uint8 or 16-bit -> auto
#         g = torchvision.io.read_image(str(gt_dir / fname))
#         # convert to float in [0,1] and ensure 1- or 3-channel
#         r = r.float()
#         g = g.float()
#         if r.dtype != torch.float32:
#             r = r.to(torch.float32)
#         if g.dtype != torch.float32:
#             g = g.to(torch.float32)

#         # Scale by bit depth if needed (read_image gives 0..255 for 8-bit, 0..65535 for 16-bit)
#         # We normalize by max possible value inferred from dtype range; here we assume uint8/16 already to float.
#         # Safer approach: divide by 255 if max>255 else by 65535 — but most PNGs are 8-bit.
#         # If your GT is true float saved to PNG, clamp later anyway.
#         r = (r / 255.0).clamp(0, 1)
#         g = (g / 255.0).clamp(0, 1)

#         # If grayscale, expand to (1,H,W); if more than 3 channels, keep first 3
#         if r.shape[0] == 1:
#             pass
#         elif r.shape[0] > 3:
#             r = r[:3]
#         if g.shape[0] == 1:
#             pass
#         elif g.shape[0] > 3:
#             g = g[:3]

#         renders.append(r.unsqueeze(0))
#         gts.append(g.unsqueeze(0))
#     if not renders:
#         return None, None, []
#     R = torch.cat(renders, dim=0).to(device)  # (N,C,H,W)
#     G = torch.cat(gts, dim=0).to(device)
#     return R, G, fnames


# def find_channel_indices(method_dir: Path):
#     gt_idxs = {int(p.name.split('_')[-1]) for p in method_dir.iterdir()
#                if p.is_dir() and p.name.startswith('gt_') and p.name.split('_')[-1].isdigit()}
#     r_idxs  = {int(p.name.split('_')[-1]) for p in method_dir.iterdir()
#                if p.is_dir() and p.name.startswith('renders_') and p.name.split('_')[-1].isdigit()}
#     return sorted(gt_idxs.intersection(r_idxs))


# @torch.no_grad()
# def evaluate(model_paths, *, device="cuda:0", batch_size=16, use_colormap=True):
#     dev = torch.device(device)
#     full_dict = {}
#     per_view_dict = {}
#     print("")

#     # Instantiate LPIPS once on GPU
#     lpips_fn = lambda x, y: lpips(x, y, net_type='vgg')

#     for scene_dir in model_paths:
#         try:
#             print("Scene:", scene_dir)
#             full_dict.setdefault(scene_dir, {})
#             per_view_dict.setdefault(scene_dir, {})

#             test_dir = Path(scene_dir) / "train"
#             if not test_dir.exists():
#                 print(f"  Missing {test_dir}, skipping.")
#                 continue

#             for method in sorted(os.listdir(test_dir)):
#                 method_dir = test_dir / method
#                 if not method_dir.is_dir():
#                     continue

#                 print("Method:", method)
#                 full_dict[scene_dir].setdefault(method, {})
#                 per_view_dict[scene_dir].setdefault(method, {})
#                 full_dict[scene_dir][method].setdefault("per_channel", {})
#                 per_view_dict[scene_dir][method].setdefault("per_channel", {})

#                 ch_idxs = find_channel_indices(method_dir)
#                 if not ch_idxs:
#                     print(f"  No channel folders found in {method_dir} (expect gt_*/renders_*). Skipping.")
#                     continue

#                 # accumulators for overall means
#                 all_ssim_vals, all_psnr_vals, all_lpips_vals = [], [], []

#                 for ch in ch_idxs:
#                     gt_dir = method_dir / f"gt_{ch}"
#                     r_dir  = method_dir / f"renders_{ch}"
#                     if not gt_dir.exists() or not r_dir.exists():
#                         print(f"  Missing {gt_dir} or {r_dir}, skipping channel {ch}.")
#                         continue

#                     R, G, names = read_dir_as_tensor_batch(r_dir, gt_dir, dev)
#                     if R is None:
#                         print(f"  No images in {r_dir}, skipping channel {ch}.")
#                         continue

#                     # If grayscale, colorize (or promote to 3-ch) on GPU
#                     if use_colormap:
#                         if R.shape[1] == 1:
#                             R_rgb = viridis_gpu_unit(R)
#                         else:
#                             R_rgb = R[:, :3]
#                         if G.shape[1] == 1:
#                             G_rgb = viridis_gpu_unit(G)
#                         else:
#                             G_rgb = G[:, :3]
#                     else:
#                         # ensure 3-channel for LPIPS/SSIM (repeat grayscale)
#                         R_rgb = R.repeat(1, 3, 1, 1) if R.shape[1] == 1 else R[:, :3]
#                         G_rgb = G.repeat(1, 3, 1, 1) if G.shape[1] == 1 else G[:, :3]

#                     N = R_rgb.shape[0]
#                     ssim_vals = []
#                     psnr_vals = []
#                     lpips_vals = []

#                     # batched metric computation
#                     for i in tqdm(range(0, N, batch_size), desc=f"Ch {ch} metric eval"):
#                         r_batch = R_rgb[i:i+batch_size]
#                         g_batch = G_rgb[i:i+batch_size]

#                         # SSIM from your utils; many implementations return mean over batch.
#                         # To keep per-image scores, compute per-image by reshaping loops in small batches.
#                         ssim_b = ssim(r_batch, g_batch)  # expected to be scalar mean; if so, expand below
#                         # If your ssim() returns mean over batch, approximate per-image by duplicating that value:
#                         # (For exact per-image SSIM, switch to a per-image SSIM or torchmetrics SSIM with reduction='none')
#                         ssim_vals.extend([float(ssim_b)] * r_batch.size(0))

#                         # PSNR per-image (vectorized)
#                         mse = torch.mean((r_batch - g_batch) ** 2, dim=(1,2,3))
#                         psnr_b = 10.0 * torch.log10(1.0 / (mse + 1e-12))
#                         psnr_vals.extend(psnr_b.tolist())

#                         # LPIPS per-image
#                         # lpips() from lpipsPyTorch accepts batches and returns (B,1,1,1) or (B,)
#                         lp_b = lpips_fn(r_batch, g_batch)        # returns (B,) or (B,1,1,1) depending on impl
#                         lp_b = lp_b.view(-1)
#                         lpips_vals.extend(lp_b.tolist())

#                     # per-channel means
#                     ssim_mean  = float(torch.tensor(ssim_vals, device=dev).mean().item())
#                     psnr_mean  = float(torch.tensor(psnr_vals, device=dev).mean().item())
#                     lpips_mean = float(torch.tensor(lpips_vals, device=dev).mean().item())

#                     print(f"Channel {ch}:")
#                     print("  SSIM : {:>12.7f}".format(ssim_mean))
#                     print("  PSNR : {:>12.7f}".format(psnr_mean))
#                     print("  LPIPS: {:>12.7f}".format(lpips_mean))
#                     print("")

#                     full_dict[scene_dir][method]["per_channel"][str(ch)] = {
#                         "SSIM":  ssim_mean,
#                         "PSNR":  psnr_mean,
#                         "LPIPS": lpips_mean,
#                     }

#                     # per-view, per-channel metrics (as in your original JSON)
#                     per_view_dict[scene_dir][method]["per_channel"][str(ch)] = {
#                         "SSIM":  {name: val for name, val in zip(names, ssim_vals)},
#                         "PSNR":  {name: val for name, val in zip(names, psnr_vals)},
#                         "LPIPS": {name: val for name, val in zip(names, lpips_vals)},
#                     }

#                     all_ssim_vals.extend(ssim_vals)
#                     all_psnr_vals.extend(psnr_vals)
#                     all_lpips_vals.extend(lpips_vals)

#                 # overall means across channels
#                 if all_ssim_vals:
#                     ssim_overall  = float(torch.tensor(all_ssim_vals, device=dev).mean().item())
#                     psnr_overall  = float(torch.tensor(all_psnr_vals, device=dev).mean().item())
#                     lpips_overall = float(torch.tensor(all_lpips_vals, device=dev).mean().item())

#                     print("Total (all channels):")
#                     print("  SSIM : {:>12.7f}".format(ssim_overall))
#                     print("  PSNR : {:>12.7f}".format(psnr_overall))
#                     print("  LPIPS: {:>12.7f}".format(lpips_overall))
#                     print("")

#                     full_dict[scene_dir][method].update({
#                         "SSIM":  ssim_overall,
#                         "PSNR":  psnr_overall,
#                         "LPIPS": lpips_overall,
#                     })

#             # write JSONs for this scene
#             with open(os.path.join(scene_dir, "results.json"), "w") as fp:
#                 json.dump(full_dict[scene_dir], fp, indent=True)
#             with open(os.path.join(scene_dir, "per_view.json"), "w") as fp:
#                 json.dump(per_view_dict[scene_dir], fp, indent=True)

#         except Exception as e:
#             print(f"Unable to compute metrics for model {scene_dir}: {e}")

# if __name__ == "__main__":
#     device = torch.device("cuda:0")
#     torch.cuda.set_device(device)

#     # Set up command line argument parser
#     parser = ArgumentParser(description="Training script parameters")
#     parser.add_argument('--model_paths', '-m', required=True, nargs="+", type=str, default=[])
#     args = parser.parse_args()
#     evaluate(args.model_paths)