#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

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
                        # render = renders[idx]
                        # gt = gts[idx]

                        render = apply_colormap_viridis(renders[idx])
                        gt = apply_colormap_viridis(gts[idx])

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
#                 gt_dir = method_dir/ "gt_0"
#                 renders_dir = method_dir / "renders_0"
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