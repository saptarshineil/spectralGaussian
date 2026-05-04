import argparse
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt


pcd_path = "data_generator/real_dataset/output/mouse_forelimb/init_pcd.npy"
max_points = 300000


def load_pcd(path):
    data = np.load(path)
    if data.ndim != 2 or data.shape[1] < 3:
        raise ValueError(f"Expected Nx3 or Nx4 array, got shape {data.shape}")
    points = data[:, :3]
    density = data[:, 3] if data.shape[1] > 3 else None
    return points, density


def sample_points(points, density, max_points):
    if max_points and points.shape[0] > max_points:
        idx = np.random.choice(points.shape[0], max_points, replace=False)
        points = points[idx]
        if density is not None:
            density = density[idx]
    return points, density


def density_to_colors(density):
    if density is None:
        return None
    low, high = np.percentile(density, [2, 98])
    if low == high:
        low = density.min()
        high = density.max() if density.max() > low else low + 1.0
    norm = np.clip((density - low) / (high - low), 0.0, 1.0)
    return plt.get_cmap("viridis")(norm)[:, :3]



points, density = load_pcd(pcd_path)
points, density = sample_points(points, density, max_points)

pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(points)
colors = density_to_colors(density)
if colors is not None:
    # Color points by density for quick visual inspection.
    pcd.colors = o3d.utility.Vector3dVector(colors)

o3d.visualization.draw_geometries([pcd], window_name="init_pcd")
