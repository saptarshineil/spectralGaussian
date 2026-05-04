import numpy as np
import pyvista as pv


vol_path = "data_generator/real_dataset/output/mouse_forelimb/vol_gt.npy"
# vol_path = "data_generator/real_dataset/output/lizard_reprojected_fdk_fixed/vol_gt.npy"


vol = np.load(vol_path)

print(vol.min(), vol.max(), vol.mean())

plotter = pv.Plotter(window_size=[800, 800], line_smoothing=True, off_screen=False)
plotter.add_volume(vol, cmap="viridis", opacity="linear")
plotter.show()
