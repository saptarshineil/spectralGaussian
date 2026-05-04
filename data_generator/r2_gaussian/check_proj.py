import os
import os.path as osp
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import torch
from PIL import Image


proj_path = "data_generator/real_dataset/output/mouse_forelimb/spectra/spectral_2"
# proj_path = "data_generator/real_dataset/output/lizard_reprojected_fdk_fixed/spectra/spectral_0"
# proj_path = "data/ct_spec_new/quail/spectra/spectral_2"

# proj_path = "data_generator/real_dataset/output/iodine_phantom_temp/0"
# proj_path = "data_generator/real_dataset/output/iodine_phantom/spectra/spectral_0"

# proj_path = "data_generator/real_dataset/intermediate/lizard/spectral_0"

proj_list = sorted(
    filename
    for filename in os.listdir(proj_path)
    if filename.lower().endswith((".tif", ".tiff", ".npy"))
)

# Load .tif files using PIL instead of np.load
def load_image(filepath):
    if filepath.endswith('.tif') or filepath.endswith('.tiff'):
        return np.array(Image.open(filepath))
    else:
        return np.load(filepath)

projs = np.stack([load_image(osp.join(proj_path, proj_id)) for proj_id in proj_list], axis=-1)


def t2a(tensor):
    if torch.is_tensor(tensor):
        return tensor.detach().cpu().numpy()
    else:
        return tensor

def show_one_volume(volume1, cmap="viridis"):
    if torch.is_tensor(volume1):
        volume1 = t2a(volume1)
    # Create a figure and axis
    fig, ax = plt.subplots(1, 1)
    plt.subplots_adjust(bottom=0.25)

    # Initial slice index
    initial_slice = int(volume1.shape[2] / 2)

    # Display the initial slices
    img1 = ax.imshow(
        volume1[:, :, initial_slice], cmap=cmap, vmin=0.0, vmax=volume1.max()
    )

    # Add colorbars
    cbar1 = fig.colorbar(img1, ax=ax, orientation="vertical")

    # Add a single slider for both volumes
    ax_slider = plt.axes([0.1, 0.01, 0.65, 0.03], facecolor="lightgoldenrodyellow")
    slice_slider = Slider(
        ax_slider,
        "Slice",
        0,
        volume1.shape[2] - 1,
        valinit=initial_slice,
        valstep=1,
    )

    # Update function for the slider
    def update(val):
        slice_index = int(slice_slider.val)
        img1.set_array(volume1[:, :, slice_index])
        fig.canvas.draw_idle()

    # Connect the slider to the update function
    slice_slider.on_changed(update)

    # Show the plot
    plt.show()

show_one_volume(projs)
