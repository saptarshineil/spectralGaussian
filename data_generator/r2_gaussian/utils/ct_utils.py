import numpy as np
import tigre
import tigre.algorithms as algs



def get_geometry_tigre(cfg):
    """For TIGRE only."""
    if cfg["mode"] == "parallel":
        geo = tigre.geometry(mode="parallel", nVoxel=np.array(cfg["nVoxel"][::-1]))
    elif cfg["mode"] == "cone":
        geo = tigre.geometry(mode="cone")
    else:
        raise NotImplementedError("Unsupported scanner mode!")

    geo.DSD = cfg["DSD"]  # Distance Source Detector
    geo.DSO = cfg["DSO"]  # Distance Source Origin
    # Detector parameters
    geo.nDetector = np.array(cfg["nDetector"])  # number of pixels
    geo.sDetector = np.array(cfg["sDetector"])  # size of each pixel
    geo.dDetector = geo.sDetector / geo.nDetector  # total size of the detector
    # Image parameters
    geo.nVoxel = np.array(cfg["nVoxel"][::-1])  # number of voxels
    geo.sVoxel = np.array(cfg["sVoxel"][::-1])  # size of each voxel
    geo.dVoxel = geo.sVoxel / geo.nVoxel  # total size of the image
    # Offsets
    geo.offOrigin = np.array(cfg["offOrigin"][::-1])  # Offset of image from origin
    geo.offDetector = np.array(
        [cfg["offDetector"][1], cfg["offDetector"][0], 0]
    )  # Offset of Detector
    # Auxiliary
    geo.accuracy = cfg["accuracy"]  # Accuracy of FWD proj
    # Mode
    geo.filter = cfg["filter"]
    return geo


def recon_volume(projs, angles, geo, recon_method):
    """Reconstruct ct with traditional methods."""
    if recon_method == "fdk":
        vol = algs.fdk(projs[:, ::-1, :], geo, angles)
    elif recon_method == "cgls":
        vol, _ = algs.cgls(projs[:, ::-1, :], geo, angles, 60, computel2=True)
    else:
        raise ValueError("Unsupported reconstruction method")
    vol = np.transpose(vol, (2, 1, 0))
    return vol
