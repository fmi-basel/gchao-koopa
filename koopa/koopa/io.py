"""I/O functions for all filetypes used in koopa."""

import configparser
import glob
import os
import re

import czifile
import numpy as np
import pandas as pd
import pystackreg
import skimage.io
import tifffile

from . import align


# TODO add more logic / help to ensure only creating subdirs etc?
def create_path(fname: os.PathLike):
    path = os.path.dirname(fname)
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def find_full_path(path: os.PathLike, fname: str, file_ext: str):
    """Return the absolute path to the input file with recursive globbing."""
    files = glob.glob(os.path.join(path, "**", f"{fname}.{file_ext}"), recursive=True)
    if len(files) != 1:
        raise ValueError(f"Could not find unique file for {fname}.")
    return files[0]


def basename(path: os.PathLike) -> str:
    """Returns the basename removing path and extension."""
    return os.path.splitext(os.path.basename(path))[0]


def load_czi(fname: os.PathLike) -> np.ndarray:
    """Read .czi files as numpy array."""
    if not os.path.exists(fname):
        raise ValueError(f"File {fname} does not exist.")

    image = czifile.imread(fname).squeeze()
    min_shape = min(image.shape[2:])
    image = image[..., :min_shape, :min_shape]
    return image


def parse_nd(fname: os.PathLike) -> dict:
    """Parse .nd configuration files as dictionary."""
    if not os.path.exists(fname):
        raise ValueError(f"File {fname} does not exist.")

    nd_data = {}
    with open(fname, "r") as file:
        for line in file.readlines():
            try:
                key, value = re.search(r'^"(.+)", "?([^"]+)"?\s$', line).groups()
                nd_data[key] = value
            except AttributeError:
                pass
    return nd_data


def load_nd(fname: os.PathLike) -> np.ndarray:
    """Read and merge all files mentioned in one nd file as uint16."""
    nd_data = parse_nd(fname)
    basename = os.path.splitext(fname)[0]

    # Parse channels
    channels = int(nd_data["NWavelengths"])
    images = []

    for channel in range(1, channels + 1):
        channel_name = nd_data[f"WaveName{channel}"]
        basename_image = f"{basename}_w{channel}{channel_name}"
        fname_image = (
            f"{basename_image}.stk"
            if os.path.isfile(f"{basename_image}.stk")
            else f"{basename_image}.tif"
        )
        image = skimage.io.imread(fname_image).astype(np.uint16)
        images.append(image)

    # Merge channels
    try:
        image = np.stack(images, axis=0)
    except ValueError:
        raise ValueError(f"Could not merge channels. Check shapes for {fname}.")
    return image


def load_tif(fname: os.PathLike) -> np.ndarray:
    """Read a single tif file and add dimensions."""
    image = skimage.io.imread(fname).astype(np.uint16)
    while image.ndim < 4:
        image = np.expand_dims(image, axis=0)
    return image


def load_stk(fname: os.PathLike) -> np.ndarray:
    """Wrapper for tif loader with stk extension."""
    return load_tif(fname)


def load_raw_image(fname: str, file_ext: str) -> np.ndarray:
    """Open image file based on file extension."""
    if file_ext == "nd":
        return load_nd(fname)
    if file_ext == "czi":
        return load_czi(fname)
    if file_ext == "stk":
        return load_stk(fname)
    if file_ext == "tif":
        return load_tif(fname)
    raise ValueError(
        f"Unknown file extension: {file_ext}. Please use nd, czi, stk or tif."
    )


def load_image(fname: os.PathLike) -> np.ndarray:
    """Open a tif file with image or segmentation data."""
    if os.path.splitext(fname)[1] != ".tif":
        raise ValueError(f'Image file must end in ".tif". {fname} does not.')
    return tifffile.imread(fname)


def load_parquet(fname: os.PathLike) -> pd.DataFrame:
    """Open a parquet file."""
    return pd.read_parquet(fname)


def load_alignment(fname: os.PathLike) -> pystackreg.StackReg:
    """Create alignment matrix from file."""
    matrix = np.load(fname)
    return align.get_stackreg(matrix)


def load_config(fname: os.PathLike) -> configparser.ConfigParser:
    """Load configuration file."""
    if not os.path.isfile(fname):
        raise ValueError(f"Configuration file must exist. {fname} does not.")
    if os.path.splitext(fname)[1] != ".cfg":
        raise ValueError(f'Configuration file must end in ".cfg". {fname} does not.')

    config = configparser.ConfigParser()
    config.read(fname)
    return config


def save_alignment(fname: os.PathLike, sr: pystackreg.StackReg):
    """Save alignment matrix to file."""
    np.save(fname, sr.get_matrix())


def save_image(fname: os.PathLike, image: np.ndarray):
    """Save image to disk."""
    create_path(fname)
    skimage.io.imsave(fname, image, check_contrast=False)


def save_config(fname: os.PathLike, config: configparser.ConfigParser):
    """Save configuration file to disk."""
    with open(fname, "w") as f:
        config.write(f)


def save_parquet(fname: os.PathLike, df: pd.DataFrame):
    """Save parquet file to disk."""
    create_path(fname)
    df.to_parquet(fname)


def save_csv(fname: os.PathLike, df: pd.DataFrame):
    """Save csv file to disk."""
    df.to_csv(fname, index=False)
