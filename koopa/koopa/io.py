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
    files = glob.glob(
        os.path.join(path, "**", f"{fname}.{file_ext}"),
        recursive=True,
    )
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
        fname_image = f"{basename}_w{channel}{channel_name}.stk"
        image = skimage.io.imread(fname_image).astype(np.uint16)
        images.append(image)

    # Merge channels
    try:
        image = np.stack(images, axis=0)
    except ValueError:
        raise ValueError(f"Could not merge channels. Check shapes for {fname}.")
    return image


def load_raw_image(fname: str, file_ext: str) -> np.ndarray:
    """Open image file based on file extension."""
    if file_ext == "nd":
        return load_nd(fname)
    if file_ext == "czi":
        return load_czi(fname)
    raise ValueError(f"Unknown file extension: {file_ext}. Please use nd or czi.")


def load_image(fname: os.PathLike) -> np.ndarray:
    return tifffile.imread(fname)


def load_parquet(fname: os.PathLike) -> pd.DataFrame:
    return pd.read_parquet(fname)


def load_alignment(fname: os.PathLike) -> pystackreg.StackReg:
    """Load alignment matrix from file."""
    matrix = np.load(fname)
    return align.get_stackreg(matrix)


def load_config(fname: os.PathLike) -> dict:
    config = configparser.ConfigParser()
    config.read(fname)
    return config


def save_alignment(fname: os.PathLike, sr: pystackreg.StackReg):
    """Save alignment matrix to file."""
    np.save(fname, sr.get_matrix())


def save_image(fname: os.PathLike, image: np.ndarray):
    create_path(fname)
    skimage.io.imsave(fname, image, check_contrast=False)


def save_parquet(fname: os.PathLike, df: pd.DataFrame):
    create_path(fname)
    df.to_parquet(fname)


def save_config(fname: os.PathLike, config: configparser.ConfigParser):
    with open(fname, "w") as f:
        config.write(f)


def save_csv(fname: os.PathLike, df: pd.DataFrame):
    df.to_csv(fname, index=False)
