from typing import Literal
import os
import re

import luigi
import numpy as np
import skimage.io

from config import globalConfig


def create_directories(basedir: str, spot_indexes: list):
    """Create all analysis directories for a given path."""
    directories = ["preprocessed", "segmentation_cells"]
    directories.extend([f"detection_c{i}" for i in spot_indexes])
    for folder in directories:
        path = os.path.join(basedir, folder)
        if not os.path.exists(path):
            os.makedirs(path)


def parse_nd(filename: str) -> dict:
    """Parse .nd configuration files as dictionary."""
    nd_data = {}
    with open(filename, "r") as file:
        for line in file.readlines():
            try:
                key, value = re.search(r'^"(.+)", "?([^"]+)"?\s$', line).groups()
                nd_data[key] = value
            except AttributeError:
                pass
    return nd_data


def open_nd_file(fname_nd: str) -> np.ndarray:
    """Read and merge all files mentioned in one nd file."""
    nd_data = parse_nd(fname_nd)
    basename = os.path.splitext(fname_nd)[0]

    # Parse channels
    channels = int(nd_data["NWavelengths"])
    images = []

    for channel in range(1, channels + 1):
        channel_name = nd_data[f"WaveName{channel}"]
        fname_image = f"{basename}_w{channel}{channel_name}.stk"
        image = skimage.io.imread(fname_image)
        images.append(image)

    # Merge channels
    try:
        image = np.stack(images, axis=0)
    except ValueError:
        raise ValueError(f"Could not merge channels. Check shapes for {fname_nd}.")
    return image


def z_project(image: np.ndarray, method: Literal["max", "mean"]) -> np.ndarray:
    """Maximum intensity projection along the z-axis."""
    z_axis = image.shape.index(min(image.shape))
    if method == "max":
        return image.max(axis=z_axis)
    elif method == "mean":
        return image.mean(axis=z_axis)
    raise ValueError(f"Unknown z-projection method: {method}")


class Preprocess(luigi.Task):
    """Task to open, crop, bin, and arrange raw images."""

    FileID = luigi.Parameter()

    @property
    def input_name(self):
        input_name = os.path.join(globalConfig().ImageDir, f"{self.FileID}.nd")
        if not os.path.exists(input_name):
            raise ValueError(f"File {input_name} does not exist.")
        return input_name

    @property
    def output_name(self):
        output_name = os.path.join(
            globalConfig().AnalysisDir, "preprocessed", f"{self.FileID}.tif"
        )
        return output_name

    def requires(self):
        return None

    def output(self):
        return luigi.LocalTarget(self.output_name)

    def run(self):
        create_directories(
            basedir=globalConfig().AnalysisDir, spot_indexes=globalConfig().ChannelSpots
        )
        image = open_nd_file(self.input_name)
        image = np.array([z_project(c, globalConfig().ZProjection) for c in image])
        skimage.io.imsave(self.output_name, image, check_contrast=False)

