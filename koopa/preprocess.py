"""Image preprocessing (raw data to pipeline input)."""

from typing import Literal
import glob
import logging
import os
import re

import czifile
import luigi
import numpy as np
import pystackreg
import skimage.io
import skimage.transform

from .config import General
from .config import PreprocessingAlignment
from .config import PreprocessingNormalization
from .registration import ReferenceAlignment


def open_czi_file(fname_czi: os.PathLike) -> np.ndarray:
    """Read .czi files as numpy array."""
    if not os.path.exists(fname_czi):
        raise ValueError(f"File {fname_czi} does not exist.")

    image = czifile.imread(fname_czi).squeeze()
    min_shape = min(image.shape[2:])
    image = image[..., :min_shape, :min_shape]
    return image


def parse_nd(filename: os.PathLike) -> dict:
    """Parse .nd configuration files as dictionary."""
    if not os.path.exists(filename):
        raise ValueError(f"File {filename} does not exist.")

    nd_data = {}
    with open(filename, "r") as file:
        for line in file.readlines():
            try:
                key, value = re.search(r'^"(.+)", "?([^"]+)"?\s$', line).groups()
                nd_data[key] = value
            except AttributeError:
                pass
    return nd_data


def open_nd_file(fname_nd: os.PathLike) -> np.ndarray:
    """Read and merge all files mentioned in one nd file as uint16."""
    nd_data = parse_nd(fname_nd)
    basename = os.path.splitext(fname_nd)[0]

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
        raise ValueError(f"Could not merge channels. Check shapes for {fname_nd}.")
    return image


def get_sharpest_slice(image: np.ndarray, axis: int = 0) -> np.ndarray:
    """Returns index of the sharpest slice in an image array."""
    sharpness = []
    for array in np.swapaxes(image, 0, axis):
        y, x = np.gradient(array)
        norm = np.sqrt(x ** 2 + y ** 2)
        sharpness.append(np.average(norm))
    sharpest = sharpness.index(max(sharpness))
    return image[sharpest]


def register_3d_image(
    image: np.ndarray, method: Literal["max", "mean", "sharpest"]
) -> np.ndarray:
    """Intensity projection to convert 3D image to 2D."""
    z_axis = image.shape.index(sorted(image.shape)[1])  # 0 is channel, 1 is z
    if method == "max":
        return image.max(axis=z_axis)
    if method == "mean":
        return image.mean(axis=z_axis)
    if method == "sharpest":
        return get_sharpest_slice(image, z_axis)
    raise ValueError(f"Unknown 3D registration method: {method}")


def crop_image(image: np.ndarray) -> np.ndarray:
    """Crop image to only observe central region."""
    if image.ndim != 4:
        raise ValueError("Image must be 4D.")
    start = PreprocessingNormalization().crop_start
    end = PreprocessingNormalization().crop_end
    return image[..., start:end, start:end]


def bin_image(image: np.ndarray) -> np.ndarray:
    """Bin image along the x and y axes."""
    bin_axes = PreprocessingNormalization().bin_axes
    if image.ndim != len(bin_axes):
        raise ValueError("bin_axes must have same shape as image.")
    return skimage.transform.rescale(image, scale=bin_axes)


class Preprocess(luigi.Task):
    """Task to open, trim, and align images."""

    FileID = luigi.Parameter()
    logger = logging.getLogger("koopa")

    def requires(self):
        requirements = {}
        if PreprocessingAlignment().enabled:
            requirements["alignment"] = ReferenceAlignment()
        return requirements

    def output(self):
        return luigi.LocalTarget(
            os.path.join(General().analysis_dir, "preprocessed", f"{self.FileID}.tif")
        )

    def run(self):
        image = self.open_image()
        image = self.register_image(image)
        skimage.io.imsave(self.output().path, image, check_contrast=False)

    @property
    def input_name(self):
        """Return the absolute path to the input file with recursive globbing."""
        files = glob.glob(
            os.path.join(
                General().image_dir, "**", f"{self.FileID}.{General().file_ext}"
            ),
            recursive=True,
        )
        if len(files) != 1:
            raise ValueError(f"Could not find unique file for {self.FileID}.")
        return files[0]

    def open_image(self) -> np.ndarray:
        """Open image file based on file extension."""
        if General().file_ext == "nd":
            return open_nd_file(self.input_name)
        if General().file_ext == "czi":
            return open_czi_file(self.input_name)
        raise ValueError(
            f"Unknown file extension: {General().file_ext}. Please use nd or czi."
        )

    def trim_image(self, image: np.ndarray) -> np.ndarray:
        """Remove first and last frames from image."""
        frame_start = PreprocessingNormalization().frame_start
        if frame_start > 0:
            image = image[:, frame_start:]
            self.logger.info(f"Trimmed before frame {frame_start}")

        frame_end = PreprocessingNormalization().frame_end - frame_start
        if frame_end > 0:
            image = image[:, :frame_end]
            self.logger.info(f"Frames after frame {frame_end}.")

        return image

    def load_alignment(self):
        """Load alignment matrix from file."""
        sr = pystackreg.StackReg(pystackreg.StackReg.RIGID_BODY)
        matrix = np.load(self.requires()["alignment"].output()[0].path)
        sr.set_matrix(matrix)
        self.sr = sr

    def align_image(self, image: np.ndarray) -> np.ndarray:
        """Align a stack of images (two channels only)."""
        idx = PreprocessingAlignment().channel_alignment

        if image.ndim == 3:
            self.logger.info(f"Aligning 2D image {self.FileID}.")
            image[idx] = self.sr.transform(image[idx])

        if image.ndim == 4:
            self.logger.info(f"Aligning stack {self.FileID}.")
            image[idx] = np.array([self.sr.transform(i) for i in image[idx]])
        return image

    def register_image(self, image: np.ndarray) -> np.ndarray:
        if image.ndim != 4:
            raise ValueError(
                f"Image {self.FileID} has {image.ndim} dimensions, expected 4."
            )

        if not General().do_3D and not General().do_TimeSeries:
            self.logger.info(f"Registering 3D to 2D image {self.FileID}.")
            image = register_3d_image(image, PreprocessingNormalization().registration)

        if General().do_3D or General().do_TimeSeries:
            self.logger.info(f"Registering 3D/2D+T image {self.FileID}.")
            image = self.trim_image(image)

        if (
            PreprocessingNormalization().crop_start
            or PreprocessingNormalization().crop_end
        ):
            self.logger.info("Cropping image.")
            image = crop_image(image)

        if PreprocessingNormalization().bin_axes:
            self.logger.info("Binning image.")
            image = bin_image(image)

        if PreprocessingAlignment().enabled:
            self.logger.info(f"Registering image {self.FileID}.")
            self.load_alignment()
            image = self.align_image(image)

        return image
