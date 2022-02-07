from typing import List, Literal
import glob
import logging
import os
import re

import luigi
import matplotlib.pyplot as plt
import numpy as np
import pystackreg
import skimage.io

from config import General
from config import PreprocessingAlignment
from config import PreprocessingNormalization
from setup import SetupPipeline


def parse_nd(filename: str) -> dict:
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


def open_nd_file(fname_nd: str) -> np.ndarray:
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


class ReferenceAlignment(luigi.Task):
    def output(self):
        return [
            luigi.LocalTarget(os.path.join(General().analysis_dir, "alignment.npy")),
            luigi.LocalTarget(
                os.path.join(General().analysis_dir, "alignment.png"),
            ),
        ]

    def run(self):
        self.load_images()
        self.register_alignment_matrix()
        self.save_alignment()
        self.plot_alignment()

    @staticmethod
    def read_stack(fnames: List[os.PathLike]) -> np.ndarray:
        """Maximum project a stack of image files."""
        images = [skimage.io.imread(f) for f in fnames]
        return np.max(images, axis=0)

    def load_images(self) -> None:
        """Load reference and alignment images."""
        fnames = sorted(
            glob.glob(os.path.join(PreprocessingAlignment().alignment_dir, "*.tif"))
        )
        channel_reference = PreprocessingAlignment().channel_reference + 1
        channel_transform = PreprocessingAlignment().channel_alignment + 1
        fnames_reference = [i for i in fnames if f"w{channel_reference}conf" in i]
        fnames_transform = [i for i in fnames if f"w{channel_transform}conf" in i]
        self.image_reference = self.read_stack(fnames_reference)
        self.image_transform = self.read_stack(fnames_transform)

    def register_alignment_matrix(self) -> None:
        """Calculate a rigid body transformation matrix."""
        sr = pystackreg.StackReg(pystackreg.StackReg.RIGID_BODY)
        sr.register(self.image_reference, self.image_transform)
        self.sr = sr

    def save_alignment(self) -> None:
        """Save alignment matrix to file."""
        np.save(self.output()[0].path, self.sr.get_matrix())

    def plot_alignment(self) -> None:
        """Plot chromatic transform before and after alignment."""
        transform = self.sr.transform(self.image_transform)
        _, ax = plt.subplots(1, 2, figsize=(20, 20))
        ax[0].set_title("Pre-alignment")
        ax[0].imshow(self.image_reference, cmap="Greens")
        ax[0].imshow(self.image_transform, cmap="Reds", alpha=0.5)
        ax[1].set_title("Post-alignment")
        ax[1].imshow(self.image_reference, cmap="Greens")
        ax[1].imshow(transform, cmap="Reds", alpha=0.5)
        plt.savefig(self.output()[1].path, bbox_inches="tight")
        plt.close()


class Preprocess(luigi.Task):
    """Task to open, trim, and align images."""

    FileID = luigi.Parameter()
    logger = logging.getLogger("luigi-interface")

    def requires(self):
        requirements = {"setup": SetupPipeline()}
        if PreprocessingAlignment().enabled:
            requirements["alignment"] = ReferenceAlignment()
        return requirements

    def output(self):
        return luigi.LocalTarget(
            os.path.join(General().analysis_dir, "preprocessed", f"{self.FileID}.tif")
        )

    def run(self):
        input_name = os.path.join(General().image_dir, f"{self.FileID}.nd")
        image = open_nd_file(input_name)
        image = self.register_image(image)
        skimage.io.imsave(self.output().path, image, check_contrast=False)

    def register_image(self, image: np.ndarray) -> np.ndarray:
        if image.ndim != 4:
            raise ValueError(
                f"Image {self.FileID} has {image.ndim} dimensions, expected 4."
            )

        if not General().do_3D and not General().do_TimeSeries:
            self.logger.info(f"Registering 3D to 2D image {self.FileID}.")
            image = register_3d_image(image, PreprocessingNormalization().registration)

        if General().do_3D:
            self.logger.info(f"Registering 3D image {self.FileID}.")
            trim_frames = PreprocessingNormalization().remove_n_frames
            if trim_frames > 0 and trim_frames < image.shape[1]:
                image = image[:, trim_frames:-trim_frames]
            else:
                self.logger.info(f"No frames removed from {self.FileID}.")

        if General().do_TimeSeries and not General().do_3D:
            self.logger.info(f"Registering time series {self.FileID}.")
            frame_start = PreprocessingNormalization().frame_start
            frame_end = PreprocessingNormalization().frame_end
            image = image[:, frame_start:frame_end]

        if PreprocessingAlignment().enabled:
            self.load_alignment()
            image = self.align_image(image)

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
