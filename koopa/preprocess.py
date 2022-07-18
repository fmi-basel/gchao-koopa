"""Image preprocessing."""

from typing import Iterable, Literal
import glob
import logging
import os
import re

import czifile
import deepblink as pink
import luigi
import matplotlib.pyplot as plt
import numpy as np
import pystackreg
import scipy.optimize
import scipy.spatial
import skimage.io

from .config import General
from .config import PreprocessingAlignment
from .config import PreprocessingNormalization
from .setup import SetupPipeline


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


def register_coordinates(
    reference: np.ndarray, transform: np.ndarray, distance_cutoff: float = 3.0
):
    """Match coordinates of reference to coordinates of transform below a distance."""
    cdist = scipy.spatial.distance.cdist(reference, transform, metric="euclidean")
    rows, cols = scipy.optimize.linear_sum_assignment(cdist)
    for r, c in zip(rows, cols):
        if cdist[r, c] > distance_cutoff:
            rows = rows[rows != r]
            cols = cols[cols != c]
    return reference[rows], transform[cols]


def compute_affine_transform(reference: Iterable, transform: Iterable) -> np.ndarray:
    """Compute the affine transform by point set registration.

    The affine transform is the composition of a translation and a linear map.
    The two ordered lists of points must be of the same length larger or equal to 3.
    The order of the points in the two list must match.

    The 2D affine transform A has 6 parameters (2 for the translation and 4 for the
    linear transform). The best estimate of A can be computed using at least 3 pairs
    of matching points. Adding more pair of points will improve the quality of the
    estimate. The matching pairs are usually obtained by selecting unique features
    in both images and measuring their coordinates.

    Credit: Will Lenthe and Pymicro
    """
    assert len(reference) == len(transform)
    assert len(reference) >= 3
    fixed_centroid = np.average(reference, 0)
    moving_centroid = np.average(transform, 0)

    # Offset every point by the center of mass of all the points in the set
    fixed_from_centroid = reference - fixed_centroid
    moving_from_centroid = transform - moving_centroid
    covariance = moving_from_centroid.T.dot(fixed_from_centroid)
    variance = moving_from_centroid.T.dot(moving_from_centroid)

    # Compute the full affine transform: translation + linear map
    linear_map = np.linalg.inv(variance).dot(covariance).T
    translation = fixed_centroid - linear_map.dot(moving_centroid)

    # Create affine transform matrix
    matrix = np.zeros((3, 3))
    matrix[:2, :2] = linear_map
    matrix[:2, 2] = translation.T
    matrix[2, 2] = 1
    return matrix


class ReferenceAlignment(luigi.Task):
    def output(self):
        return [
            luigi.LocalTarget(os.path.join(General().analysis_dir, "alignment.npy")),
            luigi.LocalTarget(
                os.path.join(General().analysis_dir, "alignment.png"),
            ),
        ]

    def run(self):
        self.sr = pystackreg.StackReg(pystackreg.StackReg.AFFINE)
        self.get_image_files()
        if PreprocessingAlignment().method == "pystackreg":
            self.register_alignment_pystackreg()
        elif PreprocessingAlignment().method == "deepblink":
            self.register_alignment_deepblink()
        else:
            raise ValueError(
                f"Unknown alignment method: {PreprocessingAlignment().method}"
            )
        self.save_alignment()
        self.plot_alignment()

    def get_image_files(self) -> None:
        """Load reference and alignment images."""
        fnames = sorted(
            glob.glob(os.path.join(PreprocessingAlignment().alignment_dir, "*.tif"))
        )
        channel_reference = PreprocessingAlignment().channel_reference + 1
        channel_transform = PreprocessingAlignment().channel_alignment + 1
        self.fnames_reference = [i for i in fnames if f"w{channel_reference}conf" in i]
        self.fnames_transform = [i for i in fnames if f"w{channel_transform}conf" in i]

    def register_alignment_pystackreg(self) -> None:
        """Calculate and register an affine transformation matrix with pystackreg."""
        self.image_reference = np.max(
            [skimage.io.imread(f) for f in self.fnames_reference], axis=0
        )
        self.image_transform = np.max(
            [skimage.io.imread(f) for f in self.fnames_transform], axis=0
        )
        self.sr.register(self.image_reference, self.image_transform)

    def register_alignment_deepblink(self) -> None:
        """Calculate and register an affine transformation matrix with deepBlink."""
        model = pink.io.load_model(PreprocessingAlignment().model)

        # Get coordinates of reference and transform beads
        reference = []
        transform = []
        for fname_reference, fname_transform in zip(
            self.fnames_reference, self.fnames_transform
        ):
            self.image_reference = skimage.io.imread(fname_reference)
            self.image_transform = skimage.io.imread(fname_transform)
            raw_reference = pink.inference.predict(self.image_reference, model)
            raw_transform = pink.inference.predict(self.image_transform, model)
            coords_reference, coords_transform = register_coordinates(
                raw_reference, raw_transform
            )
            reference.extend(coords_reference)
            transform.extend(coords_transform)

        # Create 3x3 affine transformation matrix
        matrix = compute_affine_transform(reference, transform)
        self.sr.set_matrix(matrix)

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
            f"Unknown file extension: {General().file_ext}. Use nd or czi."
        )

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
            self.logger.info(f"Registering image {self.FileID}.")
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
