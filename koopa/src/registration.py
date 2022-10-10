"""Registration for camera or chromatic aberation alignment."""

from typing import Iterable
import glob
import logging
import os

import deepblink as pink
import luigi
import numpy as np
import pystackreg
import scipy.optimize
import scipy.spatial
import skimage.io

from .config import General
from .config import PreprocessingAlignment


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
    """Task to create affine matrix for two reference channels."""

    logger = logging.getLogger("koopa")

    def output(self):
        return [
            luigi.LocalTarget(os.path.join(General().analysis_dir, "alignment.npy")),
            luigi.LocalTarget(
                os.path.join(General().analysis_dir, "alignment_pre.tif"),
            ),
            luigi.LocalTarget(
                os.path.join(General().analysis_dir, "alignment_post.tif"),
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
        pre_alignment = np.stack([self.image_reference, self.image_transform])
        skimage.io.imsave(self.output()[1].path, pre_alignment, check_contrast=False)

        transform = self.sr.transform(self.image_transform)
        post_alignment = np.stack([self.image_reference, transform])
        skimage.io.imsave(self.output()[2].path, post_alignment, check_contrast=False)

        # _, ax = plt.subplots(1, 2, figsize=(20, 20))
        # ax[0].set_title("Pre-alignment")
        # ax[0].imshow(self.image_reference, cmap="Greens")
        # ax[0].imshow(self.image_transform, cmap="Reds", alpha=0.5)
        # ax[1].set_title("Post-alignment")
        # ax[1].imshow(self.image_reference, cmap="Greens")
        # ax[1].imshow(transform, cmap="Reds", alpha=0.5)
        # plt.savefig(self.output()[1].path, bbox_inches="tight")
        # plt.close()
