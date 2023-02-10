"""Registration for camera or chromatic aberation alignment."""

from typing import Iterable, List
import os
import glob

import deepblink as pink
import numpy as np
import pystackreg
import scipy.optimize
import scipy.spatial
import skimage.io

from . import io


def align_image(
    image: np.ndarray, sr: pystackreg.StackReg, index_transforms: List[int]
) -> np.ndarray:
    """Align a stack of images."""
    for idx in index_transforms:
        if image.ndim == 3:
            image[idx] = sr.transform(image[idx])

        if image.ndim == 4:
            image[idx] = np.array([sr.transform(i) for i in image[idx]])
    return image


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


def load_alignment_images(
    path: str, channel_reference: int, channel_transform: int
) -> Iterable[List[np.ndarray]]:
    """Load reference and alignment images."""
    fnames = sorted(glob.glob(os.path.join(path, "*.tif")))

    channel_reference = channel_reference + 1
    channel_transform = channel_transform + 1
    fnames_reference = [i for i in fnames if f"w{channel_reference}conf" in i]
    fnames_transform = [i for i in fnames if f"w{channel_transform}conf" in i]
    images_reference = [skimage.io.imread(f) for f in fnames_reference]
    images_transform = [skimage.io.imread(f) for f in fnames_transform]
    return images_reference, images_transform


def register_alignment_pystackreg(
    images_reference: List[np.ndarray], images_transform: List[np.ndarray]
) -> np.ndarray:
    """Calculate and register an affine transformation matrix with pystackreg."""
    image_reference = np.max(images_reference, axis=0)
    image_transform = np.max(images_transform, axis=0)
    sr = pystackreg.StackReg(pystackreg.StackReg.AFFINE)
    sr.register(image_reference, image_transform)
    return sr.get_matrix()


def register_alignment_deepblink(
    fname_model: str,
    images_reference: List[np.ndarray],
    images_transform: List[np.ndarray],
) -> np.ndarray:
    """Calculate and register an affine transformation matrix with deepBlink."""
    model = pink.io.load_model(fname_model)

    # Get coordinates of reference and transform beads
    reference = []
    transform = []
    for image_reference, image_transform in zip(images_reference, images_transform):
        raw_reference = pink.inference.predict(image_reference, model)
        raw_transform = pink.inference.predict(image_transform, model)
        coords_reference, coords_transform = register_coordinates(
            raw_reference, raw_transform
        )
        reference.extend(coords_reference)
        transform.extend(coords_transform)

    # Create 3x3 affine transformation matrix
    matrix = compute_affine_transform(reference, transform)
    return matrix


def get_stackreg(matrix: np.ndarray) -> pystackreg.StackReg:
    """Placeholder to prevent deepblink requirement in io."""
    return io.get_stackreg(matrix)


def visualize_alignment(
    sr: pystackreg.StackReg,
    image_reference: np.ndarray,
    image_transform: np.ndarray,
    fname_pre: os.PathLike,
    fname_post: os.PathLike,
) -> None:
    """Plot chromatic transform before and after alignment."""
    pre_alignment = np.stack([image_reference, image_transform])
    skimage.io.imsave(fname_pre, pre_alignment, check_contrast=False)

    transform = sr.transform(image_transform)
    post_alignment = np.stack([image_reference, transform])
    skimage.io.imsave(fname_post, post_alignment, check_contrast=False)
