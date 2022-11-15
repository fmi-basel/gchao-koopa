"""Image preprocessing (raw data to pipeline input)."""

from typing import Literal, List

import numpy as np
import skimage.io
import skimage.transform


def get_sharpest_slice(image: np.ndarray, axis: int = 0) -> np.ndarray:
    """Returns index of the sharpest slice in an image array.

    Uses the average gradient magnitude.
    Adapted from https://stackoverflow.com/questions/6646371/detect-which-image-is-sharper
    """
    sharpness = []
    for array in np.swapaxes(image, 0, axis):
        y, x = np.gradient(array)
        norm = np.sqrt(x ** 2 + y ** 2)
        sharpness.append(np.average(norm))
    sharpest = sharpness.index(max(sharpness))
    return image[sharpest]


def register_3d_image(
    image: np.ndarray, method: Literal["maximum", "mean", "sharpest"]
) -> np.ndarray:
    """Intensity projection to convert 3D+C image to 2D."""
    if image.ndim != 4:
        raise ValueError("Image must be 4D.")

    z_axis = image.shape.index(sorted(image.shape)[1])

    if method == "maximum":
        return image.max(axis=z_axis)
    if method == "mean":
        return image.mean(axis=z_axis)
    if method == "sharpest":
        c_axis = image.shape.index(sorted(image.shape)[0])
        channels = []
        for channel in np.moveaxis(image, c_axis, 0):
            z_axis = channel.shape.index(sorted(channel.shape)[0])
            channels.append(get_sharpest_slice(channel, z_axis))
        return np.stack(channels)
    raise ValueError(f"Unknown 3D registration method: {method}")


def crop_image(image: np.ndarray, crop_start: int, crop_end: str) -> np.ndarray:
    """Crop image to only observe central region."""
    if image.ndim != 4:
        raise ValueError("Image must be 4D.")
    return image[..., crop_start:crop_end, crop_start:crop_end]


def bin_image(image: np.ndarray, bin_axes: List[float]) -> np.ndarray:
    """Bin image along the x and y axes."""
    if image.ndim != len(bin_axes):
        raise ValueError("bin_axes must have same shape as image.")
    return skimage.transform.rescale(image, scale=bin_axes)


def trim_image(image: np.ndarray, frame_start: int, frame_end: int) -> np.ndarray:
    """Remove first and last frames from image."""
    if image.ndim != 4:
        raise ValueError("Image must be 4D.")

    if frame_start > 0:
        image = image[:, frame_start:]

    new_frame_end = frame_end - frame_start
    if new_frame_end > 0:
        image = image[:, :new_frame_end]
    return image
