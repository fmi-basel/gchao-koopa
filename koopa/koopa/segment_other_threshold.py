"""Segment other features."""

import os

import numpy as np

import skimage.exposure
import skimage.filters
import skimage.morphology


def segment(image: np.ndarray, channel: int, method: Literal["otsu", "li",
                                                     "multiotsu"]) -> \
        np.ndarray:
    """Segment marker labels with pre-trained model or traditionally."""
    masks = []
    for frame in image:
        if method == "otsu":
            mask = image > skimage.filters.threshold_otsu(frame)
        if method == "li":
            mask = image > skimage.filters.threshold_li(frame)
        if method == "multiotsu":
            # Default 3 classes -> 2nd highest class chosen
            mask = image > skimage.filters.threshold_multiotsu(frame)[1]
        masks.append(mask)
    return np.array(masks).squeeze()
