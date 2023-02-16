"""Segment cells into nucleus and/or cytoplasm using mathematical operators."""

from typing import Tuple, Literal

import numpy as np
import scipy.ndimage as ndi
import skimage.filters
import skimage.io
import skimage.morphology
import skimage.segmentation


def relabel_array(image: np.ndarray, mapping: dict) -> np.ndarray:
    """Label an image array based on a input->output map."""
    new_image = [np.where(image == key, value, 0) for key, value in mapping.items()]
    return np.max(new_image, axis=0)


def segment_nuclei(
    image: np.ndarray, gaussian: int, min_size_nuclei: int, min_distance: int
) -> np.ndarray:
    """Segment a file using mathematical filters into nuclear maps."""
    # Intial binary threshold
    image = skimage.filters.gaussian(image, sigma=gaussian)
    image = image > skimage.filters.threshold_otsu(image)
    image = ndi.binary_fill_holes(image)
    image = skimage.morphology.remove_small_objects(image, min_size=min_size_nuclei)
    segmap = skimage.measure.label(image)

    # Separation of merged objects
    distance = ndi.distance_transform_edt(segmap)
    coords = skimage.feature.peak_local_max(
        distance, labels=segmap, min_distance=min_distance
    )
    mask = np.zeros(distance.shape, dtype=bool)
    mask[tuple(coords.T)] = True
    markers, _ = ndi.label(mask)
    segmap = skimage.segmentation.watershed(-distance, markers, mask=segmap)
    return segmap


def remove_border_objects(image: np.ndarray) -> np.ndarray:
    """Remove objects touching the border of the image."""
    shape = {0, *image.shape}
    for prop in skimage.measure.regionprops(image):
        if bool(shape & {*prop.bbox}):
            image = np.where(image == prop.label, 0, image)
    return image


def segment_background(
    image: np.ndarray,
    method: Literal["otsu", "li", "triangle"],
    upper_clip: float,
    gaussian: int,
    min_size: int,
) -> np.ndarray:
    """Thresholding of the background for cytoplasmic segmentation."""
    methods = ["otsu", "li", "triangle"]
    if method not in methods:
        raise ValueError(
            f"Unknown secondary segmentation method {method}. "
            f"Please provide one of - {methods}"
        )

    image = np.clip(image, 0, np.quantile(image, upper_clip))
    image = skimage.filters.gaussian(image, gaussian)

    if method == "otsu":
        image = image > skimage.filters.threshold_otsu(image)
    if method == "li":
        image = image > skimage.filters.threshold_li(image)
    if method == "triangle":
        image = image > skimage.filters.threshold_triangle(image)

    image = skimage.morphology.remove_small_objects(image, min_size=min_size)
    image = skimage.morphology.remove_small_holes(image, area_threshold=min_size)
    return image


def segment_both(
    image_nuclei: np.ndarray,
    image_cyto: np.ndarray,
    method_cyto: Literal["otsu", "li", "triangle"] = "triangle",
    upper_clip: float = 0.95,
    gaussian_nuclei: int = 3,
    gaussian_cyto: int = 3,
    min_size_nuclei: int = 1000,
    min_distance: int = 50,
    min_size_cyto: int = 5000,
    remove_border: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """Joint thresholded segmentation of nuclei and cytoplasm."""
    segmap_nuclei = segment_nuclei(
        image_nuclei, gaussian_nuclei, min_size_nuclei, min_distance
    )

    segmap_cyto = segment_background(
        image_cyto,
        method=method_cyto,
        upper_clip=upper_clip,
        gaussian=gaussian_cyto,
        min_size=min_size_cyto,
    )
    segmap_cyto = skimage.segmentation.watershed(
        image=~image_cyto,
        markers=segmap_nuclei,
        mask=segmap_cyto,
        watershed_line=True,
    )

    # Remove objects and keep nuclei/cyto pairing in order
    if remove_border:
        segmap_cyto = remove_border_objects(segmap_cyto)
        nuclei_to_keep = set(np.unique(segmap_nuclei)).intersection(
            set(np.unique(segmap_cyto))
        )
        mapping = {nuc: idx for idx, nuc in enumerate(nuclei_to_keep)}
        segmap_nuclei = relabel_array(segmap_nuclei, mapping)
        segmap_cyto = relabel_array(segmap_cyto, mapping)

    return segmap_nuclei, segmap_cyto
