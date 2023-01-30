"""Segment cells into nucleus and/or cytoplasm."""

from typing import List, Tuple
import os

with open(os.devnull, "w") as devnull:
    from cellpose import models
import numpy as np
import scipy.ndimage as ndi
import skimage.filters
import skimage.io
import skimage.morphology
import skimage.segmentation


def preprocess(image: np.ndarray) -> np.ndarray:
    return np.mean(image, axis=0).astype(np.uint16)


def relabel_array(image: np.ndarray, mapping: dict) -> np.ndarray:
    """Label an image array based on a input->output map."""
    new_image = [np.where(image == key, value, 0) for key, value in mapping.items()]
    return np.max(new_image, axis=0)


def segment_otsu(
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


def segment_cellpose(
    image: np.ndarray,
    model: str,
    pretrained: List[str],
    do_3d: bool,
    diameter: int,
    resample: bool,
    min_size_nuclei: int,
    gpu: bool = False,
) -> np.ndarray:
    """Segment a file using cellpose into nuclear maps."""
    cellpose_model = models.CellposeModel(
        model_type=model, gpu=gpu, pretrained_model=pretrained
    )

    if do_3d:
        image = np.array([(i - np.mean(i)) / np.std(i) for i in image])

    segmap, *_ = cellpose_model.eval(
        [image],
        channels=[0, 0],
        diameter=diameter,
        do_3D=do_3d,
        min_size=min_size_nuclei,
        resample=resample,
    )
    segmap = segmap[0]
    return segmap


def remove_border_objects(image: np.ndarray) -> np.ndarray:
    """Remove objects touching the border of the image."""
    shape = {0, *image.shape}
    for prop in skimage.measure.regionprops(image):
        if bool(shape & {*prop.bbox}):
            image = np.where(image == prop.label, 0, image)
    return image


def segment_background(
    image: np.ndarray, method: str, upper_clip: int, gaussian: int, min_size: int
) -> np.ndarray:
    """Segmentation of the channel of interest."""
    image = np.clip(image, 0, np.quantile(image, upper_clip))
    image = skimage.filters.gaussian(image, gaussian)

    methods = ["otsu", "li", "triangle"]
    if method not in methods:
        raise ValueError(
            f"Unknown secondary segmentation method {method}. "
            f"Please provide one of - {methods}"
        )

    if method == "otsu":
        image = image > skimage.filters.threshold_otsu(image)
    if method == "li":
        image = image > skimage.filters.threshold_li(image)
    if method == "triangle":
        image = image > skimage.filters.threshold_triangle(image)

    image = skimage.morphology.remove_small_objects(image, min_size=min_size)
    image = skimage.morphology.remove_small_holes(image, area_threshold=min_size)
    return image


def segment_nuclei(image: np.ndarray, config: dict) -> np.ndarray:
    method = config["method_nuclei"]
    if method == "cellpose":
        return segment_cellpose(
            image,
            model="nuclei",
            pretrained=config["cellpose_models"],
            do_3d=config["do_3d"],
            diameter=config["cellpose_diameter"],
            resample=config["cellpose_resample"],
            min_size_nuclei=config["min_size_nuclei"],
            gpu=config["gpu"],
        )
    if method == "otsu":
        return segment_otsu(
            image,
            gaussian=config["gaussian"],
            min_size_nuclei=config["min_size_nuclei"],
            min_distance=config["min_distance"],
        )
    raise ValueError(f"Unknown nuclei segmentation method {method}.")


def segment_cyto(image: np.ndarray, config: dict) -> np.ndarray:
    return segment_cellpose(
        image,
        model="cyto",
        pretrained=config["cellpose_models"],
        do_3d=config["do_3d"],
        diameter=config["cellpose_diameter"],
        resample=config["cellpose_resample"],
        min_size_nuclei=config["min_size_nuclei"],
        gpu=config["gpu"],
    )


def segment_both(
    image_nuclei: np.ndarray, image_cyto: np.ndarray, config: dict
) -> Tuple[np.ndarray]:
    segmap_nuclei = segment_nuclei(image_nuclei, config)

    segmap_cyto = segment_background(
        image_cyto,
        method=config["method_cyto"],
        upper_clip=config["upper_clip"],
        gaussian=config["gaussian"],
        min_size=config["min_size_cyto"],
    )
    segmap_cyto = skimage.segmentation.watershed(
        image=~image_cyto,
        markers=segmap_nuclei,
        mask=segmap_cyto,
        watershed_line=True,
    )

    # Remove objects and keep nuclei/cyto pairing in order
    if config["remove_border"]:
        segmap_cyto = remove_border_objects(segmap_cyto)
        nuclei_to_keep = set(np.unique(segmap_nuclei)).intersection(
            set(np.unique(segmap_cyto))
        )
        mapping = {nuc: idx for idx, nuc in enumerate(nuclei_to_keep)}
        segmap_nuclei = relabel_array(segmap_nuclei, mapping)
        segmap_cyto = relabel_array(segmap_cyto, mapping)

    return segmap_nuclei, segmap_cyto
