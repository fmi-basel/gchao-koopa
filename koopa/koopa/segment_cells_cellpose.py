"""Segment cells into nucleus and/or cytoplasm."""

from typing import Literal
import os

with open(os.devnull, "w") as devnull:
    from cellpose import models
import numpy as np


def preprocess(image: np.ndarray) -> np.ndarray:
    """Average an image along t to avoid merging/tracking segmented cells."""
    return np.mean(image, axis=0).astype(np.uint16)


def segment_cellpose(
    image: np.ndarray,
    model: Literal["nuclei", "cyto"],
    do_3d: bool = False,
    cellpose_diameter: int = 150,
    cellpose_resample: bool = True,
    cellpose_min_size: int = 1000,
    gpu: bool = False,
) -> np.ndarray:
    """Segment an image using cellpose."""
    # Normalize along Z
    if do_3d:
        image = np.array([(i - np.mean(i)) / np.std(i) for i in image])

    cellpose_model = models.CellposeModel(model_type=model, gpu=gpu)
    segmap, *_ = cellpose_model.eval(
        [image],
        channels=[0, 0],
        diameter=cellpose_diameter,
        do_3D=do_3d,
        min_size=cellpose_min_size,
        resample=cellpose_resample,
    )
    segmap = segmap[0]
    return segmap


def segment_nuclei(image: np.ndarray, *args, **kwargs) -> np.ndarray:
    return segment_cellpose(image, *args, **kwargs, model="nuclei")


def segment_cyto(image: np.ndarray, *args, **kwargs) -> np.ndarray:
    return segment_cellpose(image, *args, **kwargs, model="cyto")
