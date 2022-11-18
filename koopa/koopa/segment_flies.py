import gc

from cellpose import models
import cellpose
import numpy as np
import pandas as pd
import scipy.ndimage as ndi
import skimage
import skimage.feature
import skimage.filters
import skimage.segmentation
import torch


def normalize_nucleus(image: np.ndarray) -> np.ndarray:
    """Z-normalize nuclear images."""
    return np.array([(i - np.mean(i)) / np.std(i) for i in image])


def cellpose_predict(image: np.ndarray, batch_size: int = 4) -> np.ndarray:
    """Segment nuclei with cellpose."""
    model = models.CellposeModel(model_type="nuclei", gpu=True)
    model.batch_size = batch_size

    image = cellpose.transforms.convert_image(
        image, [[0, 0]], do_3D=True, normalize=False
    )

    if image.ndim < 4:
        image = image[np.newaxis, ...]

    gc.collect()
    torch.cuda.empty_cache()

    img = np.asarray(image)
    img = cellpose.transforms.normalize_img(img, invert=False)
    yf, _ = model._run_3D(img)  # (Y-flow, X-flow, cell probability), style
    return yf


def merge_masks(yf: np.ndarray) -> np.ndarray:
    """Merge flow masks to nuclei with cellpose."""
    dist = yf[0][-1] + yf[1][-1] + yf[2][-1]
    dP = np.stack(
        (yf[1][0] + yf[2][0], yf[0][0] + yf[2][1], yf[0][1] + yf[1][1]), axis=0
    )  # (dZ, dY, dX)
    masks, *_ = cellpose.dynamics.compute_masks(dP, dist, do_3D=True)
    return masks.squeeze()


def remove_false_objects(
    image: np.ndarray,
    segmap: np.ndarray,
    min_intensity: int,
    min_area: int,
    max_area: int,
) -> np.ndarray:
    """Remove over/under-segmented objects from segmentation.

    Args:
        image: Input image used for intensity thresholding.
        segmap: Segmentation map.
    """
    # Get properties of each segment
    df = pd.DataFrame(
        skimage.measure.regionprops_table(
            segmap, image, properties=("label", "mean_intensity", "area")
        )
    ).set_index("label", drop=True)
    labels = df[
        (df["mean_intensity"] > min_intensity)
        & (df["area"] > min_area)
        & (df["area"] < max_area)
    ]
    # self.logger.info(f"Keeping {len(labels):} nuclei from {len(np.unique(segmap)):}")

    # Filter objects
    for label in df.index[~df.index.isin(labels.index)]:
        segmap[np.where(segmap == label)] = 0
    return segmap


def dilate_segmap(segmap: np.ndarray, dilation: int) -> np.ndarray:
    """Segment nuclei with cellpose."""
    structure = skimage.morphology.ball(4)
    mask = ndi.binary_dilation(segmap, structure, iterations=dilation)
    mask = ndi.binary_fill_holes(mask)
    dilated = skimage.segmentation.watershed(mask, markers=segmap, mask=mask)
    return dilated
