import gc
import logging
import os

from cellpose import models
import cellpose
import luigi
import numpy as np
import pandas as pd
import scipy.ndimage as ndi
import skimage
import skimage.feature
import skimage.filters
import skimage.segmentation
import torch

from .config import FlyBrainCells
from .config import General
from preprocess import Preprocess


class SegmentCellsPredict(luigi.Task):
    """Task to segment cells in 3D."""

    FileID = luigi.Parameter()
    logger = logging.getLogger("koopa")

    def requires(self):
        return Preprocess(FileID=self.FileID)

    def output(self):
        return luigi.LocalTarget(
            os.path.join(
                General().analysis_dir,
                "segmentation_nuclei_prediction",
                f"{self.FileID}.tif",
            )
        )

    def run(self):
        image = skimage.io.imread(self.requires().output().path)
        image_nucleus = self.normalize_nucleus(image[FlyBrainCells().channel])

        segmap = self.cellpose_predict(image_nucleus)
        skimage.io.imsave(self.output().path, segmap, check_contrast=False)

    @staticmethod
    def normalize_nucleus(image: np.ndarray) -> np.ndarray:
        """Z-normalize nuclear images."""
        return np.array([(i - np.mean(i)) / np.std(i) for i in image])

    @staticmethod
    def cellpose_predict(image):
        """Segment nuclei with cellpose."""
        model = models.CellposeModel(model_type="nuclei", gpu=True)
        model.batch_size = FlyBrainCells().batch_size

        image = cellpose.transforms.convert_image(
            image, [[0, 0]], do_3D=True, normalize=False
        )

        if image.ndim < 4:
            image = image[np.newaxis, ...]

        gc.collect()
        torch.cuda.empty_cache()

        img = np.asarray(image)
        img = cellpose.transforms.normalize_img(img, invert=False)
        yf, _ = model._run_3D(img)
        return yf


class SegmentCellsMerge(luigi.Task):
    """Task to segment cells in 3D."""

    FileID = luigi.Parameter()
    logger = logging.getLogger("koopa")

    def requires(self):
        return [Preprocess(FileID=self.FileID), SegmentCellsPredict(FileID=self.FileID)]

    def output(self):
        return luigi.LocalTarget(
            os.path.join(
                General().analysis_dir,
                "segmentation_nuclei_merge",
                f"{self.FileID}.tif",
            )
        )

    def run(self):
        # Load and normalize
        image = skimage.io.imread(self.requires()[0].output().path)[
            FlyBrainCells().channel
        ]
        yf = skimage.io.imread(self.requires()[1].output().path)
        segmap = self.merge_masks(yf)
        self.logger.info("Merged cellpose masks.")
        segmap = self.remove_false_objects(image, segmap)
        skimage.io.imsave(self.output().path, segmap, check_contrast=False)

    @staticmethod
    def merge_masks(yf: np.ndarray) -> np.ndarray:
        """Merge flow masks to nuclei with cellpose."""
        dist = yf[0][-1] + yf[1][-1] + yf[2][-1]
        dP = np.stack(
            (yf[1][0] + yf[2][0], yf[0][0] + yf[2][1], yf[0][1] + yf[1][1]), axis=0
        )  # (dZ, dY, dX)
        masks, *_ = cellpose.dynamics.compute_masks(dP, dist, do_3D=True)
        return masks.squeeze()

    def remove_false_objects(self, image: np.ndarray, segmap: np.ndarray) -> np.ndarray:
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
            (df["mean_intensity"] > FlyBrainCells().min_intensity)
            & (df["area"] > FlyBrainCells().min_area)
            & (df["area"] < FlyBrainCells().max_area)
        ]
        self.logger.info(
            f"Keeping {len(labels):} nuclei from {len(np.unique(segmap)):}"
        )

        # Filter objects
        for label in df.index[~df.index.isin(labels.index)]:
            segmap[np.where(segmap == label)] = 0
        return segmap


class DilateCells(luigi.Task):
    """Task to dilate cells in 3D (to help spot-to-cell assignment)."""

    FileID = luigi.Parameter()

    def requires(self):
        return SegmentCellsMerge(FileID=self.FileID)

    def output(self):
        return luigi.LocalTarget(
            os.path.join(
                General().analysis_dir, "segmentation_nuclei", f"{self.FileID}.tif"
            )
        )

    def run(self):
        segmap = skimage.io.imread(self.requires().output().path)
        dilated = self.dilate_segmap(segmap)
        skimage.io.imsave(self.output().path, dilated, check_contrast=False)

    @staticmethod
    def dilate_segmap(segmap: np.ndarray) -> np.ndarray:
        """Segment nuclei with cellpose."""
        structure = skimage.morphology.ball(4)
        mask = ndi.binary_dilation(
            segmap, structure, iterations=FlyBrainCells().dilation
        )
        mask = ndi.binary_fill_holes(mask)
        dilated = skimage.segmentation.watershed(mask, markers=segmap, mask=mask)
        return dilated
