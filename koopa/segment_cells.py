"""Segment cells primary and secondary."""

import logging
import os


with open(os.devnull, "w") as devnull:
    from cellpose import models
import luigi
import numpy as np
import skimage.filters
import skimage.io
import skimage.morphology
import skimage.segmentation
import tifffile
import torch

from .config import General
from .config import SegmentationPrimary
from .config import SegmentationSecondary
from .preprocess import Preprocess


class SegmentPrimary(luigi.Task):
    """Task to segment cells in 2D/2D+T/3D."""

    FileID = luigi.Parameter()
    logger = logging.getLogger("luigi-interface")

    def requires(self):
        return Preprocess(FileID=self.FileID)

    def output(self):
        return luigi.LocalTarget(
            os.path.join(
                General().analysis_dir, "segmentation_primary", f"{self.FileID}.tif"
            )
        )

    def run(self):
        image = tifffile.imread(self.requires().output().path)
        image_primary = image[SegmentationPrimary().channel]
        segmap = self.segment(image_primary)
        skimage.io.imsave(self.output().path, segmap, check_contrast=False)

    def segment(self, image: np.ndarray) -> np.ndarray:
        """Segment a file using cellpose into nuclear maps."""
        torch.set_num_threads(4)
        cellpose_model = models.Cellpose(
            model_type=SegmentationPrimary().model, gpu=False
        )
        self.logger.info(f"Loaded segmentation model {SegmentationPrimary().model}.")

        # Squeeze image along time dimension
        if not General().do_3D and General().do_TimeSeries:
            image = np.mean(image, axis=0).astype(np.uint16)
            self.logger.info("Squeezed image along time dimension.")

        if General().do_3D:
            image = np.array([(i - np.mean(i)) / np.std(i) for i in image])
            self.logger.info("Normalized image along z-dimension.")

        segmap, *_ = cellpose_model.eval(
            [image],
            diameter=SegmentationPrimary().diameter,
            resample=SegmentationPrimary().resample,
            do_3D=General().do_3D,
            channels=[0, 0],
            min_size=SegmentationPrimary().min_size,
        )
        segmap = segmap[0]
        return segmap


class SegmentSecondary(luigi.Task):
    """Task to segment cells in 2D/2D+T/3D."""

    FileID = luigi.Parameter()
    logger = logging.getLogger("luigi-interface")

    def requires(self):
        return [Preprocess(FileID=self.FileID), SegmentPrimary(FileID=self.FileID)]

    def output(self):
        return luigi.LocalTarget(
            os.path.join(
                General().analysis_dir, "segmentation_secondary", f"{self.FileID}.tif"
            )
        )

    def run(self):
        image = tifffile.imread(self.requires()[0].output().path)
        image_secondary = image[SegmentationSecondary().channel]
        self.logger.debug("Loaded secondary image.")
        segmap_primary = tifffile.imread(self.requires()[1].output().path)
        self.logger.debug("Loaded primary segmap.")

        segmap = self.segment_secondary(segmap_primary, image_secondary)
        skimage.io.imsave(self.output().path, segmap, check_contrast=False)

    @staticmethod
    def foreground_segmentation(image: np.ndarray) -> np.ndarray:
        """Segmentation of the channel of interest."""
        method = SegmentationSecondary().method
        image = np.clip(
            image, 0, np.quantile(image, SegmentationSecondary().upper_clip)
        )
        image = skimage.filters.gaussian(image, SegmentationSecondary().gaussian)
        if method == "otsu":
            return image > skimage.filters.threshold_otsu(image)
        if method == "li":
            return image > skimage.filters.threshold_li(image)
        if method == "median":
            return image > (np.median(image) * SegmentationSecondary().value)
        raise ValueError(f"Unknown secondary segmentation method {method}.")

    def segment_secondary(
        self, primary: np.ndarray, secondary: np.ndarray
    ) -> np.ndarray:
        """Segmentation removing previously completed primary segmentation."""
        foreground = self.foreground_segmentation(secondary)
        foreground = skimage.morphology.remove_small_objects(foreground)

        # 1. nucleus, 2. cytoplasm
        if SegmentationPrimary().model == "nuclei":
            self.logger.debug("Segmenting cytoplasm from nuclei.")
            segmap = skimage.segmentation.watershed(
                ~secondary, primary, mask=foreground, watershed_line=True
            )

        # 1. cytoplasm, 2. nucleus
        if SegmentationPrimary().model == "cyto":
            self.logger.debug("Segmenting nuclei from cytoplasm.")
            segmap = np.where(foreground, primary, 0)

        return segmap
