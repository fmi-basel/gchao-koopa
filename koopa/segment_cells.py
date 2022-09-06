"""Segment cells primary and secondary."""

import logging
import os


with open(os.devnull, "w") as devnull:
    from cellpose import models
import luigi
import numpy as np
import scipy.ndimage as ndi
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
        if SegmentationPrimary().remove_border:
            segmap = self.remove_border_objects(segmap)
        skimage.io.imsave(self.output().path, segmap, check_contrast=False)

    def segment(self, image: np.ndarray) -> np.ndarray:
        """Runner for segmentation options."""
        method = SegmentationPrimary().method
        if method not in ["otsu", "cellpose"]:
            raise ValueError(f"Unknown secondary segmentation method {method}.")
        if method == "otsu" and SegmentationPrimary().model != "nuclei":
            raise ValueError("If method otsu is selected, model must be set to nuclei!")

        if method == "otsu":
            return self.segment_otsu(image)
        if method == "cellpose":
            return self.segment_cellpose(image)

    def remove_border_objects(self, image: np.ndarray) -> np.ndarray:
        """Remove objects touching the border of the image."""
        self.logger.info("Removing border objects.")
        for idx, prop in enumerate(skimage.measure.regionprops(image)):
            if bool({0, *image.shape} & {*prop.bbox}):
                labels = np.where(image == idx, 0, image)
        return skimage.measure.label(labels)

    def segment_otsu(self, image: np.ndarray) -> np.ndarray:
        """Segment a file using mathematical filters into nuclear maps."""
        self.logger.info("Started segmenting nuclei with otsu.")
        # Intial binary threshold
        image = skimage.filters.gaussian(image, sigma=SegmentationPrimary().gaussian)
        image = image > skimage.filters.threshold_otsu(image)
        image = ndi.binary_fill_holes(image)
        image = skimage.morphology.remove_small_objects(
            image, min_size=SegmentationPrimary().min_size
        )
        segmap = skimage.measure.label(image)

        # Separation of merged objects
        distance = ndi.distance_transform_edt(segmap)
        coords = skimage.feature.peak_local_max(
            distance, labels=segmap, min_distance=SegmentationPrimary().min_distance
        )
        mask = np.zeros(distance.shape, dtype=bool)
        mask[tuple(coords.T)] = True
        markers, _ = ndi.label(mask)
        segmap = skimage.segmentation.watershed(-distance, markers, mask=segmap)
        return segmap

    def segment_cellpose(self, image: np.ndarray) -> np.ndarray:
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

        if method not in ["otsu", "li", "triangle", "median"]:
            raise ValueError(f"Unknown secondary segmentation method {method}.")

        if method == "otsu":
            image = image > skimage.filters.threshold_otsu(image)
        if method == "li":
            image = image > skimage.filters.threshold_li(image)
        if method == "triangle":
            image = image > skimage.filters.threshold_triangle(image)
        if method == "median":
            image = image > (np.median(image) * SegmentationSecondary().value)
        image = skimage.morphology.remove_small_objects(
            image, min_size=SegmentationSecondary().min_size
        )
        return image

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
