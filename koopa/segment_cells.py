"""Segment cells into nucleus and/or cytoplasm."""

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

from .config import General
from .config import SegmentationCells
from .preprocess import Preprocess


class SegmentCells(luigi.Task):
    """Task to segment cells in 2D/2D+T/3D."""

    FileID = luigi.Parameter()
    logger = logging.getLogger("koopa")
    gpu = General().gpu_index != -1

    def requires(self):
        return Preprocess(FileID=self.FileID)

    def output(self):
        outputs = {}
        if SegmentationCells().selection in ("nuclei", "both"):
            outputs["nuclei"] = luigi.LocalTarget(
                os.path.join(
                    General().analysis_dir,
                    "segmentation_nuclei",
                    f"{self.FileID}.tif",
                )
            )

        if SegmentationCells().selection in ("cyto", "both"):
            outputs["cyto"] = luigi.LocalTarget(
                os.path.join(
                    General().analysis_dir,
                    "segmentation_cyto",
                    f"{self.FileID}.tif",
                )
            )
        return outputs

    def run(self):
        image = tifffile.imread(self.requires().output().path)

        image_nuclei = image[SegmentationCells().channel_nuclei]
        image_cyto = image[SegmentationCells().channel_cyto]
        selection = SegmentationCells().selection

        # TODO add support for cell tracking over time?
        if not General().do_3D and General().do_TimeSeries:
            image_nuclei = np.mean(image_nuclei, axis=0).astype(np.uint16)
            image_cyto = np.mean(image_cyto, axis=0).astype(np.uint16)
            self.logger.info("Squeezed images along time dimension.")

        # Single output
        if selection == "nuclei":
            segmap = self.segment_nuclei(image_nuclei)
        if selection == "cyto":
            segmap = self.segment_cyto(image_cyto)
        if selection in ("nuclei", "cyto"):
            skimage.io.imsave(
                self.output()[selection].path, segmap, check_contrast=False
            )

        # Dual output
        if selection == "both":
            segmap_nuclei, segmap_cyto = self.segment_both(image_nuclei, image_cyto)
            skimage.io.imsave(
                self.output()["nuclei"].path, segmap_nuclei, check_contrast=False
            )
            skimage.io.imsave(
                self.output()["cyto"].path, segmap_cyto, check_contrast=False
            )

    def segment_nuclei(self, image: np.ndarray) -> np.ndarray:
        method = SegmentationCells().method_nuclei
        if method == "cellpose":
            return self.segment_cellpose(image, "nuclei")
        if method == "otsu":
            return self.segment_otsu(image)
        raise ValueError(f"Unknown nuclei segmentation method {method}.")

    def segment_cyto(self, image: np.ndarray) -> np.ndarray:
        return self.segment_cellpose(image, "cyto")

    def segment_both(
        self, image_nuclei: np.ndarray, image_cyto: np.ndarray
    ) -> np.ndarray:
        segmap_nuclei = self.segment_nuclei(image_nuclei)

        segmap_cyto = self.segment_background(image_cyto)
        self.logger.debug("Segmenting cytoplasm from nuclei.")
        segmap_cyto = skimage.segmentation.watershed(
            image=~segmap_cyto,
            markers=segmap_nuclei,
            mask=segmap_cyto,
            watershed_line=True,
        )
        return segmap_nuclei, segmap_cyto

    def segment_otsu(self, image: np.ndarray) -> np.ndarray:
        """Segment a file using mathematical filters into nuclear maps."""
        self.logger.info("Started segmenting nuclei with otsu.")
        # Intial binary threshold
        image = skimage.filters.gaussian(image, sigma=SegmentationCells().gaussian)
        image = image > skimage.filters.threshold_otsu(image)
        image = ndi.binary_fill_holes(image)
        image = skimage.morphology.remove_small_objects(
            image, min_size=SegmentationCells().min_size_nuclei
        )
        segmap = skimage.measure.label(image)

        # Separation of merged objects
        distance = ndi.distance_transform_edt(segmap)
        coords = skimage.feature.peak_local_max(
            distance, labels=segmap, min_distance=SegmentationCells().min_distance
        )
        mask = np.zeros(distance.shape, dtype=bool)
        mask[tuple(coords.T)] = True
        markers, _ = ndi.label(mask)
        segmap = skimage.segmentation.watershed(-distance, markers, mask=segmap)
        return segmap

    def segment_cellpose(self, image: np.ndarray, model: str) -> np.ndarray:
        """Segment a file using cellpose into nuclear maps."""
        cellpose_model = models.Cellpose(model_type=model, gpu=self.gpu)
        self.logger.info(f"Loaded cellpose segmentation model {model}.")

        if General().do_3D:
            image = np.array([(i - np.mean(i)) / np.std(i) for i in image])
            self.logger.info("Normalized image along z-dimension.")

        segmap, *_ = cellpose_model.eval(
            [image],
            diameter=SegmentationCells().cellpose_diameter,
            resample=SegmentationCells().cellpose_resample,
            do_3D=General().do_3D,
            channels=[0, 0],
            min_size=SegmentationCells().min_size_nuclei,
        )
        segmap = segmap[0]
        return segmap

    def remove_border_objects(self, image: np.ndarray) -> np.ndarray:
        """Remove objects touching the border of the image."""
        self.logger.info("Removing border objects.")
        for idx, prop in enumerate(skimage.measure.regionprops(image)):
            if bool({0, *image.shape} & {*prop.bbox}):
                image = np.where(image == idx, 0, image)
        return skimage.measure.label(image)

    def segment_background(self, image: np.ndarray) -> np.ndarray:
        """Segmentation of the channel of interest."""
        method = SegmentationCells().method_cyto
        self.logger.info(f"Segmenting background with {method}.")
        image = np.clip(image, 0, np.quantile(image, SegmentationCells().upper_clip))
        image = skimage.filters.gaussian(image, SegmentationCells().gaussian)

        if method not in ["otsu", "li", "triangle"]:
            raise ValueError(f"Unknown secondary segmentation method {method}.")

        if method == "otsu":
            image = image > skimage.filters.threshold_otsu(image)
        if method == "li":
            image = image > skimage.filters.threshold_li(image)
        if method == "triangle":
            image = image > skimage.filters.threshold_triangle(image)

        return image
