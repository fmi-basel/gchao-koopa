import os
import sys

from cellpose import models
import luigi
import numpy as np
import skimage
import skimage.feature
import skimage.filters
import skimage.segmentation
import torch

from config import globalConfig
from preprocess import Preprocess


class SegmentCells(luigi.Task):
    """Task to segment cells in 3D."""

    FileID = luigi.Parameter()

    @property
    def input_file(self):
        return os.path.join(
            globalConfig().AnalysisDir, "preprocessed", f"{self.FileID}.tif"
        )

    @property
    def output_file(self):
        return os.path.join(
            globalConfig().AnalysisDir, "segmentation_cells", f"{self.FileID}.tif"
        )

    def segment_nucleus(self, diameter_nucleus: int, minimum_size: int,) -> np.ndarray:
        """Segment a file using cellpose into nuclear maps.
        
        Args:
            diameter_nucleus: Expected diameter of the nucleus in px to
                be passed to the CellPose model.
            minimum_size: Minimum area in px of the nucleus.
        """
        sys.stdout = open(os.devnull, "w")
        torch.set_num_threads(4)
        cellpose_model = models.Cellpose(model_type="nuclei", gpu=False)
        mask_nucl, *_ = cellpose_model.eval(
            [self.image_nucleus],
            diameter=diameter_nucleus,
            channels=[0, 0],
            min_size=minimum_size,
        )
        mask_nucl = mask_nucl[0]
        sys.stdout = sys.__stdout__
        return mask_nucl

    def segment_cell(self, mask_nucl: np.ndarray) -> np.ndarray:
        mask = self.image_cytoplasm > np.median(self.image_cytoplasm)
        mask = skimage.morphology.remove_small_objects(mask)
        mask_cell = skimage.segmentation.watershed(
            ~self.image_nucleus, mask_nucl, mask=mask, watershed_line=True
        )
        return mask_cell

    def requires(self):
        return Preprocess(FileID=self.FileID)

    def output(self):
        return luigi.LocalTarget(self.output_file)

    def run(self):
        # Load and normalize
        image = skimage.io.imread(self.input_file)
        self.image_nucleus = image[globalConfig().ChannelNucleus]
        self.image_cytoplasm = image[globalConfig().ChannelBackground]

        # Segment image
        mask_nucl = self.segment_nucleus(
            diameter_nucleus=globalConfig().NucleusDiameter,
            minimum_size=globalConfig().NucleusMinsize,
        )
        mask_cell = self.segment_cell(mask_nucl)
        segmap = np.stack([mask_nucl, mask_cell], axis=0).astype(np.uint16)
        skimage.io.imsave(self.output_file, segmap, check_contrast=False)
