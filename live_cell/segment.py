import os
import sys

from cellpose import models
import luigi
import numpy as np
import skimage.io
import torch

from config import CustomConfig
from preprocess import Preprocess


class Segment(luigi.Task):
    """Task to segment cells in 3D."""

    FileID = luigi.Parameter()

    def requires(self):
        return Preprocess(FileID=self.FileID)

    def output(self):
        return luigi.LocalTarget(
            os.path.join(
                CustomConfig().analysis_dir, "segmentation", f"{self.FileID}.tif"
            )
        )

    def run(self):
        # Load and normalize
        image = skimage.io.imread(self.requires().output().path)
        image_cytoplasm = image[CustomConfig().channel_suntag]
        segmap = self.segment(image_cytoplasm)
        skimage.io.imsave(self.output().path, segmap, check_contrast=False)

    @staticmethod
    def segment(image: np.ndarray) -> np.ndarray:
        """Segment a file using cellpose into nuclear maps.

        Args:
            diameter_nucleus: Expected diameter of the nucleus in px to
                be passed to the CellPose model.
            minimum_size: Minimum area in px of the nucleus.
        """
        torch.set_num_threads(4)
        cellpose_model = models.Cellpose(model_type="cyto", gpu=False)

        image = np.max(image, axis=0).astype(np.uint16)
        sys.stdout = open(os.devnull, "w")
        segmap, *_ = cellpose_model.eval(
            [image],
            diameter=CustomConfig().cytoplasm_diameter,
            resample=CustomConfig().cytoplasm_resample,
            channels=[0, 0],
            min_size=CustomConfig().cytoplasm_minsize,
        )
        segmap = segmap[0]
        sys.stdout = sys.__stdout__
        return segmap
