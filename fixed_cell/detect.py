import os

import deepblink as pink
import luigi
import numpy as np
import pandas as pd
import skimage
import skimage.filters
import tensorflow as tf
import trackpy as tp

from config import CustomConfig
from preprocess import Preprocess

tp.quiet()


class Detect(luigi.Task):
    """Task for raw spot detection detect in an image."""

    FileID = luigi.Parameter()
    SpotChannel = luigi.IntParameter()

    @property
    def input_file(self):
        return os.path.join(
            CustomConfig().analysis_dir, "preprocessed", f"{self.FileID}.tif"
        )

    @property
    def output_file(self):
        return os.path.join(
            CustomConfig().analysis_dir,
            f"detection_c{self.SpotChannel}",
            f"{self.FileID}.parq",
        )

    def detect(
        self,
        image: np.ndarray,
        model_spots: tf.keras.models.Model,
        refinement_radius: int,
    ) -> pd.DataFrame:
        """Detect spots in an image using deepBlink.
        
        Args:
            model_spots: deepBlink model for spot detection.
            refinement_radius: Radius of spot refinement / intensity measurements.
        """
        if image.ndim != 2:
            raise ValueError("Image must be 2D.")
        image = np.pad(image, refinement_radius + 1, mode="constant", constant_values=0)

        # deepBlink prediction
        yx = pink.inference.predict(image=image, model=model_spots)
        y, x = yx.T
        df = tp.refine_com(
            raw_image=image,
            image=image,
            radius=refinement_radius,
            coords=yx,
            engine="numba",
        )
        df["x"] = x - refinement_radius - 1
        df["y"] = y - refinement_radius - 1
        df["c"] = self.SpotChannel
        return df

    @staticmethod
    def configure_tensorflow():
        os.environ["CUDA_VISIBLE_DEVICES"] = "None"
        os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
        tf.config.threading.set_inter_op_parallelism_threads(8)
        tf.config.threading.set_intra_op_parallelism_threads(8)

    def requires(self):
        return Preprocess(FileID=self.FileID)

    def output(self):
        return luigi.LocalTarget(self.output_file)

    def run(self):
        self.configure_tensorflow()
        image = skimage.io.imread(self.input_file)
        image_spots = image[CustomConfig().channel_spots]
        model = pink.io.load_model(CustomConfig().model_spots)
        df_spots = self.detect(image_spots, model, CustomConfig().refinement_radius)
        df_spots.to_parquet(self.output_file)
