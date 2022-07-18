"""Raw spot detection."""

import logging
import os

from tqdm import tqdm
import deepblink as pink
import luigi
import numpy as np
import pandas as pd
import tensorflow as tf
import tifffile
import trackpy as tp

from .config import General
from .config import SpotsDetection
from .preprocess import Preprocess

tp.quiet()


class Detect(luigi.Task):
    """Task for raw spot detection detect in an image."""

    FileID = luigi.Parameter()
    ChannelIndex = luigi.IntParameter()
    logger = logging.getLogger("luigi-interface")

    def requires(self):
        return Preprocess(FileID=self.FileID)

    def output(self):
        return luigi.LocalTarget(
            os.path.join(
                General().analysis_dir,
                f"detection_raw_c{SpotsDetection().channels[self.ChannelIndex]}",
                f"{self.FileID}.parq",
            )
        )

    def run(self):
        self.load_deepblink_model()
        image = tifffile.imread(self.requires().output().path)
        image_spots = image[SpotsDetection().channels[self.ChannelIndex]]

        df_spots = self.detect(image_spots)
        df_spots.insert(loc=0, column="FileID", value=self.FileID)
        df_spots.to_parquet(self.output().path)

    def detect_frame(self, image: np.ndarray) -> pd.DataFrame:
        """Detect spots in a single frame using deepBlink."""
        # Padding to allow for refinement at edges
        image = np.pad(
            image,
            SpotsDetection().refinement_radius + 1,
            mode="constant",
            constant_values=0,
        )

        # Prediction and refinement
        yx = pink.inference.predict(image=image, model=self.model)
        y, x = yx.T
        df = tp.refine_com(
            raw_image=image,
            image=image,
            radius=SpotsDetection().refinement_radius,
            coords=yx,
            engine="numba",
        )
        df["x"] = x - SpotsDetection().refinement_radius - 1
        df["y"] = y - SpotsDetection().refinement_radius - 1
        df = df.drop("raw_mass", axis=1)
        return df

    def detect(self, image: np.ndarray) -> pd.DataFrame:
        """Detect spots in an image series (single, z, or t)."""
        if image.ndim == 2:
            self.logger.info("Detecting spots in single frame")
            image = np.expand_dims(image, axis=0)
        if image.ndim != 3:
            raise ValueError(f"Image must be 3D. Got {image.ndim}D.")

        frames = []
        for frame, image_curr in tqdm(enumerate(image), total=image.shape[0]):
            df = self.detect_frame(image_curr)
            df["frame"] = frame
            df["channel"] = SpotsDetection().channels[self.ChannelIndex]
            frames.append(df)

        df = pd.concat(frames, ignore_index=True)
        return df

    def load_deepblink_model(self):
        """Set environment variables and load deepBlink model."""
        os.environ["OMP_NUM_THREADS"] = "10"
        os.environ["OPENBLAS_NUM_THREADS"] = "10"
        os.environ["MKL_NUM_THREADS"] = "10"
        os.environ["VECLIB_MAXIMUM_THREADS"] = "10"
        os.environ["NUMEXPR_NUM_THREADS"] = "10"

        os.environ["CUDA_VISIBLE_DEVICES"] = "None"
        os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
        tf.config.threading.set_intra_op_parallelism_threads(4)
        tf.config.threading.set_inter_op_parallelism_threads(4)

        self.model = pink.io.load_model(SpotsDetection().models[self.ChannelIndex])
        self.logger.info(
            f"Loaded model {SpotsDetection().models[self.ChannelIndex]} "
            f"for channel {SpotsDetection().channels[self.ChannelIndex]}"
        )
