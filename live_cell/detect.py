import os

from tqdm import tqdm
import deepblink as pink
import luigi
import numpy as np
import pandas as pd
import skimage
import tensorflow as tf
import trackpy as tp

from config import CustomConfig
from preprocess import Preprocess

tp.quiet()


class Detect(luigi.Task):
    """Task for raw spot detection detect in an image."""

    FileID = luigi.Parameter()
    Channel = luigi.Parameter()

    def requires(self):
        return Preprocess(FileID=self.FileID)

    def output(self):
        return luigi.LocalTarget(
            os.path.join(
                CustomConfig().analysis_dir,
                f"detection_{self.Channel}",
                f"{self.FileID}.parq",
            )
        )

    def run(self):
        self.configure_tensorflow()
        image = skimage.io.imread(self.requires().output().path)
        image_spots = image[CustomConfig().__getattribute__(f"channel_{self.Channel}")]

        self.model = pink.io.load_model(
            CustomConfig().__getattribute__(f"model_{self.Channel}")
        )
        df_spots = self.detect(image_spots)
        df_spots.to_parquet(self.output().path)

    def detect_frame(self, image: np.ndarray) -> pd.DataFrame:
        """Detect spots in a single frame using deepBlink."""
        if image.ndim != 2:
            raise ValueError("Image must be 2D.")
        image = np.pad(
            image,
            CustomConfig().intensity_radius + 1,
            mode="constant",
            constant_values=0,
        )

        yx = pink.inference.predict(image=image, model=self.model)
        y, x = yx.T
        df = tp.refine_com(
            raw_image=image,
            image=image,
            radius=CustomConfig().intensity_radius,
            coords=yx,
            engine="numba",
        )
        df["x"] = x - CustomConfig().intensity_radius - 1
        df["y"] = y - CustomConfig().intensity_radius - 1
        df = df.drop("raw_mass", axis=1)
        return df

    def detect(self, image: np.ndarray) -> pd.DataFrame:
        """Detect spots in an image series using deepBlink."""
        frames = []

        for frame, image_curr in tqdm(
            enumerate(image), total=image.shape[0], desc=f"Detecting {self.Channel}"
        ):
            df = self.detect_frame(image_curr)
            df["frame"] = frame
            df["channel"] = self.Channel
            frames.append(df)

        df = pd.concat(frames, ignore_index=True)
        return df

    @staticmethod
    def configure_tensorflow():
        os.environ["OMP_NUM_THREADS"] = "10"
        os.environ["OPENBLAS_NUM_THREADS"] = "10"
        os.environ["MKL_NUM_THREADS"] = "10"
        os.environ["VECLIB_MAXIMUM_THREADS"] = "10"
        os.environ["NUMEXPR_NUM_THREADS"] = "10"
        os.environ["CUDA_VISIBLE_DEVICES"] = "None"
        tf.config.threading.set_intra_op_parallelism_threads(4)
        tf.config.threading.set_inter_op_parallelism_threads(4)
