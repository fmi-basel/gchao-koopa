"""Segment other features."""

import os

from tqdm import tqdm
import luigi
import numpy as np

with open(os.devnull, "w") as devnull:
    import segmentation_models as sm
import skimage.exposure
import skimage.filters
import skimage.morphology
import tensorflow as tf
import tifffile

from .config import General
from .config import SegmentationOther
from .preprocess import Preprocess


class SegmentOther(luigi.Task):
    """Task to segment the presence of a marker label."""

    FileID = luigi.Parameter()
    ChannelIndex = luigi.IntParameter()

    def requires(self):
        return Preprocess(FileID=self.FileID)

    def output(self):
        return luigi.LocalTarget(
            os.path.join(
                General().analysis_dir,
                f"segmentation_c{SegmentationOther().channels[self.ChannelIndex]}",
                f"{self.FileID}.tif",
            )
        )

    def run(self):
        image = tifffile.imread(self.requires().output().path)
        image_other = image[SegmentationOther().channels[self.ChannelIndex]]

        segmap = self.segment(image_other)
        skimage.io.imsave(self.output().path, segmap, check_contrast=False)

    def normalize(self, image: np.ndarray) -> np.ndarray:
        """Apply default model normalization and z-normalization."""
        preprocessor = sm.get_preprocessing(
            SegmentationOther().backbones[self.ChannelIndex]
        )
        image = preprocessor(image)
        image = (image - image.mean()) / image.std()
        return image

    @staticmethod
    def next_power(x: int, k: int = 2) -> int:
        """Calculate x's next higher power of k."""
        y, power = 0, 1
        while y < x:
            y = k ** power
            power += 1
        return y

    def add_padding(self, image: np.ndarray) -> np.ndarray:
        """Add padding to image to avoid edge effects."""
        self.pad_bottom = self.next_power(image.shape[0], 2) - image.shape[0]
        self.pad_right = self.next_power(image.shape[1], 2) - image.shape[1]
        image_pad = np.pad(
            image, ((0, self.pad_bottom), (0, self.pad_right)), "reflect"
        )
        return image_pad

    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """Prepare image to be fed into model."""
        image = self.normalize(image)
        image = self.add_padding(image)
        image = np.stack((image,) * 3, axis=-1)
        return image[None]

    def segment_deep(self, image: np.ndarray) -> np.ndarray:
        """Segmentation using deep `segmentation_models`."""
        image = self.preprocess_image(image)
        mask = self.model.predict(image).squeeze()
        mask = mask[: -self.pad_bottom, : -self.pad_right]
        mask = np.round(mask)
        return mask

    def segment(self, image: np.ndarray) -> np.ndarray:
        """Segment marker labels with pre-trained model or traditionally."""
        method = SegmentationOther().methods[self.ChannelIndex]

        # Preparation
        if not General().do_3D and General().do_TimeSeries:
            image = np.max(image, axis=0)
        if not General().do_3D:
            image = np.expand_dims(image, axis=0)

        # Segmentation
        if method == "deep":
            self.model = tf.keras.models.load_model(
                SegmentationOther().models[self.ChannelIndex],
                custom_objects={
                    "binary_crossentropy_plus_jaccard_loss": sm.losses.bce_jaccard_loss,
                    "iou_score": sm.metrics.iou_score,
                },
            )

        def __single_frame(image: np.ndarray) -> np.ndarray:
            if method == "deep":
                return self.segment_deep(image)
            elif method == "otsu":
                return image > skimage.filters.threshold_otsu(image)
            elif method == "li":
                return image > skimage.filters.threshold_li(image)
            elif method == "multiotsu":
                # Default 3 classes -> 2nd highest class chosen
                return image > skimage.filters.threshold_multiotsu(image)[1]
            raise ValueError(f"Unknown other segmentation method {method}.")

        masks = [__single_frame(i) for i in tqdm(image)]
        return np.array(masks).squeeze()
