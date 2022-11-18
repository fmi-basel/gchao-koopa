"""Segment other features."""

import os

import numpy as np

with open(os.devnull, "w") as devnull:
    import segmentation_models as sm
import skimage.exposure
import skimage.filters
import skimage.morphology
import tensorflow as tf


class DeepSegmentation:
    """Wrapper to use segmentation_models based segmentation."""

    def __init__(self, fname_model: str, backbone: str) -> None:
        self.backbone = backbone
        self.model = tf.keras.models.load_model(
            fname_model,
            custom_objects={
                "binary_crossentropy_plus_jaccard_loss": sm.losses.bce_jaccard_loss,
                "iou_score": sm.metrics.iou_score,
            },
        )

    def segment(self, image: np.ndarray) -> np.ndarray:
        """Segmentation using deep `segmentation_models`."""
        image = self.preprocess_image(image)
        mask = self.model.predict(image).squeeze()
        mask = mask[: -self.pad_bottom, : -self.pad_right]
        mask = np.round(mask)
        return mask

    def normalize(self, image: np.ndarray) -> np.ndarray:
        """Apply default model normalization and z-normalization."""
        # SegmentationOther().backbones[self.ChannelIndex]
        preprocessor = sm.get_preprocessing(self.backbone)
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


def segment(image: np.ndarray, index_list: int, config: dict) -> np.ndarray:
    """Segment marker labels with pre-trained model or traditionally."""
    method = config["sego_methods"][index_list]
    methods = ["deep", "otsu", "li", "multiotsu"]
    if method not in methods:
        raise ValueError(
            f"Unknown other segmentation method {method}. "
            f"Please select one of {methods}"
        )

    # Preparation
    if not config["do_3d"] and config["do_timeseries"]:
        image = np.max(image, axis=0)
    if not config["do_3d"]:
        image = np.expand_dims(image, axis=0)

    # Segmentation
    if method == "deep":
        segmenter = DeepSegmentation(
            fname_model=config["sego_models"][index_list],
            backbone=config["sego_backbones"][index_list],
        )

    masks = []
    for frame in image:
        if method == "deep":
            mask = segmenter.segment(frame)
        if method == "otsu":
            mask = image > skimage.filters.threshold_otsu(frame)
        if method == "li":
            mask = image > skimage.filters.threshold_li(frame)
        if method == "multiotsu":
            # Default 3 classes -> 2nd highest class chosen
            mask = image > skimage.filters.threshold_multiotsu(frame)[1]
        masks.append(mask)

    return np.array(masks).squeeze()
