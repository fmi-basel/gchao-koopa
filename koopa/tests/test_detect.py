import os

import deepblink as pink
import numpy as np
import pytest
import skimage.io

from koopa import detect


@pytest.fixture
def model():
    fname = os.path.join(os.path.dirname(__file__), "./data/pink_model.h5")
    return pink.io.load_model(fname)


@pytest.fixture
def image() -> np.ndarray:
    fname = os.path.join(os.path.dirname(__file__), "./data/spots.png")
    return skimage.io.imread(fname)


@pytest.mark.parametrize("radius", [1, 2, 5])
def test_detect_frame(image, model, radius):
    df = detect.detect_frame(image, model, refinement_radius=radius)
    assert len(df) == 3
    assert all(
        col in df.columns
        for col in ["x", "y", "mass", "eccentricity", "signal", "size"]
    )


@pytest.mark.parametrize("index", [0, 1])
def test_detect_multiple_frames(image, model, index):
    stack = np.stack([image, image], axis=0)

    df = detect.detect_image(
        stack, index_channel=index, model=model, refinement_radius=1
    )
    assert len(df) == 3
    assert "frame" in df.columns
    assert df["frame"].unique() == [0]

    assert "channel" in df.columns
    assert df["channel"].unique() == [index]


def test_detect_error_too_few_channels(model):
    with pytest.raises(ValueError):
        detect.detect_image(
            np.zeros((100,)), index_channel=0, model=model, refinement_radius=1
        )


def test_detect_error_too_many_channels(model):
    with pytest.raises(ValueError):
        detect.detect_image(
            np.zeros((10, 10, 10, 10, 10)),
            index_channel=0,
            model=model,
            refinement_radius=1,
        )
