import sys

import deepblink as pink
import numpy as np
import pytest
import skimage.io

sys.path.append("../")
from koopa.detect import Detect


@pytest.fixture
def task_detect() -> Detect:
    task = Detect(FileID="example", ChannelIndex=0)
    task.model = pink.io.load_model(
        "/tungstenfs/scratch/gchao/deepblink/pink_particle.h5"
    )
    return task


@pytest.fixture
def image() -> np.ndarray:
    return skimage.io.imread("./tests/data/spots.png")


def test_detect_frame(task_detect, image):
    df = task_detect.detect_frame(image)
    assert len(df) == 3
    assert all(
        col in df.columns
        for col in ["x", "y", "mass", "eccentricity", "signal", "size"]
    )


def test_detect_single_frame(task_detect, image):
    df = task_detect.detect(image, channels=[0])
    assert len(df) == 3
    assert "frame" in df.columns


def test_detect_multiple_frames(task_detect, image):
    df = task_detect.detect(np.stack([image, image], axis=0), channels=[20])
    assert len(df) == 6
    assert "frame" in df.columns
    assert all(i in df["frame"].unique() for i in [0, 1])
    assert "channel" in df.columns
    assert df["channel"].unique() == 20


def test_detect_error(task_detect):
    with pytest.raises(ValueError):
        task_detect.detect(np.zeros((100,)))
    with pytest.raises(ValueError):
        task_detect.detect(np.zeros((10, 10, 10, 10)))
