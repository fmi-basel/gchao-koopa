import numpy as np
import pytest

from koopa import preprocess


@pytest.fixture
def image():
    image = np.zeros((20, 100, 100))
    image[9, 50:, 50:] = 1
    image[10] = np.random.random((100, 100))
    return image


def test_get_sharpest_slice(image):
    sharpest_slice = preprocess.get_sharpest_slice(image)
    assert (sharpest_slice == image[10]).all()


@pytest.mark.parametrize("method", ["maximum", "mean", "sharpest"])
def test_register_3d_image(image, method):
    output = preprocess.register_3d_image(
        np.expand_dims(image, axis=0), method
    ).squeeze()
    assert output.shape == (100, 100)


def test_crop_image(image):
    output = preprocess.crop_image(np.expand_dims(image, axis=0), 50, 100).squeeze()
    assert (output[9] == 1).all()


def test_bin_image(image):
    output = preprocess.bin_image(image, (1, 0.5, 0.5))
    assert output.shape == (20, 50, 50)

    # Add 1 padding for interpolation
    assert (output[9, 26:, 26:] == 1).all()


def test_trim_image(image):
    output = preprocess.trim_image(np.expand_dims(image, axis=0), 8, 16).squeeze()
    assert len(output) == 8
    assert (output[1] == image[9]).all()
