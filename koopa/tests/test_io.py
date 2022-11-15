import configparser
import os

import numpy as np
import pytest

from koopa import io


def test_find_full_path_normal(tmp_path):
    subdir = os.path.join(tmp_path, "sub", "dir")
    os.makedirs(subdir)

    full_path = os.path.join(subdir, "test.ext")
    with open(full_path, "w"):
        pass

    assert io.find_full_path(tmp_path, "test", "ext") == full_path
    assert io.find_full_path(subdir, "test", "ext") == full_path


def test_find_full_path_error():
    with pytest.raises(ValueError):
        io.find_full_path("does_not_exits/", "test", "ext")


@pytest.mark.parametrize(
    "path, bname",
    [
        ("/dir/dir/file.ext", "file"),
        ("./../../file.ext.txt", "file.ext"),
        ("file", "file"),
    ],
)
def test_basename(path, bname):
    assert io.basename(path) == bname


def test_load_czi():
    # TODO example czi
    # check shape
    pass


@pytest.mark.parametrize(
    "name, shape",
    [
        ("./data/Beads_1.nd", (2, 1024, 1024)),
        ("./data/20221115_2z_1.nd", (2, 2, 1200, 1200)),
    ],
)
def test_load_nd(name, shape):
    fname = os.path.join(os.path.dirname(__file__), name)
    image = io.load_nd(fname)
    assert isinstance(image, np.ndarray)
    assert image.shape == shape


def test_load_image_tif():
    fname = os.path.join(os.path.dirname(__file__), "./data/example_tif.tif")
    image = io.load_image(fname)
    assert isinstance(image, np.ndarray)
    assert image.shape == (2, 1200, 1200)


def test_load_image_not_tif():
    fname = os.path.join(os.path.dirname(__file__), "./data/Beads_1.nd")
    with pytest.raises(ValueError):
        io.load_image(fname)


def test_load_config():
    fname = os.path.join(os.path.dirname(__file__), "./data/example.cfg")
    config = io.load_config(fname)
    assert isinstance(config, configparser.ConfigParser)


def test_save_image(tmp_path):
    fname = os.path.join(tmp_path, "image.tif")
    image = np.zeros((10, 10))
    io.save_image(fname, image)
    assert os.path.isfile(fname)


def test_save_image_recursive(tmp_path):
    fname = os.path.join(tmp_path, "sub", "dir", "image.tif")
    image = np.zeros((10, 10))
    io.save_image(fname, image)
    assert os.path.isfile(fname)
