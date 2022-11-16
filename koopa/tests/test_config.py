import os

import pytest

from koopa import io
from koopa import config


@pytest.fixture
def cfg():
    fname = os.path.join(os.path.dirname(__file__), "./data/example.cfg")
    cfg = io.load_config(fname)
    cfg["SpotsDetection"]["detect_models"] = str(
        [
            os.path.join(os.path.dirname(__file__), "./data/pink_model.h5"),
            os.path.join(os.path.dirname(__file__), "./data/pink_model.h5"),
        ]
    )
    return cfg


def test_validate_config_valid(cfg):
    assert config.validate_config(cfg)


def test_validate_config_bad_item(cfg):
    cfg["General"]["I-do-not-exist"] = "False"
    with pytest.raises(ValueError):
        config.validate_config(cfg)


def test_validate_config_bad_boolean(cfg):
    cfg["General"]["do_3d"] = "not-really-true"
    with pytest.raises(ValueError):
        config.validate_config(cfg)


def test_validate_config_bad_integer(cfg):
    cfg["SpotsTracking"]["gap_frames"] = "1.25"
    with pytest.raises(ValueError):
        config.validate_config(cfg)


def test_validate_config_bad_list(cfg):
    cfg["SpotsDetection"]["detect_channels"] = str([3.5, "4", True])
    with pytest.raises(ValueError):
        config.validate_config(cfg)


def test_validate_config_bad_path(cfg):
    cfg["General"]["image_dir"] = "/path/to/nowhere"
    with pytest.raises(ValueError):
        config.validate_config(cfg)


def test_validate_config_unequal_channels(cfg):
    cfg["SpotsDetection"]["channels"] = str([0, 1])
    cfg["SpotsDetection"]["detect_models"] = str(["./data/pink_models.h5"])
    with pytest.raises(ValueError):
        config.validate_config(cfg)


def test_validate_config_ignore_block(cfg):
    cfg["SpotsColocalization"]["coloc_enabled"] = "True"
    cfg["SpotsColocalization"]["bad-input"] = "very-bad"
    with pytest.raises(ValueError):
        config.validate_config(cfg)


def test_add_versioning(cfg):
    output = config.add_versioning(cfg)
    assert "Versioning" in output.sections()


def test_flatten_config(cfg):
    output = config.flatten_config(cfg)
    assert isinstance(output, dict)
    assert isinstance(output["do_3d"], bool)
    assert isinstance(output["cellpose_diameter"], int)
