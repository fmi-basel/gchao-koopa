import numpy as np
import pandas as pd
import pytest

from koopa import postprocess


@pytest.fixture
def image():
    image = np.zeros((10, 10))
    image[0, 5] = 1.1
    image[3, 3] = 1.2
    image[9, 9] = 1.3
    return image


@pytest.mark.parametrize(
    "x, y, expected", [(0, 0, 0), (5, 0, 1.1), (3, 3, 1.2), (15, 15, 1.3)]
)
def test_get_value_2d(image, x, y, expected):
    output = postprocess.get_value(pd.Series(dict(x=x, y=y)), image)
    assert output == expected


@pytest.mark.parametrize(
    "x, y, frame, expected",
    [(0, 0, 0, 0), (5, 0, 2, 1.1), (3, 3, 1, 1.2)],
)
def test_get_value_3d(image, x, y, frame, expected):
    image = np.stack([image] * 3, axis=0)
    output = postprocess.get_value(pd.Series(dict(x=x, y=y, frame=frame)), image)
    assert output == expected


@pytest.mark.usefixtures("segmap")
@pytest.mark.parametrize(
    "x, y, cell_id, expected", [(28, 28, 0, 8), (52, 44, 0, 10), (52, 44, 3, 9.5)]
)
def test_get_distance_from_segmap(segmap, x, y, cell_id, expected):
    df = pd.DataFrame(dict(x=[x], y=[y], cell_id=[cell_id]))
    output = postprocess.get_distance_from_segmap(df, segmap)[0]
    assert output == expected


@pytest.mark.usefixtures("segmap")
def test_get_cell_properties(segmap):
    output = postprocess.get_cell_properties(segmap, "test", False)
    assert "area_test" in output.columns
    assert "eccentricity_test" in output.columns
    assert output.loc[2, "area_test"] == 1245  # pi*r**2 ~1250
    assert output.loc[2, "eccentricity_test"] == 0  # perfect circle
    assert output.loc[1, "eccentricity_test"] > 0  # imperfect circle


def test_add_segmentation_data():
    pass
