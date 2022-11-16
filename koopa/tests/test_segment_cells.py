import pytest
import numpy as np

from koopa import segment_cells


@pytest.fixture
def segmap():
    x, y = np.indices((80, 80))
    x1, y1, x2, y2, x3, y3 = 5, 10, 28, 28, 44, 52
    r1, r2, r3 = 13, 16, 20
    mask_circle1 = ((x - x1) ** 2 + (y - y1) ** 2 < r1 ** 2) * 1
    mask_circle2 = ((x - x2) ** 2 + (y - y2) ** 2 < r2 ** 2) * 2
    mask_circle3 = ((x - x3) ** 2 + (y - y3) ** 2 < r3 ** 2) * 3
    image = np.max([mask_circle1, mask_circle2, mask_circle3], axis=0)
    return image


@pytest.mark.parametrize(
    "gaussian, min_size, min_distance, expected",
    [(1, 10, 10, 3), (1, 1000, 10, 2), (1, 10, 20, 2)],
)
def test_segment_otsu(segmap, gaussian, min_size, min_distance, expected):
    image = np.where(segmap == 2, 0, segmap).astype(bool)
    output = segment_cells.segment_otsu(image, gaussian, min_size, min_distance)
    assert len(np.unique(output)) == expected


def test_segment_cellpose(segmap):
    image = np.where(segmap == 2, 0, segmap).astype(bool)
    output = segment_cells.segment_cellpose(
        image,
        model="nuclei",
        pretrained=[],
        do_3d=False,
        diameter=30,
        resample=False,
        min_size_nuclei=10,
    )
    assert len(np.unique(output)) == 3


def test_remove_border_objects(segmap):
    output = segment_cells.remove_border_objects(segmap)
    assert len(np.unique(output)) == 3  # bg + 2 objects


def test_segment_background():
    pass


def test_segment_nuclei():
    pass


def test_segment_both():
    pass
