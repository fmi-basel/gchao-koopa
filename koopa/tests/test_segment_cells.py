import pytest
import numpy as np

from koopa import segment_cells


@pytest.mark.usefixtures("segmap")
@pytest.mark.parametrize(
    "gaussian, min_size, min_distance, expected",
    [(1, 10, 10, 3), (1, 1000, 10, 2), (1, 10, 20, 2)],
)
def test_segment_otsu(segmap, gaussian, min_size, min_distance, expected):
    image = np.where(segmap == 2, 0, segmap).astype(bool)
    output = segment_cells.segment_otsu(image, gaussian, min_size, min_distance)
    assert len(np.unique(output)) == expected


@pytest.mark.usefixtures("segmap")
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


@pytest.mark.usefixtures("segmap")
def test_remove_border_objects(segmap):
    output = segment_cells.remove_border_objects(segmap)
    assert len(np.unique(output)) == 3  # bg + 2 objects


def test_segment_background():
    pass


def test_segment_nuclei():
    pass


def test_segment_both():
    pass
