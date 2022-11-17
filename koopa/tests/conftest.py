import pytest
import numpy as np


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
