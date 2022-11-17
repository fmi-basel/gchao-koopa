import numpy as np
import pandas as pd
import pytest

from koopa import colocalize


@pytest.fixture
def coords_one():
    return np.array([(5, 5), (10, 10), (20, 15)])


@pytest.fixture
def coords_two():
    return np.array([(9, 11), (5, 6), (20, 16)])


@pytest.fixture
def coords_three():
    return np.array([(0, 10), (15, 2), (2, 15)])


@pytest.fixture
def df_one(coords_one):
    return pd.DataFrame(
        {
            "x": [*coords_one.T[0]] * 3,
            "y": [*coords_one.T[1]] * 3,
            "particle": list(range(3)) * 3,
            "frame": [1, 1, 1, 2, 2, 2, 3, 3, 3],
        }
    )


@pytest.fixture
def df_two(coords_two):
    return pd.DataFrame(
        {
            "x": [*coords_two.T[0]] * 3,
            "y": [*coords_two.T[1]] * 3,
            "particle": list(range(3)) * 3,
            "frame": [1, 1, 1, 2, 2, 2, 3, 3, 3],
        }
    )


@pytest.mark.parametrize("distance", [0, 1, 5, 20])
def test_colocalize_frames_identical(coords_one, distance):
    rows, cols = colocalize.__colocalize_frames(coords_one, coords_one, distance)
    assert (rows == [0, 1, 2]).all()
    assert (rows == cols).all()


@pytest.mark.parametrize(
    "distance, out_rows, out_cols",
    [(0, [], []), (1, [0, 2], [1, 2]), (2, [0, 1, 2], [1, 0, 2])],
)
def test_colocalize_frames_matching(
    coords_one, coords_two, distance, out_rows, out_cols
):
    rows, cols = colocalize.__colocalize_frames(coords_one, coords_two, distance)
    assert (rows == out_rows).all()
    assert (cols == out_cols).all()


@pytest.mark.parametrize("distance, length", [(0, 0), (5, 0), (10, 2), (20, 3)])
def test_colocalize_frames_not_matching(coords_one, coords_three, distance, length):
    rows, cols = colocalize.__colocalize_frames(coords_one, coords_three, distance)
    assert len(rows) == len(cols) == length


@pytest.mark.parametrize(
    "min_frames, distance, length", [(0, 0, 3), (1, 1, 3), (2, 2, 3), (3, 0, 0)]
)
def test_colocalize_tracks_identical(df_one, min_frames, distance, length):
    rows, cols = colocalize.__colocalize_tracks(df_one, df_one, min_frames, distance)
    assert len(rows) == len(cols) == length
    if length == 3:
        assert (rows == [0, 1, 2]).all()


@pytest.mark.parametrize(
    "min_frames, distance, length", [(1, 0, 0), (0, 1, 2), (2, 2, 3)]
)
def test_colocalize_tracks_matching(df_one, df_two, min_frames, distance, length):
    rows, cols = colocalize.__colocalize_tracks(df_one, df_two, min_frames, distance)
    assert len(rows) == len(cols) == length


def test_colocalize_dff_identical(df_one):
    output = colocalize.colocalize_frames(df_one, df_one, "test", 1, 1)
    assert all(output["particle_test"] == output["coloc_particle_test"])


def test_colocalize_dff_matching(df_one, df_two):
    # Without 0-values
    output = colocalize.colocalize_frames(df_one, df_two, "", 1, 5)
    particles = output["particle_"].iloc[output["coloc_particle_"].values - 1]
    assert all(particles == output["coloc_particle_"].values)

    # With 0-values / non-colocs
    output = colocalize.colocalize_frames(df_one, df_two, "", 1, 1)
    index = output["coloc_particle_"].values[output["coloc_particle_"].values != 0]
    particles = output["particle_"].iloc[index - 1]
    assert all(particles == index)


def test_colocalize_dft_identical(df_one):
    output = colocalize.colocalize_tracks(df_one, df_one, 1, 1)
    assert all(output["particle"] == output["coloc_particle"])


def test_colocalize_dft_matching(df_one, df_two):
    output = colocalize.colocalize_tracks(df_one, df_two, 1, 1)
    assert output["coloc_particle"].isna().sum() == 6


def test_colocalize_dft_output_present(df_one, df_two):
    output = colocalize.colocalize_tracks(df_one, df_two, 5, 1)
    assert "coloc_particle" in output.columns
