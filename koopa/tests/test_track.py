import pandas as pd
import pytest

from koopa import track


@pytest.fixture
def df_spots():
    # Two tracks (1px, 5px), two fakes
    df = pd.DataFrame(
        {
            "x": [0, 1, 2, 10, 10, 10, 20, 0, 25],
            "y": [0, 0, 0, 5, 10, 15, 30, 5, 64],
            "frame": [0, 1, 2, 0, 1, 2, 0, 1, 2],
            "mass": [1, 10, 1, 10, 1, 1, 1, 10, 1],
            "particle": [0, 0, 5, 5, 3, 3, 2, 2, 0],
        }
    )
    return df


def test_track(df_spots):
    output = track.track(df_spots, search_range=1, gap_frames=0, min_length=3)
    assert len(output) == 3
    assert output["particle"].nunique() == 1

    output = track.track(df_spots, search_range=5, gap_frames=0, min_length=3)
    assert len(output) == 6
    assert output["particle"].nunique() == 2


def test_clean_particles(df_spots):
    output = track.clean_particles(df_spots)
    assert all(output["particle"] == [0, 0, 1, 1, 2, 2, 3, 3, 0])


def test_link_brightest_particles(df_spots):
    df_track = track.track(df_spots, search_range=5, gap_frames=0, min_length=3)
    output = track.link_brightest_particles(df_spots, df_track)
    assert len(output) == 5  # NMS 6->2 + non NMS 3->3
    assert len(output.dropna()) == 2
    assert output.dropna()["mass"].nunique() == 1
    assert output.dropna()["mass"].unique()[0] == 10


def test_subtract_drift():
    pass
