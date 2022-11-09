"""Frame and track colocalization."""

from typing import Tuple
import logging
import os

import numpy as np
import pandas as pd
import scipy


def __colocalize_frames(
    coords_one: np.ndarray, coords_two: np.ndarray, distance_cutoff: float
) -> Tuple[np.ndarray]:
    """Single frame colocalization.

    Euclidean distance based linear sum assigment between coordinates in
            frame_one and frame_two. Removes all assignments above the distance_cutoff.
    """
    cdist = scipy.spatial.distance.cdist(coords_one, coords_two, metric="euclidean")
    rows, cols = scipy.optimize.linear_sum_assignment(cdist)

    # Distance cutoff
    for r, c in zip(rows, cols):
        if cdist[r, c] > distance_cutoff:
            rows = rows[rows != r]
            cols = cols[cols != c]

    return rows, cols


def __colocalize_tracks(
    track_one: pd.DataFrame,
    track_two: pd.DataFrame,
    min_frames: int,
    distance_cutoff: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Full track colocalization between track_one, track_two.

    Do single frame colocalization (track.colocalize_frame) across
        the entire movie. Return the particle numbers which
        colocalized in along at least "min_coloc_tracks" frames.
    """
    # Create track assignment matrix
    assignments = np.zeros(
        (track_one["particle"].nunique(), track_two["particle"].nunique())
    )

    for frame in track_one["frame"].unique():
        # Slice to get current frame
        frame_one = track_one[track_one["frame"] == frame].reset_index(drop=True)
        frame_two = track_two[track_two["frame"] == frame].reset_index(drop=True)

        # Colocalize single frames below distance_cutoff
        coords_one = frame_one[["y", "x"]].to_numpy()
        coords_two = frame_two[["y", "x"]].to_numpy()
        rows, cols = __colocalize_frames(coords_one, coords_two, distance_cutoff)

        # Assign "real" track-particle number
        real_rows = np.array([frame_one.particle[i] for i in rows])
        real_cols = np.array([frame_two.particle[i] for i in cols])

        # Update assignment matrix, IndexError if none colocalizing / empty
        try:
            assignments[real_rows, real_cols] += 1
        except IndexError:
            pass
            # logger.exception("Could not update assignment matrix.")

    # Get colocalizing track numbers from assignment matrix
    coloc_one, coloc_two = np.where(np.where(assignments, assignments > min_frames, 0))

    return coloc_one, coloc_two


def colocalize_frames(
    df_one: pd.DataFrame,
    df_two: pd.DataFrame,
    name: str,
    z_distance: float,
    distance_cutoff: int,
) -> pd.DataFrame:
    coords_one = df_one[["y", "x", "frame"]].to_numpy()
    coords_two = df_two[["y", "x", "frame"]].to_numpy()
    coords_one[:, 2] *= z_distance
    coords_two[:, 2] *= z_distance

    # Colocalize both channels
    coloc_one, coloc_two = __colocalize_frames(coords_one, coords_two, distance_cutoff)
    df_one[f"particle_{name}"] = df_one.index + 1
    df_two[f"particle_{name}"] = df_two.index + 1
    df_one[f"coloc_particle_{name}"] = 0
    df_two[f"coloc_particle_{name}"] = 0
    df_one.loc[coloc_one, f"coloc_particle_{name}"] = coloc_two + 1
    df_two.loc[coloc_two, f"coloc_particle_{name}"] = coloc_one + 1

    # Merge
    df = pd.concat([df_one, df_two])
    return df


def colocalize_tracks(
    df_one: pd.DataFrame,
    df_two: pd.DataFrame,
    min_frames: int,
    distance_cutoff: int,
) -> pd.DataFrame:
    # Colocalize both channels
    coloc_one, coloc_two = __colocalize_tracks(
        df_one, df_two, min_frames, distance_cutoff
    )
    for idx_one, idx_two in zip(coloc_one, coloc_two):
        df_one.loc[df_one["particle"] == idx_one, ["coloc_particle"]] = idx_two
        df_two.loc[df_two["particle"] == idx_two, ["coloc_particle"]] = idx_one

    # Merge
    df = pd.concat([df_one, df_two])
    return df