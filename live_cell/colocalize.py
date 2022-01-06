from typing import Tuple
import os

import luigi
import numpy as np
import pandas as pd
import scipy

from config import CustomConfig
from track import Track


class Colocalize(luigi.Task):

    FileID = luigi.Parameter()

    def requires(self):
        return [
            Track(FileID=self.FileID, Channel="ms2"),
            Track(FileID=self.FileID, Channel="suntag"),
        ]

    def output(self):
        return luigi.LocalTarget(
            os.path.join(
                CustomConfig().analysis_dir, "colocalization", f"{self.FileID}.parq"
            )
        )

    def run(self):
        track_ms2 = pd.read_parquet(self.requires()[0].output().path)
        track_sun = pd.read_parquet(self.requires()[1].output().path)

        # Colocalize both channels
        coloc_ms2, coloc_sun = self.colocalize_tracks(track_ms2, track_sun)
        for idx_ms2, idx_sun in zip(coloc_ms2, coloc_sun):
            track_ms2.loc[
                track_ms2["particle"] == idx_ms2, ["coloc_particle"]
            ] = idx_sun
            track_sun.loc[
                track_sun["particle"] == idx_sun, ["coloc_particle"]
            ] = idx_ms2

        # Merge
        track = pd.concat([track_ms2, track_sun])
        track.insert(loc=0, column="FileID", value=self.FileID)
        track.to_parquet(self.output().path)

    @staticmethod
    def colocalize_frame(frame_1: pd.DataFrame, frame_2: pd.DataFrame):
        """Single frame colocalization.

        Euclidean distance based linear sum assigment between coordinates in
            frame_1 and frame_2. Removes all assignments above the distance_cutoff.
        """
        # Euclidean distance matrix
        cdist = scipy.spatial.distance.cdist(
            frame_1[["y", "x"]].to_numpy(),
            frame_2[["y", "x"]].to_numpy(),
            metric="euclidean",
        )

        # Linear sum assignment
        rows, cols = scipy.optimize.linear_sum_assignment(cdist)

        # Distance cutoff
        for r, c in zip(rows, cols):
            if cdist[r, c] > CustomConfig().distance_cutoff:
                rows = rows[rows != r]
                cols = cols[cols != c]

        return rows, cols

    def colocalize_tracks(
        self, track_ms2: pd.DataFrame, track_sun: pd.DataFrame
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Full track colocalization between track_ms2, track_sun.

        Do single frame colocalization (track.colocalize_frame) across
            the entire movie. Return the particle numbers which
            colocalized in along at least "min_coloc_tracks" frames.
        """

        # Create track assignment matrix
        assignments = np.zeros(
            (track_ms2["particle"].nunique(), track_sun["particle"].nunique())
        )

        for frame in track_ms2["frame"].unique():
            # Slice to get current frame
            frame_ms2 = track_ms2[track_ms2["frame"] == frame].reset_index()
            frame_sun = track_sun[track_sun["frame"] == frame].reset_index()

            # Colocalize single frames below distance_cutoff
            rows, cols = self.colocalize_frame(frame_ms2, frame_sun)

            # Assign "real" track-particle number
            real_rows = np.array([frame_ms2.particle[i] for i in rows])
            real_cols = np.array([frame_sun.particle[i] for i in cols])

            # Update assignment matrix, IndexError if none colocalizing / empty
            try:
                assignments[real_rows, real_cols] += 1
            except IndexError:
                pass

        # Get colocalizing track numbers from assignment matrix
        coloc_ms2, coloc_sun = np.where(
            np.where(assignments, assignments > CustomConfig().min_coloc_frames, 0)
        )

        return coloc_ms2, coloc_sun
