from typing import Tuple
import os

import luigi
import numpy as np
import pandas as pd
import scipy

from config import General, SpotsDetection
from config import SpotsColocalization
from track import Track


class Colocalize(luigi.Task):

    FileID = luigi.Parameter()
    ChannelPairIndex = luigi.IntParameter()

    @property
    def channel_pair(self) -> Tuple[int, int]:
        return SpotsColocalization().channels[self.ChannelPairIndex]

    def requires(self):
        channel_detect = SpotsDetection().channels
        index_one = channel_detect.index(self.channel_pair[0])
        index_two = channel_detect.index(self.channel_pair[1])
        return [
            Track(FileID=self.FileID, ChannelIndex=index_one),
            Track(FileID=self.FileID, ChannelIndex=index_two),
        ]

    def output(self):
        return luigi.LocalTarget(
            os.path.join(
                General().analysis_dir,
                f"colocalization_{self.channel_pair[0]}-{self.channel_pair[1]}",
                f"{self.FileID}.parq",
            )
        )

    def run(self):
        track_one = pd.read_parquet(self.requires()[0].output().path)
        track_two = pd.read_parquet(self.requires()[1].output().path)

        # Colocalize both channels
        coloc_one, coloc_two = self.colocalize_tracks(track_one, track_two)
        for idx_one, idx_two in zip(coloc_one, coloc_two):
            track_one.loc[
                track_one["particle"] == idx_one, ["coloc_particle"]
            ] = idx_two
            track_two.loc[
                track_two["particle"] == idx_two, ["coloc_particle"]
            ] = idx_one

        # Merge
        track = pd.concat([track_one, track_two])
        track.insert(loc=0, column="FileID", value=self.FileID)
        track.to_parquet(self.output().path)

    @staticmethod
    def colocalize_frame(frame_one: pd.DataFrame, frame_two: pd.DataFrame):
        """Single frame colocalization.

        Euclidean distance based linear sum assigment between coordinates in
            frame_one and frame_two. Removes all assignments above the distance_cutoff.
        """
        # Euclidean distance matrix
        cdist = scipy.spatial.distance.cdist(
            frame_one[["y", "x"]].to_numpy(),
            frame_two[["y", "x"]].to_numpy(),
            metric="euclidean",
        )

        # Linear sum assignment
        rows, cols = scipy.optimize.linear_sum_assignment(cdist)

        # Distance cutoff
        for r, c in zip(rows, cols):
            if cdist[r, c] > SpotsColocalization().distance_cutoff:
                rows = rows[rows != r]
                cols = cols[cols != c]

        return rows, cols

    def colocalize_tracks(
        self, track_one: pd.DataFrame, track_two: pd.DataFrame
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
            frame_one = track_one[track_one["frame"] == frame].reset_index()
            frame_two = track_two[track_two["frame"] == frame].reset_index()

            # Colocalize single frames below distance_cutoff
            rows, cols = self.colocalize_frame(frame_one, frame_two)

            # Assign "real" track-particle number
            real_rows = np.array([frame_one.particle[i] for i in rows])
            real_cols = np.array([frame_two.particle[i] for i in cols])

            # Update assignment matrix, IndexError if none colocalizing / empty
            try:
                assignments[real_rows, real_cols] += 1
            except IndexError:
                pass

        # Get colocalizing track numbers from assignment matrix
        coloc_one, coloc_two = np.where(
            np.where(assignments, assignments > SpotsColocalization().min_frames, 0)
        )

        return coloc_one, coloc_two
