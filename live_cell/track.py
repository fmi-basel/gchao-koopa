import os

import luigi
import numpy as np
import pandas as pd
import skimage.io
import trackpy as tp

from config import CustomConfig
from detect import Detect
from segment import Segment

tp.quiet()


class Track(luigi.Task):

    FileID = luigi.Parameter()
    Channel = luigi.Parameter()

    def requires(self):
        return [
            Detect(FileID=self.FileID, Channel=self.Channel),
            Segment(FileID=self.FileID),
        ]

    def output(self):
        return luigi.LocalTarget(
            os.path.join(
                CustomConfig().analysis_dir,
                f"track_{self.Channel}",
                f"{self.FileID}.parq",
            )
        )

    def run(self):
        self.segmap = skimage.io.imread(self.requires()[1].output().path)
        df = pd.read_parquet(self.requires()[0].output().path)
        df = self.track(df)
        df.to_parquet(self.output().path)

    def assign_cell(self, df: pd.DataFrame) -> pd.DataFrame:
        """Assign the cell index based on the segmap to x,y columns in df."""

        def __get_value(arr, y, x):
            return arr[
                min(int(round(y)), arr.shape[0] - 1),
                min(int(round(x)), arr.shape[1] - 1),
            ]

        df["cell"] = df.apply(
            lambda row: __get_value(self.segmap, row["y"], row["x"]), axis=1
        )
        df["cell_count"] = len(np.unique(self.segmap)) - 1
        return df

    def track(self, df: pd.DataFrame):
        """Nearest neighbor based tracking.

        Tracks particles, removes background tracks, removes
            too short tracks, and subtrackts global drift.
        """
        track = tp.link_df(
            df,
            search_range=CustomConfig().track_search_range,
            memory=CustomConfig().track_gap_frames,
        )
        track = self.assign_cell(track)
        track = track[track["cell"] != 0]
        track = tp.filter_stubs(track, threshold=CustomConfig().track_min_length)

        drift = tp.compute_drift(track)
        trackd = tp.subtract_drift(track.copy(), drift)
        trackd["particle"] = pd.factorize(trackd["particle"])[0]
        trackd = trackd.reset_index(drop=True)
        return trackd
