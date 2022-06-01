import glob
import multiprocessing
import os

from tqdm import tqdm
import luigi
import numpy as np
import pandas as pd
import tifffile

from config import General
from config import SegmentationOther
from config import SegmentationSecondary
from config import SpotsColocalization
from config import SpotsDetection

from colocalize import ColocalizeFrame
from colocalize import ColocalizeTrack
from detect import Detect
from segment_cells import SegmentPrimary
from segment_cells import SegmentSecondary
from segment_other import SegmentOther
from track import Track


class Merge(luigi.Task):
    """Task to merge workflow tasks into a summary and initiate paralellization."""

    force = luigi.BoolParameter(significant=False, default=False)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.force is True:
            outputs = luigi.task.flatten(self.output())
            for out in outputs:
                if out.exists():
                    os.remove(self.output().path)

    @property
    def file_list(self):
        files = sorted(
            glob.glob(
                os.path.join(General().image_dir, "**", f"*.{General().file_ext}"),
                recursive=True,
            )
        )
        files = [
            os.path.basename(f).replace(f".{General().file_ext}", "") for f in files
        ]
        if len(files) != len(set(files)):
            raise ValueError("Found duplicate file names in image directory.")
        return files

    def requires(self):
        requiredInputs = {}
        for fname in self.file_list:
            required = {}

            # Segmentation Primary/Secondary
            required["primary"] = SegmentPrimary(FileID=fname)
            if SegmentationSecondary().enabled:
                required["secondary"] = SegmentSecondary(FileID=fname)

            # Segmentation Other
            if SegmentationOther().enabled:
                for idx, _ in enumerate(SegmentationOther().channels):
                    required[f"other_{idx}"] = SegmentOther(
                        FileID=fname, ChannelIndex=idx
                    )

            # Spots detection
            for idx, _ in enumerate(SpotsDetection().channels):
                required[f"detect_{idx}"] = Detect(FileID=fname, ChannelIndex=idx)
                if General().do_3D or General().do_TimeSeries:
                    required[f"track_{idx}"] = Track(FileID=fname, ChannelIndex=idx)

            # Colocalization
            if SpotsColocalization().enabled:
                for idx, _ in enumerate(SpotsColocalization().channels):
                    required[f"colocalize_{idx}"] = (
                        ColocalizeTrack(FileID=fname, ChannelPairIndex=idx)
                        if General().do_TimeSeries
                        else ColocalizeFrame(FileID=fname, ChannelPairIndex=idx)
                    )

            requiredInputs[fname] = required
        return requiredInputs

    def output(self):
        return luigi.LocalTarget(os.path.join(General().analysis_dir, "summary.csv"))

    def run(self):
        """Merge all analysis files into a single summary file."""
        # TODO take luigi workers into account (and potential memory overflows)
        with multiprocessing.Pool(multiprocessing.cpu_count() // 2) as pool:
            dfs = pool.map(self.merge_file, self.file_list)
        df = pd.concat(dfs, ignore_index=True)
        df.to_csv(self.output().path, index=False)

    def read_spots_file(self, file_id: str) -> pd.DataFrame:
        if SpotsColocalization().enabled:
            dfs = [
                pd.read_parquet(
                    self.requires()[file_id][f"colocalize_{idx}"].output().path
                )
                for idx, _ in enumerate(SpotsColocalization().channels)
            ]
        elif General().do_3D or General().do_TimeSeries:
            dfs = [
                pd.read_parquet(self.requires()[file_id][f"track_{idx}"].output().path)
                for idx, _ in enumerate(SpotsDetection().channels)
            ]
        else:
            dfs = [
                pd.read_parquet(self.requires()[file_id][f"detect_{idx}"].output().path)
                for idx, _ in enumerate(SpotsDetection().channels)
            ]
        return pd.concat(dfs, ignore_index=True)

    def read_image_file(self, file_id: str, name: str) -> np.ndarray:
        return tifffile.imread(self.requires()[file_id][name].output().path)

    def get_value(self, row: pd.Series, image: np.ndarray) -> int:
        if image.ndim == 3:
            return image[
                int(row["frame"]),
                min(int(row["y"]), image.shape[1] - 1),
                min(int(row["x"]), image.shape[2] - 1),
            ]
        if image.ndim == 2:
            return image[
                min(int(row["y"]), image.shape[0] - 1),
                min(int(row["x"]), image.shape[1] - 1),
            ]
        raise ValueError(f"Segmentation image must be 2D or 3D. Got {image.ndim}D.")

    def merge_file(self, file_id: str):
        df = self.read_spots_file(file_id)

        # Primary segmentation
        primary = self.read_image_file(file_id, "primary")
        df["primary"] = df.apply(lambda row: self.get_value(row, primary), axis=1)
        df["primary_count"] = len(np.unique(primary)) - 1

        # Secondary segmentation
        if SegmentationSecondary().enabled:
            secondary = self.read_image_file(file_id, "secondary")
            df["secondary"] = df.apply(
                lambda row: self.get_value(row, secondary),
                axis=1,
            )

        # Other segmentation
        if SegmentationOther().enabled:
            for idx, _ in SegmentationOther().channels:
                other = self.read_image_file(file_id, f"other_{idx}")
                df[f"other_{idx}"] = df.apply(
                    lambda row: self.get_value(row, other),
                    axis=1,
                ).astype(bool)

        return df
