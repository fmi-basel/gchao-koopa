from pathlib import Path
import configparser
import datetime
import os
import subprocess

from tqdm import tqdm
import pandas as pd
import luigi
import numpy as np
import skimage.io

from config import CustomConfig
from detect import Detect
from preprocess import Preprocess
from segment_cells import SegmentCells


class Version(luigi.Task):
    """Version analysis workflow to ensure reproducible results."""

    def output(self):
        return luigi.LocalTarget(os.path.join(CustomConfig().analysis_dir, "luigi.cfg"))

    @staticmethod
    def get_git_hash():
        return (
            subprocess.check_output(["git", "rev-parse", "HEAD"])
            .decode("ascii")
            .strip()
        )

    @staticmethod
    def get_timestamp():
        return str(datetime.datetime.now().timestamp())

    def run(self):
        config = configparser.ConfigParser()
        config.read("./luigi.cfg")
        config["Versioning"] = {
            "timestamp": self.get_timestamp(),
            "githash": self.get_git_hash(),
        }
        with open(self.output().path, "w") as configfile:
            config.write(configfile)


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
        files_nd = sorted(Path(CustomConfig().image_dir).rglob("*.nd"))
        files = [os.path.basename(f).replace(".nd", "") for f in files_nd]
        return files

    def requires(self):
        requiredInputs = []
        for i in self.file_list:
            requiredInputs.append(SegmentCells(FileID=i))
            for c in CustomConfig().channel_spots:
                requiredInputs.append(Detect(FileID=i, SpotChannel=c))
        requiredInputs.append(Version())
        return requiredInputs

    def output(self):
        return luigi.LocalTarget(
            os.path.join(CustomConfig().analysis_dir, "summary.csv")
        )

    def read_files(self, file_id):
        """Read all analysis files associated with one file id."""
        # Segmentation
        fname_segmap = os.path.join(
            CustomConfig().analysis_dir, "segmentation_cells", f"{file_id}.tif"
        )
        segmap = skimage.io.imread(fname_segmap)
        segmap_nucleus = segmap[0]
        segmap_cells = segmap[1]

        # Spots
        dfs = []
        for c in CustomConfig().channel_spots:
            fname_spots = os.path.join(
                CustomConfig().analysis_dir, f"detection_c{c}", f"{file_id}.parq"
            )
            dfs.append(pd.read_parquet(fname_spots))
        df = pd.concat(dfs)
        return df, segmap_nucleus, segmap_cells

    @staticmethod
    def get_value(arr: np.ndarray, row: pd.Series) -> float:
        return arr[int(row["y"]), int(row["x"])]

    def run(self):
        """Merge all analysis files into a single summary file."""
        dfs = []
        for file_id in tqdm(self.file_list):
            df, segmap_nucleus, segmap_cells = self.read_files(file_id)
            df["cell"] = df.apply(lambda row: self.get_value(segmap_cells, row), axis=1)
            df["nucleus"] = df.apply(
                lambda row: self.get_value(segmap_nucleus, row), axis=1
            )
            df["cell_count"] = len(np.unique(segmap_nucleus)) - 1
            df.insert(loc=0, column="file_id", value=file_id)
            dfs.append(df)

        df = pd.concat(dfs)
        df.to_csv(self.output().path)
