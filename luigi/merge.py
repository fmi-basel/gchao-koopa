from pathlib import Path
import os

from tqdm import tqdm
import pandas as pd
import luigi
import numpy as np
import skimage.io

from config import globalConfig
from detect import Detect
from preprocess import Preprocess
from segment_cells import SegmentCells


class Merge(luigi.Task):
    """Task to merge workflow tasks into a summary and initiate paralellization."""

    @property
    def file_list(self):
        files_nd = sorted(Path(globalConfig().ImageDir).rglob("*.nd"))
        files = [os.path.basename(f).replace(".nd", "") for f in files_nd]
        return files

    def requires(self):
        requiredInputs = []
        for i in self.file_list:
            requiredInputs.append(Preprocess(FileID=i))
            requiredInputs.append(SegmentCells(FileID=i))
            for c in globalConfig().ChannelSpots:
                requiredInputs.append(Detect(FileID=i, SpotChannel=c))
        return requiredInputs

    def output(self):
        return luigi.LocalTarget(
            os.path.join(globalConfig().AnalysisDir, "summary.csv")
        )

    def read_files(self, file_id):
        """Read all analysis files associated with one file id."""
        # Segmentation
        fname_segmap = os.path.join(
            globalConfig().AnalysisDir, "segmentation_cells", f"{file_id}.tif"
        )
        segmap = skimage.io.imread(fname_segmap)
        segmap_nucleus = segmap[0]
        segmap_cells = segmap[1]

        # Spots
        dfs = []
        for c in globalConfig().ChannelSpots:
            fname_spots = os.path.join(
                globalConfig().AnalysisDir, f"detection_c{c}", f"{file_id}.parq"
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
