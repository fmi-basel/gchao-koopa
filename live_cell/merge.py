import configparser
import datetime
import glob
import os
import subprocess

from tqdm import tqdm
import luigi
import pandas as pd

from config import CustomConfig
from colocalize import Colocalize


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
        files_nd = sorted(
            glob.glob(os.path.join(CustomConfig().image_dir, "*.nd"), recursive=True)
        )
        files = [os.path.basename(f).replace(".nd", "") for f in files_nd]
        return files

    def requires(self):
        requiredInputs = [Version()]
        for i in self.file_list:
            requiredInputs.append(Colocalize(FileID=i))
        return requiredInputs

    def output(self):
        return luigi.LocalTarget(
            os.path.join(CustomConfig().analysis_dir, "summary.csv")
        )

    def run(self):
        """Merge all analysis files into a single summary file."""
        dfs = [
            pd.read_parquet(f.output().path)
            for f in tqdm(self.requires(), desc="Reading parquet files")
            if f.output().path.endswith(".parq")
        ]
        df = pd.concat(dfs, ignore_index=True)
        df.to_csv(self.output().path)
