"""Prepare pipeline."""

import configparser
import datetime
import os
import subprocess

import luigi

from . import __version__
from .config import General
from .config import SegmentationOther
from .config import SegmentationSecondary
from .config import SpotsColocalization
from .config import SpotsDetection


class SetupPipeline(luigi.Task):
    """Version analysis workflow to ensure reproducible results."""

    config_file = luigi.Parameter(default="./koopa.cfg")

    def output(self):
        return luigi.LocalTarget(os.path.join(General().analysis_dir, "koopa.cfg"))

    def run(self):
        self.create_directories()
        config = configparser.ConfigParser()
        config.read(self.config_file)
        config["Versioning"] = {
            "timestamp": self.timestamp,
            "version": self.version,
            # "githash": self.git_hash,
        }
        with open(self.output().path, "w") as configfile:
            config.write(configfile)

    @staticmethod
    def create_directories():
        """Create all analysis directories for a given path."""
        dirs = ["preprocessed", "segmentation_primary"]

        # Segmentation
        if SegmentationSecondary().enabled:
            dirs.append("segmentation_secondary")
        if SegmentationOther().enabled:
            segmentation_channels = SegmentationOther().channels
            dirs.extend([f"segmentation_c{i}" for i in segmentation_channels])

        # Spot detection
        spot_channels = SpotsDetection().channels
        dirs.extend([f"detection_raw_c{i}" for i in spot_channels])
        if General().do_3D or General().do_TimeSeries:
            dirs.extend([f"detection_final_c{i}" for i in spot_channels])

        # Colocalization
        if SpotsColocalization().enabled:
            for channel_pair in SpotsColocalization().channels:
                for c in channel_pair:
                    if c not in SpotsDetection().channels:
                        raise ValueError(
                            f'Colocalization channel "{c}" '
                            "not listed in detection channels."
                        )
            dirs.extend(
                [f"colocalization_{i}-{j}" for i, j in SpotsColocalization().channels]
            )

        for folder in dirs:
            path = os.path.join(General().analysis_dir, folder)
            os.makedirs(path, exist_ok=True)

    @property
    def git_hash(self):
        """Find current githash as version proxy."""
        return (
            subprocess.check_output(["git", "rev-parse", "HEAD"])
            .decode("ascii")
            .strip()
        )

    @property
    def version(self):
        """Koopa version number."""
        return __version__

    @property
    def timestamp(self):
        """Current unix timestamp."""
        return str(datetime.datetime.now().timestamp())
