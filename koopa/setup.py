import configparser
import datetime
import os
import subprocess

import luigi

from config import General
from config import SegmentationOther
from config import SegmentationSecondary
from config import SpotsDetection
from config import SpotsColocalization


class SetupPipeline(luigi.Task):
    """Version analysis workflow to ensure reproducible results."""

    def output(self):
        return luigi.LocalTarget(os.path.join(General().analysis_dir, "luigi.cfg"))

    def run(self):
        self.create_directories()
        config = configparser.ConfigParser()
        # TODO change to dynamic path based on input
        config.read("./luigi.cfg")
        config["Versioning"] = {
            "timestamp": self.get_timestamp(),
            "githash": self.get_git_hash(),
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
                    if not c in SpotsDetection().channels:
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

	# TODO change to version number instead of hash
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
