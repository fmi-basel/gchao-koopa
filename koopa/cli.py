"""Command line interface."""

import argparse
import configparser
import io
import os
import sys

import luigi

sys.stdout = io.StringIO()
from .config import General
from .config import PreprocessingAlignment
from .config import PreprocessingNormalization
from .config import SegmentationOther
from .config import SegmentationPrimary
from .config import SegmentationSecondary
from .config import SpotsColocalization
from .config import SpotsDetection
from .config import SpotsTracking
from .merge import Merge
from .setup import SetupPipeline

sys.stdout = sys.__stdout__

CONFIGS = [
    General,
    PreprocessingAlignment,
    PreprocessingNormalization,
    SpotsDetection,
    SpotsTracking,
    SpotsColocalization,
    SegmentationPrimary,
    SegmentationSecondary,
    SegmentationOther,
]


def _parse_args():
    """Basic argument parser."""
    parser = argparse.ArgumentParser(prog="Koopa", description="")
    parser.add_argument(
        "--create-config",
        action="store_true",
        help="Create empty configuration file to be passed to `--config`",
    )
    parser.add_argument("--config", type=str, help="Path to luigi configuration file.")
    parser.add_argument(
        "--workers", type=int, help="Number of parallel luigi workers to spawn."
    )
    args = parser.parse_args()
    return args


def create_config():
    """Create a configuration file with default values."""
    config = configparser.ConfigParser(allow_no_value=True)

    for section in CONFIGS:
        config.add_section(section.__name__)
        for name, param in section().get_params():
            config.set(section.__name__, f"# {name}", param.description)
            config.set(
                section.__name__,
                name,
                "" if param._default is None else str(param._default),
            )

    with open("luigi.cfg", "w") as f:
        config.write(f)


def run_pipeline(config_file):
    """Run standard."""
    luigi.configuration.add_config_path(config_file)
    old_config_file = os.path.join(General().analysis_dir, "luigi.cfg")
    if os.path.exists(old_config_file):
        os.remove(old_config_file)
    luigi.build([SetupPipeline(config_file=config_file), Merge()], local_scheduler=True)


def main():
    """Run koopa tasks."""
    args = _parse_args()
    if args.create_config:
        create_config()
        return 1

    run_pipeline(args.config_file)
