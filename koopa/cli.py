"""Command line interface."""

import argparse
import io
import os
import sys

import luigi

sys.stdout = io.StringIO()
from .config import General
from .merge import Merge
from .setup import SetupPipeline

sys.stdout = sys.__stdout__


def _parse_args():
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


def main():
    """Run koopa tasks."""
    args = _parse_args()
    if args.create_config:
        return 1

    luigi.configuration.add_config_path(args.config_file)
    os.remove(os.path.join(General().analysis_dir, "luigi.cfg"))
    luigi.build([SetupPipeline(), Merge()], local_scheduler=True)
