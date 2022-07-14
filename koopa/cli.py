"""Command line interface."""

import os

import luigi

from .config import General
from .merge import Merge
from .setup import SetupPipeline


def _parse_args():
    pass


def main():
    """Run koopa tasks."""
    args = _parse_args()
    if args.create_config:
        return

    luigi.configuration.add_config_path(args.config_file)
    os.remove(os.path.join(General().analysis_dir, "luigi.cfg"))
    luigi.build([SetupPipeline(), Merge()], local_scheduler=True)
