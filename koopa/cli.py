"""Command line interface."""

import argparse
import configparser
import io
import logging
import os
import sys

import luigi

sys.stdout = io.StringIO()
from . import __version__
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
    parser = argparse.ArgumentParser(
        prog="Koopa",
        description="Workflow for analysis of cellular microscopy data.",
        add_help=False,
    )
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        required=True,
        help="Path to luigi configuration file.",
    )
    parser.add_argument(
        "-w",
        "--workers",
        type=int,
        default=4,
        help="Number of parallel luigi workers to spawn. [default: 4]",
    )
    parser.add_argument(
        "-cc",
        "--create-config",
        action="store_true",
        help="Create empty configuration file to be passed to `--config`",
    )
    parser.add_argument(
        "-h",
        "--help",
        action="help",
        default=argparse.SUPPRESS,
        help="Show this message.",
    )
    parser.add_argument(
        "-V",
        "--version",
        action="version",
        version="%(prog)s " + str(__version__),
        help="Show %(prog)s's version number.",
    )
    if len(sys.argv) == 1:
        parser.print_help()
        parser.exit(0)
    return parser.parse_args()


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

    with open("koopa.cfg", "w") as f:
        config.write(f)


def set_logging():
    """Prepare verbose logging for luigi and silence others."""
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    logging.basicConfig(filename=None, filemode="a", level=logging.ERROR)

    file_handler = logging.FileHandler("koopa.log")
    file_handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter(
        "%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s", datefmt="%H:%M:%S"
    )
    file_handler.setFormatter(formatter)
    for log_name in [
        "luigi-interface",
        "luigi.scheduler",
        "luigi",
        "cellpose",
        "cellpose.core",
    ]:
        logger = logging.getLogger(log_name)
        logger.addHandler(file_handler)


def run_pipeline(config_file, threads):
    """Run standard."""
    set_logging()

    luigi.configuration.add_config_path(os.path.abspath(config_file))
    old_config_file = os.path.join(General().analysis_dir, "koopa.cfg")
    if os.path.exists(old_config_file):
        os.remove(old_config_file)
    luigi.build(
        [SetupPipeline(config_file=config_file), Merge(threads=threads)],
        local_scheduler=True,
        workers=threads,
    )


def main():
    """Run koopa tasks."""
    args = _parse_args()

    if args.create_config:
        create_config()
        sys.exit(1)

    run_pipeline(args.config, args.workers)
