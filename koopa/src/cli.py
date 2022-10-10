"""Command line interface."""

import io
import os
import sys

import luigi

sys.stdout = io.StringIO()
from . import util
from . import argparse
from .config import General
from .setup import SetupPipeline

sys.stdout = sys.__stdout__


def initialize_setup(config_file: str, silent: bool):
    """Prepare pipeline to be run."""
    # Replace old configuration file and load parameters into memory
    luigi.configuration.add_config_path(os.path.abspath(config_file))
    old_config_file = os.path.join(General().analysis_dir, "koopa.cfg")
    if os.path.exists(old_config_file):
        os.remove(old_config_file)

    if General().gpu_index != -1:
        util.configure_gpu(General().gpu_index)

    with util.DisableLogger(silent):
        luigi.build([SetupPipeline(config_file=config_file)], local_scheduler=True)


# def set_environment_variables():
#     os.environ["CELLPOSE_LOCAL_MODELS_PATH"] = "./"
#     os.environ["KERAS_HOME"] = "./"


def main():
    """Run koopa tasks."""
    args = argparse._parse_args()
    util.set_logging()

    if args.create_config:
        util.create_config()
        sys.exit(1)

    initialize_setup(args.config, args.silent)
    tasks = util.create_task_list(args.workers, args.task)
    if args.force:
        util.delete_task_outputs(tasks)

    with util.DisableLogger(args.silent):
        luigi.build(tasks, local_scheduler=True, workers=args.workers)
