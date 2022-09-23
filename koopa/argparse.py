import argparse
import sys

from . import __version__


def _add_utils(parser):
    parser.add_argument(
        "-s", "--silent", action="store_true", help="Run koopa silently."
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


def _parse_args():
    """Basic argument parser."""
    parser = argparse.ArgumentParser(
        prog="Koopa",
        description="Workflow for analysis of cellular microscopy data.",
        add_help=False,
    )

    # Basic running
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        help="Path to luigi configuration file.",
    )

    # Single file and task options
    parser.add_argument(
        "-t",
        "--task",
        type=str,
        help="Name of task to run separately (including previous dependencies).",
    )
    parser.add_argument(
        "-f",
        "--force",
        action="store_true",
        help="Forcefully rerun task / workflow.",
    )

    # Configuration
    parser.add_argument(
        "-w",
        "--workers",
        type=int,
        default=4,
        help="Number of parallel luigi workers to spawn. [default: 4]",
    )
    parser.add_argument(
        "-g",
        "--gpu",
        type=int,
        default=None,
        help=(
            "Index of GPU to use for GPU accelerable tasks. "
            "Will overwrite workers to 1!"
        ),
    )
    parser.add_argument(
        "-cc",
        "--create-config",
        action="store_true",
        help="Create empty configuration file to be passed to `--config`",
    )
    _add_utils(parser)
    if len(sys.argv) == 1:
        parser.print_help()
        parser.exit(0)
    return parser.parse_args()
