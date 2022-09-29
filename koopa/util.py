from typing import List
import glob
import configparser
import logging
import os

import luigi
import tensorflow as tf
import torch

from .colocalize import ColocalizeFrame
from .colocalize import ColocalizeTrack
from .config import FlyBrainCells
from .config import General
from .config import PreprocessingAlignment
from .config import PreprocessingNormalization
from .config import SegmentationCells
from .config import SegmentationOther
from .config import SpotsColocalization
from .config import SpotsDetection
from .config import SpotsTracking
from .detect import Detect
from .merge import Merge
from .preprocess import Preprocess
from .registration import ReferenceAlignment
from .segment_cells import SegmentCells
from .segment_cells_flies import DilateCells
from .segment_cells_flies import SegmentCellsMerge
from .segment_cells_flies import SegmentCellsPredict
from .segment_other import SegmentOther
from .track import Track


CONFIGS = [
    General,
    PreprocessingAlignment,
    PreprocessingNormalization,
    SpotsDetection,
    SpotsTracking,
    SpotsColocalization,
    SegmentationCells,
    SegmentationOther,
    FlyBrainCells,
]

TASKS = [
    ColocalizeFrame,
    ColocalizeTrack,
    Detect,
    DilateCells,
    Preprocess,
    ReferenceAlignment,
    SegmentCells,
    SegmentCellsMerge,
    SegmentCellsPredict,
    SegmentOther,
    Track,
]

TASK_NAMES = [task.__name__ for task in TASKS]


def get_file_list():
    """All file basenames in the image directory."""
    files = sorted(
        glob.glob(
            os.path.join(General().image_dir, "**", f"*.{General().file_ext}"),
            recursive=True,
        )
    )
    files = [os.path.basename(f).replace(f".{General().file_ext}", "") for f in files]
    if len(files) != len(set(files)):
        raise ValueError("Found duplicate file names in image directory.")
    if not len(files):
        raise ValueError(
            f"No files found in directory `{General().image_dir}` "
            f"with extension `{General().file_ext}`!"
        )
    return files


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


def remove_all_handlers():
    """Remove all logging StreamHandlers."""
    loggers = [logging.getLogger()]
    loggers = loggers + [
        logging.getLogger(name) for name in logging.root.manager.loggerDict
    ]

    for logger in loggers:
        for handler in logger.handlers:
            if isinstance(handler, logging.StreamHandler):
                logger.removeHandler(handler)


def set_logging():
    """Prepare verbose logging for luigi and silence others."""
    remove_all_handlers()

    formatter = logging.Formatter(
        "%(asctime)s,%(msecs)d %(levelname)s (%(name)s) - %(message)s",
        datefmt="%H:%M:%S",
    )
    file_handler = logging.FileHandler("koopa.log", mode="a")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)

    for log_name in [
        "cellpose",
        "cellpose.core",
        "luigi",
        "luigi-interface",
        "luigi.scheduler",
    ]:
        logger = logging.getLogger(log_name)
        logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.INFO)
    stream_handler.setFormatter(formatter)
    logger = logging.getLogger("koopa")
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)


def delete_task_outputs(tasks: List[luigi.task.Task]):
    """Remove all output files to allow rerunning."""
    output_files = []
    for task in tasks:
        output = task.output()
        if isinstance(output, list):
            output_files.extend([o.path for o in output])
        elif isinstance(output, dict):
            output_files.extend([o.path for o in output.values()])
        elif isinstance(output, luigi.LocalTarget):
            output_files.append(output.path)
        else:
            raise ValueError(f"Output of undefined format - {type(output)}")

    for file_name in output_files:
        if os.path.exists(file_name):
            os.remove(file_name)


def create_task_list(threads: int, task_name: str = None) -> List[luigi.task.Task]:
    """Create a list of all tasks to run."""
    if task_name is None or task_name == "Merge":
        return [Merge(threads=threads)]

    if task_name not in TASK_NAMES:
        raise ValueError(f"Task name invalid. Must be one of {TASK_NAMES}")

    single_task = TASKS[TASK_NAMES.index(task_name)]

    # File independent task(s)
    if task_name in ("ReferenceAlignment"):
        return [single_task()]

    # File dependent tasks
    tasks = []
    for fname in get_file_list():
        if task_name in ("SegmentationOther", "Detect", "Track"):
            enumerator = (
                SegmentationOther().channels
                if task_name == "SegmentationOther"
                else SpotsDetection().channels
            )
            tasks.extend(
                [
                    single_task(FileID=fname, ChannelIndex=idx)
                    for idx, _ in enumerate(enumerator)
                ]
            )
        elif task_name in ("ColocalizeTrack", "ColocalizeFrame"):
            tasks.extend(
                [
                    single_task(FileID=fname, ChannelPairIndex=idx)
                    for idx, _ in enumerate(SpotsColocalization().channels)
                ]
            )
        else:
            tasks.append(single_task(FileID=fname))
    return tasks


class DisableLogger:
    def __init__(self, silent: bool):
        self.silent = silent

    def __enter__(self):
        if self.silent:
            logging.disable(logging.CRITICAL)

    def __exit__(self, exit_type, exit_value, exit_traceback):
        if self.silent:
            logging.disable(logging.NOTSET)


def configure_gpu(gpu_index: int, memory_limit: int = 8192):
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_index if gpu_index else "None"
    os.environ["OMP_NUM_THREADS"] = "4"
    os.environ["OPENBLAS_NUM_THREADS"] = "4"
    os.environ["MKL_NUM_THREADS"] = "4"
    os.environ["VECLIB_MAXIMUM_THREADS"] = "4"
    os.environ["NUMEXPR_NUM_THREADS"] = "4"
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

    # Tensorflow
    tf.config.threading.set_intra_op_parallelism_threads(4)
    tf.config.threading.set_inter_op_parallelism_threads(4)
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        tf.config.set_logical_device_configuration(
            gpus[0], [tf.config.LogicalDeviceConfiguration(memory_limit)]
        )

    # Pytorch
    torch.set_num_threads(4)
