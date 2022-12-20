"""Utilities for koopa flows."""

import glob
import os

import pandas as pd
import tensorflow as tf
import torch

from . import io


def get_file_list(path: str, file_ext: str):
    """All file basenames in the image directory."""
    files = sorted(
        glob.glob(
            os.path.join(path, "**", f"*.{file_ext}"),
            recursive=True,
        )
    )
    files = [os.path.basename(f).replace(f".{file_ext}", "") for f in files]
    if len(files) != len(set(files)):
        raise ValueError("Found duplicate file names in image directory.")
    if not len(files):
        raise ValueError(
            f"No files found in directory `{path}` " f"with extension `{file_ext}`!"
        )
    return files


def get_final_spot_file(fname: str, path: os.PathLike, config: dict) -> os.PathLike:
    """Read the last important spot files dependent on selected config."""
    if config["coloc_enabled"]:
        fnames = [
            os.path.join(path, f"colocalization_{i}-{j}", f"{fname}.parq")
            for i, j in config["coloc_channels"]
        ]
    elif config["do_3d"] or config["do_timeseries"]:
        fnames = [
            os.path.join(path, f"detection_final_c{i}", f"{fname}.parq")
            for i in config["detect_channels"]
        ]
    else:
        fnames = [
            os.path.join(path, f"detection_raw_c{i}", f"{fname}.parq")
            for i in config["detect_channels"]
        ]

    dfs = [io.load_parquet(f) for f in fnames]
    return pd.concat(dfs, ignore_index=True)


def configure_gpu(gpu_index: int, memory_limit: int = 8192):
    """Set environment variables for GPU usage."""
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_index) if gpu_index else "None"
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
