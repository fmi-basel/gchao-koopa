from dataclasses import dataclass
from typing import Any
import configparser
import datetime
import os

__version__ = "0.0.1"


@dataclass
class ConfigItem:
    description: str
    default: Any
    dtype: Any


general = {
    "input_path": ConfigItem(
        description="Path to raw input image (nd/stk or czi) files.",
        default="./",
        dtype=os.PathLike,
    ),
    "output_path": ConfigItem(
        description="Path where analysis results are saved (created if not given).",
        default="./",
        dtype=os.PathLike,
    ),
    "file_ext": ConfigItem(
        description="File extension of raw files (nd or czi).",
        default="nd",
        dtype=str,
    ),
    "do_3d": ConfigItem(description="Do 3D analysis?", default=False, dtype=bool),
    "do_timeseries": ConfigItem(
        description="Do TimeSeries analysis? Ignored if do_3d is set to True.",
        default=False,
        dtype=bool,
    ),
    "gpu_index": ConfigItem(
        description=(
            "Index of GPU to use (nvidia-smi). "
            "If set to -1 no GPU will be used. "
            "Only accelerates cellpose, deep segmentation, and deepBlink."
        ),
        default=-1,
        dtype=int,
    ),
}

preprocessing_alignment = {
    "alignment_enabled": ConfigItem(
        description="Align images?", default=False, dtype=bool
    ),
    "alignment_path": ConfigItem(
        description="Path to bead alignment files.", default="./", dtype=os.PathLike
    ),
    "alignment_method": ConfigItem(
        description="Strategy for image registration [options: pystackreg, deepblink].",
        default="deepblink",
        dtype=str,
    ),
    "alignment_model": ConfigItem(
        description="Path to deepBlink model used for alignment.",
        default="./model.h5",
        dtype=os.PathLike,
    ),
    "channel_reference": ConfigItem(
        description="Channel index for reference channel to stay static.",
        default=0,
        dtype=int,
    ),
    "channel_transform": ConfigItem(
        description="Channel index for alignment channel to be transformed.",
        default=1,
        dtype=int,
    ),
}

preprocessing_normalization = {
    "registration_method": ConfigItem(
        description=(
            "Registration method. "
            "Ignored if do_3d or do_timeseries is set to True. "
            "[options: max, mean, sharpest]"
        ),
        default="max",
        dtype=str,
    ),
    "frame_start": ConfigItem(
        description="Frame to start analysis. Only if do_3d or do_timeseries is True.",
        default=0,
        dtype=int,
    ),
    "frame_end": ConfigItem(
        description="Frame to end analysis. Only if do_3d or do_timeseries is True.",
        default=0,
        dtype=int,
    ),
    "crop_start": ConfigItem(
        description="Position to start crop x and y.", default=0, dtype=int
    ),
    "crop_end": ConfigItem(
        description="Position to end crop x and y.", default=0, dtype=int
    ),
    "bin_axes": ConfigItem(
        description="Mapping of axes to bin-scale.", default=[], dtype=list
    ),
}

spots_detection = {
    "detect_channels": ConfigItem(
        description="List of channel indices to detect spots.", default=[], dtype=list
    ),
    "detect_models": ConfigItem(
        description=(
            "List of models to use for spot detection. "
            "Will use the same order as the channels provided above. "
            'Must be passed using quotes (["", ...]) '
        ),
        default=[],
        dtype=list,
    ),
    "refinement_radius": ConfigItem(
        description="Radius around spots for characterization.", default=3, dtype=int
    ),
}

spots_tracking = {
    "gap_frames": ConfigItem(
        description=(
            "Maximum number of frames to skip in tracks. "
            "Set to 0 if do_3d is set to True."
        ),
        default=3,
        dtype=int,
    ),
    "min_length": ConfigItem(
        description=(
            "Minimum track length. "
            "If do_3d is set to True, shorter tracks will be removed!"
        ),
        default=5,
        dtype=int,
    ),
    "search_range": ConfigItem(
        description=("Pixel search range between spots in tracks/stacks."),
        default=5,
        dtype=int,
    ),
    "subtract_drift": ConfigItem(
        description=(
            "If ensemble drift xy(t) should be subtracted. "
            "Only if do_timeseries is set to True."
        ),
        default=False,
        dtype=bool,
    ),
}

spots_colocalization = {
    "coloc_enabled": ConfigItem(
        description="Do colocalization analysis?", default=False, dtype=bool
    ),
    "coloc_channels": ConfigItem(
        description=(
            "List of channel index-pairs for colocalization. "
            "In format of ([reference, secondary]). "
            "Must contain values from channels in SpotsDetection."
        ),
        default=[[]],
        dtype=list,
    ),
    "z_distance": ConfigItem(
        description=(
            "Relative z-distance given x/y-distances are set to 1. "
            "Only if do_3d is set to True."
        ),
        default=1,
        dtype=float,
    ),
    "distance_cutoff": ConfigItem(
        description="Maximum distance for colocalization.",
        default=5,
        dtype=int,
    ),
    "min_frames": ConfigItem(
        description="Minimum number of frames for colocalization.",
        default=3,
        dtype=int,
    ),
}

segmentation_cells = {
    # Basics
    "selection": ConfigItem(
        description=(
            "Which option for cellular segmentation should be done. "
            "[options: nuclei, cyto, both]"
        ),
        default="both",
        dtype=str,
    ),
    "method_nuclei": ConfigItem(
        description="Method for nuclear segmentation. [options: cellpose, otsu]",
        default="cellpose",
        dtype=str,
    ),
    "method_cyto": ConfigItem(
        description=(
            "Method for cytoplasmic segmentation. "
            "Will only be used for the cytoplasmic part of selection `both`. "
            "[options: otsu, li, triangle]"
        ),
        default="otsu",
        dtype=str,
    ),
    "channel_nuclei": ConfigItem(
        description="Channel index (0-indexed).", default=0, dtype=int
    ),
    "channel_cyto": ConfigItem(
        description="Channel index (0-indexed).", default=0, dtype=int
    ),
    # Cellpose options
    "cellpose_models": ConfigItem(
        description="Paths to custom cellpose models.", default=[], dtype=list
    ),
    "cellpose_diameter": ConfigItem(
        description="Expected cellular diameter. Only if method is cellpose.",
        default=150,
        dtype=int,
    ),
    "cellpose_resample": ConfigItem(
        description=(
            "If segmap should be resampled (slower, more accurate). "
            "Only if method is cellpose."
        ),
        default=True,
        dtype=bool,
    ),
    # Mathematical options
    "gaussian": ConfigItem(
        description=(
            "Sigma for gaussian filter before thresholding. "
            "Only if method nuclei is otsu."
        ),
        default=3,
        dtype=int,
    ),
    "upper_clip": ConfigItem(
        description="Upper clip limit before thresholding to remove effect of outliers.",
        default=0.95,
        dtype=float,
    ),
    # Subsequent options
    "min_size_nuclei": ConfigItem(
        description=(
            "Minimum object size - to filter out possible debris. "
            "Only if method is otsu."
        ),
        default=5000,
        dtype=int,
    ),
    "min_size_cyto": ConfigItem(
        description=(
            "Minimum object size - to filter out possible debris. "
            "Only if method is otsu."
        ),
        default=5000,
        dtype=int,
    ),
    "min_distance": ConfigItem(
        description=(
            "Minimum radial distance to separate potentially merged nuclei (between centers). "
            "Only if method is otsu."
        ),
        default=50,
        dtype=int,
    ),
    "remove_border": ConfigItem(
        description="Should segmentation maps touching the border be removed?",
        default=True,
        dtype=bool,
    ),
    "border_distance": ConfigItem(
        description="Add a column where the distance of each spot to the perifery is measured.",
        default=False,
        dtype=bool,
    ),
}

segmentation_other = {
    "sego_enabled": ConfigItem(
        description="Enable other channel segmentation?", default=False, dtype=bool
    ),
    "sego_channels": ConfigItem(
        description="List of channel indices.", default=[], dtype=list
    ),
    "sego_methods": ConfigItem(
        description=(
            'List of methods. Must be passed using quotes (["", ...]). '
            "[options: deep, otsu, li, multiotsu]."
        ),
        default=[],
        dtype=list,
    ),
    "sego_models": ConfigItem(
        description=(
            "List of models. "
            "Must match the order of the methods above. "
            "Set to None if using a mixture. "
            "Only if method is median."
        ),
        default=[],
        dtype=list,
    ),
    "sego_backbones": ConfigItem(
        description="List of segmentation_model backbones.", default=[], dtype=list
    ),
}

fly_brain_cells = {
    "brains_enabled": ConfigItem(
        description="If you're Jess to segment beautiful brains.",
        default=False,
        dtype=bool,
    ),
    "brains_channel": ConfigItem(
        description="Index for nucleus channel.", default=0, dtype=int
    ),
    "batch_size": ConfigItem(
        description="Cellpose model batch size (default 8 too large).",
        default=4,
        dtype=int,
    ),
    "min_intensity": ConfigItem(
        description="Minimum mean pixel intensity of nuclei to not get filtered.",
        default=0,
        dtype=float,
    ),
    "min_area": ConfigItem(
        description="Minimum area in pixels of nuclei (anything below filtered).",
        default=200,
        dtype=int,
    ),
    "max_area": ConfigItem(
        description="Maximum area in pixels of nuclei (anything above filtered).",
        default=8_000,
        dtype=int,
    ),
    "dilation": ConfigItem(
        description="Dilation radius for cell segmentation.", default=3, dtype=int
    ),
}

CONFIGS = {
    "General": general,
    "PreprocessingAlignment": preprocessing_alignment,
    "PreprocessingNormalization": preprocessing_normalization,
    "SpotsDetection": spots_detection,
    "SpotsTracking": spots_tracking,
    "SpotsColocalization": spots_colocalization,
    "SegmentationCells": segmentation_cells,
    "SegmentationOther": segmentation_other,
    "FlyBrainCells": fly_brain_cells,
}


def create_default_config() -> configparser.ConfigParser:
    parser = configparser.ConfigParser()
    for name, config in CONFIGS.items():
        parser[name] = {key: value.default for key, value in config.items()}
    return parser


def validate_config(config: configparser.ConfigParser) -> bool:
    # sections = config.sections()

    if not all(c in config for c in CONFIGS.keys()):
        raise ValueError("Not all sections found in config.")

    # TODO
    # dtypes
    # paths


def add_versioning(config: configparser.ConfigParser) -> configparser.ConfigParser:
    config["Versioning"] = {
        "timestamp": str(datetime.datetime.now().timestamp()),
        "version": __version__,
    }
    return config


def flatten_config(config: configparser.ConfigParser) -> dict:
    flat_config = {}
    for section in config.sections():
        for key in config[section]:
            # TODO proper type casting
            try:
                flat_config[key] = eval(config[section][key])
            except (NameError, SyntaxError):
                flat_config[key] = config[section][key]
    return flat_config
