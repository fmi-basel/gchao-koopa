"""Configuration options and validation."""

from dataclasses import dataclass
from typing import Any, List, Tuple, Union
import configparser
import datetime
import os
import textwrap

__version__ = "0.0.9"


@dataclass
class ConfigItem:
    description: str
    default: Any
    dtype: Any


general = {
    "input_path": ConfigItem(
        description=(
            "Path to raw input image (nd/stk or czi) files. "
            "Paths should be given absolutely to improve safety. "
            "Make sure to use the server based path style (/tungstenfs/...)."
        ),
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
            "[options: maximum, mean, sharpest]"
        ),
        default="maximum",
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
        description="Mapping of axes to bin-scale.", default=[], dtype=List[float]
    ),
}

spots_detection = {
    "detect_channels": ConfigItem(
        description="List of channel indices to detect spots.",
        default=[],
        dtype=List[int],
    ),
    "detect_models": ConfigItem(
        description=(
            "List of models to use for spot detection. "
            "Will use the same order as the channels provided above. "
            'Must be passed using quotes (["", ...]). '
            "There should be one model for every channel listed in `detect_channels`."
        ),
        default=[],
        dtype=List[os.PathLike],
    ),
    "refinement_radius": ConfigItem(
        description="Radius around spots for characterization.", default=3, dtype=int
    ),
}

spots_tracking = {
    "gap_frames": ConfigItem(
        description=(
            "Maximum number of frames to skip in tracks. "
            "Set to 0 if do_3d is set to True!"
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
        description=(
            "Pixel search range between spots in tracks/stacks. "
            "Calculated by euclidean distance."
        ),
        default=5,
        dtype=int,
    ),
    "subtract_drift": ConfigItem(
        description=(
            "If ensemble drift xy(t) should be subtracted. "
            "Only available if do_timeseries is set to True."
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
            "In format of [[reference, transform], ...]. "
            "Must contain values from channels in SpotsDetection."
        ),
        default=[()],
        dtype=List[Tuple[int, int]],
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
        default="otsu",
        dtype=str,
    ),
    "method_cyto": ConfigItem(
        description=(
            "Method for cytoplasmic segmentation. "
            "Will only be used for the cytoplasmic part of selection `both`. "
            "[options: otsu, li, triangle]"
        ),
        default="triangle",
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
        description="Paths to custom cellpose models.",
        default=[],
        dtype=List[os.PathLike],
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
            "Don't set too large if medium-sized blobs should be kept. "
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
        description=(
            "Should segmentation maps touching the border be removed? "
            "Currently only implemented if selection is both!"
        ),
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
        description="List of channel indices.", default=[], dtype=List[int]
    ),
    "sego_methods": ConfigItem(
        description=(
            'List of methods. Must be passed using quotes (["", ...]). '
            "[options: deep, otsu, li, multiotsu]."
        ),
        default=[],
        dtype=List[str],
    ),
    "sego_models": ConfigItem(
        description=(
            "List of models. "
            "Must match the order of the methods above. "
            "Set to None if using a mixture. "
            "Only if method is deep."
        ),
        default=[],
        dtype=List[Union[os.PathLike, None]],
    ),
    "sego_backbones": ConfigItem(
        description="List of segmentation_model backbones. Only if method is deep.",
        default=[],
        dtype=List[Union[str, None]],
    ),
}

fly_brain_cells = {
    "brains_enabled": ConfigItem(
        description="Drosophila fly brain segmentation.",
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


def create_multiline_description(description: str) -> list:
    """Create a list of lines to break up an otherwise long string."""
    lines = textwrap.wrap(description, width=85, break_on_hyphens=True)
    return [f"# {line}" for line in lines]


def create_default_config() -> configparser.ConfigParser:
    """Create template configuration with helptext and default arguments."""
    parser = configparser.ConfigParser(allow_no_value=True)

    # Sections
    for name, config in CONFIGS.items():
        parser.add_section(name)
        # Items
        for param, item in config.items():
            for desc in create_multiline_description(item.description):
                parser.set(name, desc)
            parser.set(name, param, str(item.default))
    return parser


def __validate_pathlike(value: os.PathLike, name: str) -> None:
    """Validate path inputs."""
    if not os.path.exists(value):
        raise ValueError(f'Path "{name}" must exist, "{value}" does not.')
    # pardir = os.path.abspath(os.path.join(value, os.pardir))


def __validate_primitive(value: str, dtype: type, error_msg: str) -> None:
    """Validate primitive datatypes."""
    try:
        value = eval(value)
    except NameError as err:
        raise ValueError(error_msg) from err
    # If integers are passed to float params
    if dtype == float and isinstance(value, int):
        return
    if not isinstance(value, dtype):
        raise ValueError(error_msg)


def __validate_list(value: str, name: str, dtype: type) -> None:
    """Validate lists and nested lists."""
    try:
        value = eval(value)
    except NameError as err:
        raise ValueError(
            f"Argument {name} not parseable. Please provide a valid list."
        ) from err

    if dtype == os.PathLike:
        (__validate_pathlike(x, f"{name}-item") for x in value)
        return None

    if dtype == Tuple[int, int]:
        if not all(len(x) == 2 for x in value):
            raise ValueError(f"Items in {name} must have length 2.")
        if not all(isinstance(x1, int) & isinstance(x2, int) for x1, x2 in value):
            raise ValueError(f"List items in {name} must be integers.")
        return None

    if dtype == Union[os.PathLike, None]:
        if not all(x is None or __validate_pathlike(x, name) for x in value):
            raise ValueError(f"List items in {name} must be valid paths or None.")

    if dtype == Union[str, None]:
        if not all(x is None or isinstance(x, str) for x in value):
            raise ValueError(f"List items in {name} must be strings or None.")

    if not all(isinstance(i, dtype) for i in value):
        raise ValueError(f"All items in list {name} must be of type {dtype}.")


def __validate_type(value: str, name: str, dtype: type) -> None:
    """Validate all parameter types."""
    # Primitives
    if dtype == bool:
        error_msg = f'Argument {name} not parseable. Must be "True" or "False".'
        __validate_primitive(value, bool, error_msg)
    if dtype == int:
        error_msg = f"Argument {name} not parseable. Must be an integer."
        __validate_primitive(value, int, error_msg)
    if dtype == float:
        error_msg = f"Argument {name} not parseable. Must be a floating point value."
        __validate_primitive(value, float, error_msg)

    # Composites
    if dtype == os.PathLike:
        __validate_pathlike(value, name)
    if dtype == List[os.PathLike]:
        __validate_list(value, name, os.PathLike)


def __validate_detection(config: configparser.ConfigParser) -> None:
    """Special validation for spot based configurations."""
    detect_channels = eval(config["SpotsDetection"]["detect_channels"])
    detect_models = eval(config["SpotsDetection"]["detect_models"])

    if not detect_channels:
        raise ValueError("Must pass at least one channel for detection.")
    if len(detect_channels) != len(detect_models):
        raise ValueError(
            "All channels for detection must have one and only one model associated."
        )
    if eval(config["SpotsColocalization"]["coloc_enabled"]):
        coloc_channels = eval(config["SpotsColocalization"]["coloc_channels"])
        coloc_channels = [channel for pair in coloc_channels for channel in pair]
        if not all(c in detect_channels for c in coloc_channels):
            raise ValueError(
                "All channels used in coloc_channels must be passed to detect_channels too."
            )


# TODO add check if models and backbones match up?
def __validate_sego(config: configparser.ConfigParser) -> None:
    """Special validation for SegmentationOther."""
    if not eval(config["SegmentationOther"]["sego_enabled"]):
        return None

    sego_channels = eval(config["SegmentationOther"]["sego_channels"])
    sego_methods = eval(config["SegmentationOther"]["sego_methods"])

    if len(sego_channels) != len(sego_methods):
        raise ValueError(
            "All channels for segmentation must have one and only one method associated."
        )


def validate_config(config: configparser.ConfigParser) -> None:
    """Extensive validation of configuration file to verify format and arguments."""
    # Check sections
    if not all(c in config for c in CONFIGS.keys()):
        raise ValueError("Not all sections found in config.")

    for section in CONFIGS.keys():
        for name, value in config[section].items():
            # Skip section if not enabled
            # TODO? Will deem any non False values ok to run
            if "enabled" in name and eval(value) is False:
                break

            # Check params
            try:
                source = CONFIGS[section][name]
            except KeyError as err:
                raise ValueError(
                    f"Found unknown configuration option. {section}>{name} unknown."
                ) from err

            # Check dtypes
            __validate_type(value, name, source.dtype)

    # Check special
    __validate_detection(config)
    __validate_sego(config)
    return True


def add_versioning(config: configparser.ConfigParser) -> configparser.ConfigParser:
    """Add timestamp and version data to configuration."""
    config["Versioning"] = {
        "timestamp": str(datetime.datetime.now().timestamp()),
        "version": __version__,
    }
    return config


def flatten_config(config: configparser.ConfigParser) -> dict:
    """Convert sectioned configuration to flat dictionary."""
    flat_config = {}
    for section in config.sections():
        for key in config[section]:
            # TODO proper type casting
            try:
                flat_config[key] = eval(config[section][key])
            except (NameError, SyntaxError):
                flat_config[key] = config[section][key]
    return flat_config
