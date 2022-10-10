"""Luigi configuration."""

import luigi


class General(luigi.Config):
    image_dir = luigi.Parameter(
        description="Path to raw input image (nd/stk or czi) files.", default="./"
    )
    analysis_dir = luigi.Parameter(
        description="Path where analysis results are saved (created if not given).",
        default="./",
    )
    file_ext = luigi.Parameter(
        description="File extension of raw files (nd or czi).", default="nd"
    )
    do_3D = luigi.BoolParameter(description="Do 3D analysis?", default=False)
    do_TimeSeries = luigi.BoolParameter(
        description="Do TimeSeries analysis? Ignored if do_3D is set to True.",
        default=False,
    )
    gpu_index = luigi.IntParameter(
        description=(
            "Index of GPU to use (nvidia-smi). "
            "If set to -1 no GPU will be used. "
            "Only accelerates cellpose, deep segmentation, and deepBlink."
        ),
        default=-1,
    )


class PreprocessingAlignment(luigi.Config):
    enabled = luigi.BoolParameter(description="Align images?", default=False)
    alignment_dir = luigi.Parameter(
        description="Path to bead alignment files.", default="./"
    )
    method = luigi.Parameter(
        description="Strategy for image registration [options: pystackreg, deepblink].",
        default="deepblink",
    )
    model = luigi.Parameter(
        description="Path to deepBlink model used for alignment.", default="./model.h5"
    )
    channel_reference = luigi.IntParameter(
        description="Channel index for reference.", default=0
    )
    channel_alignment = luigi.IntParameter(
        description="Channel index for alignment.", default=1
    )


class PreprocessingNormalization(luigi.Config):
    registration = luigi.Parameter(
        description=(
            "Registration method. "
            "Ignored if do_3D or do_TimeSeries is set to True. "
            "[options: max, mean, sharpest]"
        ),
        default="max",
    )
    frame_start = luigi.IntParameter(
        description="Frame to start analysis. Only if do_3D or do_TimeSeries is True.",
        default=0,
    )
    frame_end = luigi.IntParameter(
        description="Frame to end analysis. Only if do_3D or do_TimeSeries is True.",
        default=0,
    )
    crop_start = luigi.IntParameter(
        description="Position to start crop x and y.", default=0
    )
    crop_end = luigi.IntParameter(
        description="Position to end crop x and y.", default=0
    )
    bin_axes = luigi.ListParameter(
        description="Mapping of axes to bin-scale.", default=[]
    )


class SpotsDetection(luigi.Config):
    channels = luigi.ListParameter(
        description="List of channel indices to detect spots.", default=[]
    )
    models = luigi.ListParameter(
        description=(
            "List of models to use for spot detection. "
            "Will use the same order as the channels provided above. "
            'Must be passed using quotes (["", ...]) '
        ),
        default=[],
    )
    refinement_radius = luigi.IntParameter(
        description="Radius around spots for characterization.", default=3
    )


class SpotsTracking(luigi.Config):
    gap_frames = luigi.IntParameter(
        description=(
            "Maximum number of frames to skip in tracks. "
            "Set to 0 if do_3D is set to True."
        ),
        default=3,
    )
    min_length = luigi.IntParameter(
        description=(
            "Minimum track length. "
            "If do_3D is set to True, shorter tracks will be removed!"
        ),
        default=5,
    )
    search_range = luigi.IntParameter(
        description=("Pixel search range between spots in tracks/stacks."),
        default=5,
    )
    subtract_drift = luigi.BoolParameter(
        description=(
            "If ensemble drift xy(t) should be subtracted. "
            "Only if do_TimeSeries is set to True."
        ),
        default=False,
    )


class SpotsColocalization(luigi.Config):
    enabled = luigi.BoolParameter(
        description="Do colocalization analysis?", default=False
    )
    channels = luigi.ListParameter(
        description=(
            "List of channel index-pairs for colocalization. "
            "In format of ([reference, secondary]). "
            "Must contain values from channels in SpotsDetection."
        ),
        default=[[]],
    )
    z_distance = luigi.FloatParameter(
        description=(
            "Relative z-distance given x/y-distances are set to 1. "
            "Only if do_3D is set to True."
        ),
        default=1,
    )
    distance_cutoff = luigi.IntParameter(
        description="Maximum distance for colocalization.", default=5
    )
    min_frames = luigi.IntParameter(
        description="Minimum number of frames for colocalization.", default=3
    )


class SegmentationCells(luigi.Config):
    # Basics
    selection = luigi.Parameter(
        description=(
            "Which option for cellular segmentation should be done. "
            "[options: nuclei, cyto, both]"
        ),
        default="both",
    )
    method_nuclei = luigi.Parameter(
        description="Method for nuclear segmentation. [options: cellpose, otsu]",
        default="cellpose",
    )
    method_cyto = luigi.Parameter(
        description=(
            "Method for cytoplasmic segmentation. "
            "Will only be used for the cytoplasmic part of selection `both`. "
            "[options: otsu, li, triangle]"
        ),
        default="otsu",
    )
    channel_nuclei = luigi.IntParameter(
        description="Channel index (0-indexed).", default=0
    )
    channel_cyto = luigi.IntParameter(
        description="Channel index (0-indexed).", default=0
    )

    # Cellpose options
    cellpose_models = luigi.ListParameter(
        description="Paths to custom cellpose models.", default=[]
    )
    cellpose_diameter = luigi.IntParameter(
        description="Expected cellular diameter. Only if method is cellpose.",
        default=150,
    )
    cellpose_resample = luigi.BoolParameter(
        description=(
            "If segmap should be resampled (slower, more accurate). "
            "Only if method is cellpose."
        ),
        default=True,
    )

    # Mathematical options
    gaussian = luigi.FloatParameter(
        description=(
            "Sigma for gaussian filter before thresholding. "
            "Only if method nuclei is otsu."
        ),
        default=3,
    )
    upper_clip = luigi.FloatParameter(
        description="Upper clip limit before thresholding to remove effect of outliers.",
        default=0.95,
    )

    # Subsequent options
    min_size_nuclei = luigi.IntParameter(
        description=(
            "Minimum object size - to filter out possible debris. "
            "Only if method is otsu."
        ),
        default=5000,
    )
    min_size_cyto = luigi.IntParameter(
        description=(
            "Minimum object size - to filter out possible debris. "
            "Only if method is otsu."
        ),
        default=5000,
    )
    min_distance = luigi.IntParameter(
        description=(
            "Minimum radial distance to separate potentially merged nuclei (between centers). "
            "Only if method is otsu."
        ),
        default=50,
    )
    remove_border = luigi.BoolParameter(
        description="Should segmentation maps touching the border be removed?",
        default=True,
    )
    border_distance = luigi.BoolParameter(
        description="Add a column where the distance of each spot to the perifery is measured.",
        default=False,
    )


class SegmentationOther(luigi.Config):
    enabled = luigi.BoolParameter(
        description="Enable other channel segmentation?", default=False
    )
    channels = luigi.ListParameter(description="List of channel indices.", default=[])
    methods = luigi.ListParameter(
        description=(
            'List of methods. Must be passed using quotes (["", ...]). '
            "[options: deep, otsu, li, multiotsu]."
        ),
        default=[],
    )
    models = luigi.ListParameter(
        description=(
            "List of models. "
            "Must match the order of the methods above. "
            "Set to None if using a mixture. "
            "Only if method is median."
        ),
        default=[],
    )
    backbones = luigi.ListParameter(
        description="List of segmentation_model backbones.", default=[]
    )


class FlyBrainCells(luigi.Config):
    enabled = luigi.BoolParameter(
        description="If you're Jess to segment beautiful brains.",
        default=False,
    )
    channel = luigi.IntParameter(description="Index for nucleus channel.", default=0)
    batch_size = luigi.IntParameter(
        description="Cellpose model batch size (default 8 too large).", default=4
    )
    min_intensity = luigi.IntParameter(
        description="Minimum mean pixel intensity of nuclei to not get filtered.",
        default=0,
    )
    min_area = luigi.IntParameter(
        description="Minimum area in pixels of nuclei (anything below filtered).",
        default=200,
    )
    max_area = luigi.IntParameter(
        description="Maximum area in pixels of nuclei (anything above filtered).",
        default=8_000,
    )
    dilation = luigi.IntParameter(
        description="Dilation radius for cell segmentation.", default=3
    )
