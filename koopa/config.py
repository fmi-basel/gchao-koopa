"""Luigi configuration."""

import luigi


class General(luigi.Config):
    image_dir = luigi.Parameter(
        description="Path to raw input image (nd/stk, czi) files.", default="./"
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
    remove_n_frames = luigi.IntParameter(
        description=(
            "Number of frames to remove from top/bottom of stack. "
            "Only if do_3D is True."
        ),
        default=0,
    )
    frame_start = luigi.IntParameter(
        description="Frame to start analysis. Only if do_TimeSeries is True.",
        default=1,
    )
    frame_end = luigi.IntParameter(
        description="Frame to end analysis. Only if do_TimeSeries is True.", default=16
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
            "Only if do_TimeSeries is True."
        ),
        default=3,
    )
    min_length = luigi.IntParameter(
        description="Minimum track length. Only if do_TimeSeries is True.", default=5
    )
    search_range = luigi.IntParameter(
        description=(
            "Pixel search range between spots in tracks. "
            "Only if do_TimeSeries is True."
        ),
        default=5,
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
    distance_cutoff = luigi.IntParameter(
        description="Maximum distance for colocalization.", default=5
    )
    min_frames = luigi.IntParameter(
        description="Minimum number of frames for colocalization.", default=3
    )


class SegmentationPrimary(luigi.Config):
    channel = luigi.IntParameter(description="Channel index (0-indexed).", default=0)
    method = luigi.Parameter(
        description=(
            "Method for primary segmentation (otsu only valid for nuclei). "
            "[options: cellpose, otsu]"
        ),
        default="cellpose",
    )
    model = luigi.Parameter(
        description=(
            "Model for segmenation. Only if method is cellpose. "
            "[options: cyto, nuclei]"
        ),
        default="nuclei",
    )
    diameter = luigi.IntParameter(
        description="Expected cellular diameter. Only if method is cellpose.",
        default=150,
    )
    resample = luigi.BoolParameter(
        description=(
            "If segmap should be resampled (slower, more accurate). "
            "Only if method is cellpose."
        ),
        default=True,
    )
    gaussian = luigi.FloatParameter(
        description="Sigma for gaussian filter before thresholding. Only if method is otsu.",
        default=3,
    )
    min_size = luigi.IntParameter(
        description=(
            "Minimum object size - to filter out possible debris. "
            "Only if method is otsu."
        ),
        default=5000,
    )
    min_distance = luigi.IntParameter(
        description=(
            "Minimum radial distance to separate potentially merged nuclei (between centers)."
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


class SegmentationSecondary(luigi.Config):
    enabled = luigi.BoolParameter(
        description="Enable secondary segmentation?", default=False
    )
    channel = luigi.IntParameter(description="Channel index.", default=0)
    method = luigi.Parameter(
        description="Method for secondary segmentation [options: otsu, li, median, triangle].",
        default="otsu",
    )
    value = luigi.FloatParameter(
        description="Value for secondary segmentation. Only if method is median.",
        default=0.5,
    )
    upper_clip = luigi.FloatParameter(
        description="Upper percentile for clipping image.", default=0.95
    )
    gaussian = luigi.FloatParameter(
        description="Sigma for gaussian filter before thresholding.", default=5
    )
    min_size = luigi.IntParameter(
        description="Minimum object size - to filter out possible debris. ",
        default=5000,
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
            'List of methods. Must be passed using quotes (["", ...]) '
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
