"""Luigi configuration."""

import luigi


class General(luigi.Config):
    image_dir = luigi.Parameter(
        description="Directory with raw nd/stk files.", default="./"
    )
    analysis_dir = luigi.Parameter(
        description="Directory to save analysis results.", default="./"
    )
    file_ext = luigi.Parameter(
        description="File extension of raw files (nd or czi).", default="nd"
    )
    do_3D = luigi.BoolParameter(description="Do 3D analysis.", default=False)
    do_TimeSeries = luigi.BoolParameter(
        description="Do TimeSeries analysis. Ignored if do_3D is set to True",
        default=False,
    )


class PreprocessingAlignment(luigi.Config):
    enabled = luigi.BoolParameter(description="Align images.", default=False)
    alignment_dir = luigi.Parameter(
        description="Directory with bead alignment files.", default="./"
    )
    method = luigi.Parameter(
        description="Strategy for image registration [options: pystackreg, deepblink].",
        default="deepblink",
    )
    model = luigi.Parameter(
        description="deepBlink model to use for alignment.", default="./model.h5"
    )
    channel_reference = luigi.IntParameter(
        description="Channel index for reference.", default=0
    )
    channel_alignment = luigi.IntParameter(
        description="Channel index for alignment.", default=1
    )


class PreprocessingNormalization(luigi.Config):
    registration = luigi.Parameter(
        description="Registration method. "
        "Ignored if do_3D or do_TimeSeries is set to True. "
        "[options: max, mean, sharpes]",
        default="max",
    )
    remove_n_frames = luigi.IntParameter(
        description="Number of frames to remove from top/bottom of stack.", default=0
    )
    frame_start = luigi.IntParameter(description="Frame to start analysis.", default=1)
    frame_end = luigi.IntParameter(description="Frame to end analysis.", default=16)


class SpotsDetection(luigi.Config):
    channels = luigi.ListParameter(
        description="List of channel indices (0-indexed) to detect spots.", default=[]
    )
    models = luigi.ListParameter(
        description="List of models to use for spot detection. "
        "Will use the same order as the channels provided above.",
        default=[],
    )
    refinement_radius = luigi.IntParameter(
        description="Radius for intensity normalization.", default=3
    )


class SpotsTracking(luigi.Config):
    gap_frames = luigi.IntParameter(
        description="Maximum number of frames to skip in tracks.", default=3
    )
    min_length = luigi.IntParameter(description="Minimum track length.", default=5)
    search_range = luigi.IntParameter(
        description="Pixel search range between spots in tracks.", default=5
    )


class SpotsColocalization(luigi.Config):
    enabled = luigi.BoolParameter(
        description="Do colocalization analysis.", default=False
    )
    channels = luigi.ListParameter(
        description="List of channel index-pairs (0-indexed) for colocalization. "
        "In format of ([reference, secondary]). "
        "Must contain channels from SpotsDetection.",
        default=[[]],
    )
    distance_cutoff = luigi.IntParameter(
        description="Distance cutoff for colocalization.", default=5
    )
    min_frames = luigi.IntParameter(
        description="Minimum number of frames for colocalization.", default=3
    )


class SegmentationPrimary(luigi.Config):
    channel = luigi.IntParameter(description="Channel index (0-indexed).", default=0)
    model = luigi.Parameter(
        description="Cellpose model for segmenation. [options: cyto, nuclei]",
        default="nuclei",
    )
    diameter = luigi.IntParameter(description="Diameter for cellpose.", default=150)
    resample = luigi.BoolParameter(
        description="If segmap should be resampled (slower, more accurate).",
        default=True,
    )
    min_size = luigi.IntParameter(
        description="Minimum cytoplasm size for cellpose.", default=15
    )
    border_distance = luigi.BoolParameter(
        description="Add a column where the distance of each spot to the perifery is measured.",
        default=False,
    )


class SegmentationSecondary(luigi.Config):
    enabled = luigi.BoolParameter(
        description="Enable secondary segmentation.", default=False
    )
    channel = luigi.IntParameter(description="Channel index (0-indexed).", default=0)
    method = luigi.Parameter(
        description="Method for secondary segmentation. [options: otsu, li, median]",
        default="otsu",
    )
    value = luigi.FloatParameter(
        description="Value for secondary segmentation.", default=0.5
    )
    upper_clip = luigi.FloatParameter(
        description="Upper percentile for clipping image.", default=0.95
    )
    gaussian = luigi.FloatParameter(
        description="Sigma for gaussian filter before thresholding.", default=5
    )
    border_distance = luigi.BoolParameter(
        description="Add a column where the distance of each spot to the perifery is measured.",
        default=False,
    )


class SegmentationOther(luigi.Config):
    enabled = luigi.BoolParameter(
        description="Enable other segmentation.", default=False
    )
    channels = luigi.ListParameter(
        description="List of channel indices (0-indexed).", default=[]
    )
    methods = luigi.ListParameter(
        description="List of methods [options: deep, otsu, li, multiotsu].", default=[]
    )
    models = luigi.ListParameter(
        description="List of models. "
        "Must match the order of the methods above. "
        "Ignored if method is not deep. "
        "Set to None if using a mixture.",
        default=[],
    )
    backbones = luigi.ListParameter(description="List of model backbones.", default=[])
