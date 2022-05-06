import luigi


class General(luigi.Config):
    image_dir = luigi.Parameter(description="Directory with raw nd/stk files.")
    analysis_dir = luigi.Parameter(description="Directory to save analysis results.")
    file_ext = luigi.Parameter(description="File extension of raw files (nd or czi).")
    do_3D = luigi.BoolParameter(description="Do 3D analysis.")
    do_TimeSeries = luigi.BoolParameter(description="Do TimeSeries analysis.")


class PreprocessingAlignment(luigi.Config):
    enabled = luigi.BoolParameter(description="Align images.")
    alignment_dir = luigi.Parameter(description="Directory with bead alignment files.")
    method = luigi.Parameter(
        description="Strategy for image registration (pystackreg or deepblink)."
    )
    model = luigi.Parameter(description="Deepblink model to use for alignment.")
    channel_reference = luigi.IntParameter(description="Channel index for reference.")
    channel_alignment = luigi.IntParameter(description="Channel index for alignment.")


class PreprocessingNormalization(luigi.Config):
    registration = luigi.Parameter(description="Registration method.")
    remove_n_frames = luigi.IntParameter(
        description="Number of frames to remove from top/bottom of stack."
    )
    frame_start = luigi.IntParameter(description="Frame to start analysis.")
    frame_end = luigi.IntParameter(description="Frame to end analysis.")


class SpotsDetection(luigi.Config):
    channels = luigi.ListParameter(
        description="List of channel indices (0-indexed) to detect spots."
    )
    models = luigi.ListParameter(
        description="List of models to use for spot detection."
    )
    refinement_radius = luigi.IntParameter(
        description="Radius for intensity normalization."
    )


class SpotsTracking(luigi.Config):
    gap_frames = luigi.IntParameter(
        description="Maximum number of frames to skip in tracks."
    )
    min_length = luigi.IntParameter(description="Minimum track length.")
    search_range = luigi.IntParameter(
        description="Pixel search range between spots in tracks."
    )


class SpotsColocalization(luigi.Config):
    enabled = luigi.BoolParameter(description="Do colocalization analysis.")
    channels = luigi.ListParameter(
        description="List of channel index-pairs (0-indexed) for colocalization."
    )
    distance_cutoff = luigi.IntParameter(
        description="Distance cutoff for colocalization."
    )
    min_frames = luigi.IntParameter(
        description="Minimum number of frames for colocalization."
    )


class SegmentationPrimary(luigi.Config):
    channel = luigi.IntParameter(description="Channel index (0-indexed).")
    model = luigi.Parameter(
        description="Cellpose model for segmenation (cyto / nuclei)."
    )
    diameter = luigi.IntParameter(description="Diameter for cellpose.")
    resample = luigi.BoolParameter(
        description="If segmap should be resampled (slower, more accurate).",
    )
    min_size = luigi.IntParameter(description="Minimum cytoplasm size for cellpose.")


class SegmentationSecondary(luigi.Config):
    enabled = luigi.BoolParameter(description="Enable secondary segmentation.")
    channel = luigi.IntParameter(description="Channel index (0-indexed).")
    method = luigi.Parameter(description="Method for secondary segmentation.")
    value = luigi.FloatParameter(description="Value for secondary segmentation.")
    upper_clip = luigi.FloatParameter(
        description="Upper percentile for clipping image."
    )
    gaussian = luigi.FloatParameter(
        description="Sigma for gaussian filter before thresholding."
    )


class SegmentationOther(luigi.Config):
    enabled = luigi.BoolParameter(description="Enable other segmentation.")
    channels = luigi.ListParameter(description="List of channel indices (0-indexed).")
    methods = luigi.ListParameter(description="List of methods.")
    models = luigi.ListParameter(description="List of models.")
    backbones = luigi.ListParameter(description="List of model backbones.")
