import luigi


class CustomConfig(luigi.Config):
    image_dir = luigi.Parameter(description="Directory with raw nd/stk files.")
    analysis_dir = luigi.Parameter(description="Directory to save analysis results.")
    alignment_dir = luigi.Parameter(description="Directory with bead alignment files.")

    model_ms2 = luigi.Parameter(description="Path to deepblink model for MS2.")
    model_suntag = luigi.Parameter(description="Path to deepblink model for SunTag.")

    channel_reference = luigi.Parameter(description="Channel identifier for reference.")
    channel_alignment = luigi.Parameter(description="Channel identifier for alignment.")
    channel_ms2 = luigi.IntParameter(
        description=(
            "Channel number (0-indexed on alphabetical channel name) for MS2."
            " For example, 0: w1conf561 and 1: w2conf488."
        )
    )
    channel_suntag = luigi.IntParameter(
        description="Channel number (0-indexed) for SunTag."
    )

    frame_start = luigi.IntParameter(default=1, description="Frame to start analysis.")
    frame_end = luigi.IntParameter(default=15, description="Frame to end analysis.")
    cytoplasm_diameter = luigi.IntParameter(
        default=150, description="Cytoplasm diameter for cellpose."
    )
    cytoplasm_resample = luigi.BoolParameter(
        default=False,
        description="If cytoplasm should be resampled (slower, more accurate).",
    )
    cytoplasm_minsize = luigi.IntParameter(
        default=15, description="Minimum cytoplasm size for cellpose."
    )
    intensity_radius = luigi.IntParameter(
        default=5, description="Radius for intensity normalization."
    )
    track_gap_frames = luigi.IntParameter(
        default=2, description="Maximum number of frames to skip in tracks."
    )
    track_min_length = luigi.IntParameter(
        default=5, description="Minimum track length."
    )
    track_search_range = luigi.IntParameter(
        default=5, description="Pixel search range between spots in tracks."
    )
    distance_cutoff = luigi.IntParameter(
        default=5, description="Distance cutoff for colocalization."
    )
    min_coloc_frames = luigi.IntParameter(
        default=5, description="Minimum number of frames for colocalization."
    )
