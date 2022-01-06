import luigi


class CustomConfig(luigi.Config):
    image_dir = luigi.Parameter(description="Directory with raw nd/stk files.")
    analysis_dir = luigi.Parameter(description="Directory to save analysis results.")
    model_spots = luigi.Parameter(description="Path to deepblink spot model.")
    z_projection = luigi.Parameter(description="Type of z-projection (max / mean).")
    channel_spots = luigi.ListParameter(description="Indexes for spots channel.")
    channel_background = luigi.IntParameter(
        description="Index for background segmentation channel."
    )
    channel_nucleus = luigi.IntParameter(description="Index for nucleus channel.")
    refinement_radius = luigi.IntParameter(
        description="Radius for spot intensity / size refinement."
    )
    nucleus_diameter = luigi.IntParameter(description="Nuclear diameter for cellpose.")
    nucleus_minsize = luigi.IntParameter(description="Nuclear diameter for cellpose.")
