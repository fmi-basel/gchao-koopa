import luigi


class globalConfig(luigi.Config):
    ImageDir = luigi.Parameter(description="Directory with raw nd/stk files.")
    AnalysisDir = luigi.Parameter(description="Directory to save analysis results.")
    ModelSpots = luigi.Parameter(description="Path to deepblink spot model.")
    ZProjection = luigi.Parameter(description="Type of z-projection (max / mean).")
    ChannelSpots = luigi.ListParameter(description="Indexes for spots channel.")
    ChannelBackground = luigi.IntParameter(
        description="Index for background segmentation channel."
    )
    ChannelNucleus = luigi.IntParameter(description="Index for nucleus channel.")
    RefinementRadius = luigi.IntParameter(
        description="Radius for spot intensity / size refinement."
    )
    NucleusDiameter = luigi.IntParameter(description="Nuclear diameter for cellpose.")
    NucleusMinsize = luigi.IntParameter(description="Nuclear diameter for cellpose.")
