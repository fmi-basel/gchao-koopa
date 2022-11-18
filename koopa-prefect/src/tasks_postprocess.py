from typing import List
import os

from prefect import task, get_run_logger
import koopa
import pandas as pd


@task(name="Merge Output (Single)")
def merge_single(fname: str, path: os.PathLike, config: dict):
    logger = get_run_logger()

    # Input
    fname_segmaps = {
        f"other_c{i}": os.path.join(path, f"segmentation_c{i}", f"{fname}.tif")
        for i in config["sego_channels"]
        if config["sego_enabled"]
    }
    if config["brains_enabled"] or config["selection"] in ("both", "nuclei"):
        fname_segmaps["nuclei"] = os.path.join(
            path, "segmentation_nuclei", f"{fname}.tif"
        )
    if not config["brains_enabled"] and config["selection"] in ("both", "cyto"):
        fname_segmaps["cyto"] = os.path.join(path, "segmentation_cyto", f"{fname}.tif")

    segmaps = {k: koopa.io.load_image(v) for k, v in fname_segmaps.items()}
    df = koopa.util.get_final_spot_file(fname, path, config)

    # Run
    df = koopa.postprocess.add_segmentation_data(df, segmaps, config)
    logger.debug(f"Merged files for {fname}")

    # Return
    return df


@task(name="Merge Output (All)")
def merge_all(path: os.PathLike, dfs: List[pd.DataFrame]):
    """Merge all analysis files into a single summary file."""
    df = pd.concat(dfs, ignore_index=True)
    fname_out = os.path.join(path, "summary.csv")
    koopa.io.save_csv(fname_out, df)
