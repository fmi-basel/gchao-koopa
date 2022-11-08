from typing import List
import os

from prefect import task
import koopa
import pandas as pd


@task
def merge_single(fname: str, path: os.PathLike, config: dict):
    # Input
    fname_segmaps = {
        "nuclei": os.path.join(path, "segmentation_nuclei", f"{fname}.tif"),
        "cyto": os.path.join(path, "segmentation_cyto", f"{fname}.tif"),
        **{
            f"other_c{i}": os.path.join(path, f"segmentation_c{i}", f"{fname}.tif")
            for i in config["sego_channels"]
        },
    }
    segmaps = {k: koopa.io.load_image(v) for k, v in fname_segmaps.items()}
    df = koopa.util.get_final_spot_file(fname, path, config)

    # Run
    df = koopa.postprocess.add_segmentation_data(df, segmaps, config)

    # Return
    return df


@task
def merge_all(path: os.PathLike, dfs: List[pd.DataFrame]):
    """Merge all analysis files into a single summary file."""
    # with multiprocessing.Pool(self.threads) as pool:
    #     dfs = pool.map(self.merge_file, self.file_list)
    df = pd.concat(dfs, ignore_index=True)
    fname_out = os.path.join(path, "summary.csv")
    koopa.io.save_csv(fname_out, df)
