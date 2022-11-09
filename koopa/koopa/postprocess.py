"""Merge all tasks to summary."""

from typing import Dict

import numpy as np
import pandas as pd
import scipy.ndimage as ndi
import skimage.measure


def get_value(row: pd.Series, image: np.ndarray) -> float:
    """Get pixel intensity from coordinate values in df row."""
    if image.ndim == 3:
        return image[
            int(row["frame"]),
            min(int(row["y"]), image.shape[1] - 1),
            min(int(row["x"]), image.shape[2] - 1),
        ]
    if image.ndim == 2:
        return image[
            min(int(row["y"]), image.shape[0] - 1),
            min(int(row["x"]), image.shape[1] - 1),
        ]
    raise ValueError(f"Segmentation image must be 2D or 3D. Got {image.ndim}D.")


def get_distance_from_segmap(df: pd.DataFrame, segmap: np.ndarray) -> list:
    """Measure the distance of the spot to the periphery of a segmap."""
    distances = []
    for cell_id, dfg in df.groupby("cell_id"):
        mask = segmap == cell_id
        erosion = ndi.binary_erosion(mask)
        arr1 = np.array(np.where(np.bitwise_xor(erosion, mask))).T
        arr2 = dfg[["y", "x"]].values
        rmsd = np.mean(np.sqrt((arr1[None, :] - arr2[:, None]) ** 2), axis=2)
        distances.extend(np.min(rmsd, axis=1))
    return distances


def get_cell_properties(segmap: np.ndarray, name: str, do_3d: bool) -> pd.DataFrame:
    """Find common measurements using regionprops."""
    properties = ["label", "area"]
    if not do_3d:
        properties.append("eccentricity")

    props = skimage.measure.regionprops_table(segmap, properties=properties)
    df = pd.DataFrame(props)
    df.columns = ["cell_id", *(f"{prop}_{name}" for prop in properties[1:])]
    return df


def add_segmentation_data(
    df: pd.DataFrame, segmaps: Dict[str, np.ndarray], config: dict
) -> pd.DataFrame:
    """Combine information from segmaps with spots-dataframe."""
    # Config
    selection = config["selection"]
    cell_id = "nuclei" if selection == "nuclei" else "cyto"
    full_selection = ("cyto", "nuclei") if selection == "both" else (selection)

    # Cellular segmentation
    df["cell_id"] = df.apply(lambda row: get_value(row, segmaps[cell_id]), axis=1)
    df["num_cells"] = len(np.unique(segmaps[cell_id])) - 1

    for select in full_selection:
        df_cell = get_cell_properties(segmaps[select], select, config["do_3d"])
        df = pd.merge(df, df_cell, how="left", on="cell_id").fillna(0)

        # Border distance
        if config["border_distance"]:
            df[f"distance_from_{selection}"] = get_distance_from_segmap(
                df, segmaps[selection]
            )

    if not len(df):
        return df

    if selection == "both":
        df["nuclear"] = df.apply(
            lambda row: get_value(row, segmaps["nuclei"]) != 0, axis=1
        )

    # Other segmentation
    for name, segmap in {k: v for k, v in segmaps.items() if "other" in k}:
        df[name] = df.apply(lambda row: get_value(row, segmap), axis=1).astype(bool)

    return df
