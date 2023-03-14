"""Merge all tasks to summary."""

from typing import Dict, Tuple
import warnings

import numpy as np
import pandas as pd
import scipy.ndimage as ndi
import skimage.measure


def get_value(row: pd.Series, image: np.ndarray) -> float:
    """Get pixel intensity from coordinate values in df row."""
    image = image.squeeze()
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


def merge_segmaps(
    df: pd.DataFrame, segmaps: Dict[str, np.ndarray], fname: str, do_3d: bool = False
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if df["FileID"].nunique() > 1:
        raise ValueError("Spot files corrupted. Can only contain data from one image.")
    if not any(i in segmaps for i in ("nuclei", "cyto")):
        raise ValueError("At least one of 'nuclei' or 'cyto' must be in segmaps")

    # Cell data
    df_cell = pd.DataFrame()
    for select in ("cyto", "nuclei"):
        if select not in segmaps:
            warnings.warn(f"Segmap {select} not found (skipping).", RuntimeWarning)
            continue
        df_props = get_cell_properties(segmaps[select], select, do_3d)
        df_cell = pd.concat([df_cell, df_props], axis=1)
        df_cell = df_cell.loc[:, ~df_cell.columns.duplicated()]
    df_cell.insert(0, "FileID", fname)

    if not len(df):
        return df, df_cell

    # Cellular index
    cell_id_segmap = "cyto" if "cyto" in segmaps else "nuclei"
    df["cell_id"] = df.apply(
        lambda row: get_value(row, segmaps[cell_id_segmap]), axis=1
    )
    if "nuclei" in segmaps:
        df["nuclear"] = df.apply(
            lambda row: get_value(row, segmaps["nuclei"]) != 0, axis=1
        )

    # Other segmentation
    for name, segmap in segmaps.items():
        if "other" in name:
            df[name] = df.apply(lambda row: get_value(row, segmap), axis=1).astype(bool)
    return df, df_cell


def get_segmentation_data(
    df: pd.DataFrame, segmaps: Dict[str, np.ndarray], config: dict
) -> pd.DataFrame:
    """Combine information from segmaps with spots-dataframe."""
    # Config
    selection = "nuclei" if config["brains_enabled"] else config["selection"]
    full_selection = ("cyto", "nuclei") if selection == "both" else (selection,)
    cell_id = "nuclei" if selection == "nuclei" else "cyto"
    if df["FileID"].nunique() != 1:
        raise ValueError("Spot files corrupted. Can only contain data from one image.")

    # Cellular segmentation
    df["cell_id"] = df.apply(lambda row: get_value(row, segmaps[cell_id]), axis=1)

    df_cell = pd.DataFrame()
    for select in full_selection:
        df_props = get_cell_properties(segmaps[select], select, config["do_3d"])
        df_cell = pd.concat([df_cell, df_props], axis=1)
        df_cell = df_cell.loc[:, ~df_cell.columns.duplicated()]
    df_cell.insert(0, "FileID", df["FileID"].unique()[0])

    if selection == "both":
        df["nuclear"] = df.apply(
            lambda row: get_value(row, segmaps["nuclei"]) != 0, axis=1
        )

    # Other segmentation
    for name, segmap in segmaps.items():
        if "other" in name:
            df[name] = df.apply(lambda row: get_value(row, segmap), axis=1).astype(bool)

    # TODO add properly
    # # Border distance
    # if config["border_distance"]:
    #     df[f"distance_from_{selection}"] = get_distance_from_segmap(
    #         df, segmaps[selection]
    #     )
    return df, df_cell
