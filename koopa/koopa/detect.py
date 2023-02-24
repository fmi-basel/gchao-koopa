"""Raw spot detection."""

import deepblink as pink
import numpy as np
import pandas as pd
import trackpy as tp

tp.quiet()


def detect_frame(
    image: np.ndarray,
    model: pink.models.Model,
    refinement_radius: int,
    engine="numba",
) -> pd.DataFrame:
    """Detect spots in a single frame using deepBlink."""
    # Padding to allow for refinement at edges
    image = np.pad(image, refinement_radius + 1, mode="reflect")

    # Prediction and refinement
    yx = pink.inference.predict(image=image, model=model)
    yx = np.delete(
        yx,
        np.where((yx[:,0] >= image.shape[0] - (refinement_radius + 1)) |
                 (yx[:,1] >= image.shape[1] - (refinement_radius + 1)) |
                 (yx[:,0] < (refinement_radius + 1)) |
                 (yx[:,1] < (refinement_radius + 1))
                 ),
        axis=0,
    )
    y, x = yx.T
    df = tp.refine_com(
        raw_image=image,
        image=image,
        radius=refinement_radius,
        coords=yx,
        engine=engine,
    )
    df["x"] = x - refinement_radius - 1
    df["y"] = y - refinement_radius - 1
    df = df.rename({"ecc": "eccentricity"}, axis=1)
    df = df.drop("raw_mass", axis=1)
    return df


def detect_image(
    image: np.ndarray,
    index_channel: int,
    model: pink.models.Model,
    refinement_radius: int,
    engine:str = "numba",
) -> pd.DataFrame:
    """Wrapper to detect spots in an image series (single, z, or t)."""
    if image.ndim < 3:
        raise ValueError(
            f"Image does not have enough dimensions (3+). Got {image.ndim}."
        )

    image = image[index_channel]
    if image.ndim == 2:
        image = np.expand_dims(image, axis=0)
    if image.ndim != 3:
        raise ValueError(
            f"Image must have 3 dimensions for detection. Got {image.ndim}."
        )

    frames = []
    for frame, image_curr in enumerate(image):
        df = detect_frame(image_curr, model, refinement_radius, engine=engine)
        df["frame"] = frame
        df["channel"] = index_channel
        frames.append(df)

    df = pd.concat(frames, ignore_index=True)
    return df
