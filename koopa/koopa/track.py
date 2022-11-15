"""Tracking along time or Z."""

import pandas as pd
import trackpy as tp


tp.quiet()


def track(
    df: pd.DataFrame, search_range: int, gap_frames: int, min_length: int
) -> pd.DataFrame:
    """Nearest neighbor based tracking."""
    track = tp.link_df(
        df,
        search_range=search_range,
        memory=gap_frames,
    )
    track = tp.filter_stubs(track, threshold=min_length)
    return track


def subtract_drift(df: pd.DataFrame) -> pd.DataFrame:
    """Remove global drift from tracks."""
    drift = tp.compute_drift(df)
    df_clean = tp.subtract_drift(df.copy(), drift)
    return df_clean


def clean_particles(df: pd.DataFrame) -> pd.DataFrame:
    """Reset particle indices to ensure sequential numbering."""
    df["particle"] = pd.factorize(df["particle"])[0]
    df = df.reset_index(drop=True)
    return df


def link_brightest_particles(df: pd.DataFrame, track: pd.DataFrame) -> pd.DataFrame:
    """Non-maximum supression to join brightest particles along a track."""
    # Index of brightest particles
    idx = track.groupby(["particle"])["mass"].transform(max) == track["mass"]
    df_nms = track[idx]

    # Remove the non-track particles
    df_without_track = df[
        ~df.set_index(["x", "y", "frame", "mass"]).index.isin(
            track.set_index(["x", "y", "frame", "mass"]).index
        )
    ]

    # Add back nms (brightest spots)
    df_clean = pd.concat([df_nms, df_without_track]).reset_index(drop=True)
    return df_clean
