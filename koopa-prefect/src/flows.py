import argparse
import os

from prefect import flow, get_run_logger, unmapped
import koopa

import tasks_postprocess
import tasks_preprocess
import tasks_segment
import tasks_spots


# @flow(name="Single Image", task_runner=ConcurrentTaskRunner)
def workflow_single(fname: str, config: dict):
    logger = get_run_logger()
    logger.info(f"Started running {fname}")

    # Config
    path_output = config["output_path"]
    do_3d = config["do_3d"]
    do_timeseries = config["do_timeseries"]
    args = dict(fname=fname, path=path_output, config=config)
    unmapped_args = {k: unmapped(v) for k, v in args.items()}
    final_spots = None
    seg_other = []

    # Preprocess
    preprocess = tasks_preprocess.preprocess.submit(**args)

    # Segmentation cells
    if config["brains_enabled"]:
        brain_1 = tasks_segment.segment_cells_predict.submit(
            **args, wait_for=[preprocess]
        )
        brain_2 = tasks_segment.segment_cells_merge.submit(**args, wait_for=[brain_1])
        seg_cells = tasks_segment.dilate_cells.submit(**args, wait_for=[brain_2])
    else:
        seg_cells = tasks_segment.segment_cells.submit(**args, wait_for=[preprocess])

    # Segmentation Other
    if config["sego_enabled"]:
        seg_other = tasks_segment.segment_other.map(
            **unmapped_args,
            index_list=list(range(len(config["sego_channels"]))),
            wait_for=[preprocess],
        )

    # Spot detection
    spots = tasks_spots.detect.map(
        **unmapped_args,
        index_list=list(range(len(config["detect_channels"]))),
        wait_for=[preprocess],
    )
    if do_3d or do_timeseries:
        spots = tasks_spots.track.map(
            unmapped(**args), index_channel=config["detect_channels"], wait_for=spots
        )
    final_spots = spots

    # Colocalization
    if config["coloc_enabled"]:
        reference = [i[0] for i in config["coloc_channels"]]
        transform = [i[1] for i in config["coloc_channels"]]
        if do_timeseries:
            coloc = tasks_spots.colocalize_track.submit(
                **unmapped_args,
                index_reference=reference,
                index_transform=transform,
                wait_for=spots,
            )
        else:
            coloc = tasks_spots.colocalize_frame.submit(
                **unmapped_args,
                index_reference=reference,
                index_transform=transform,
                wait_for=spots,
            )
        final_spots = coloc

    return tasks_postprocess.merge_single.submit(
        **args, wait_for=[*final_spots, seg_cells, *seg_other]
    ).wait()


@flow(name="Koopa", version=koopa.__version__)
def workflow(fname: str):
    """Core koopa workflow."""
    cfg = koopa.io.load_config(fname)
    koopa.config.validate_config(cfg)
    config = koopa.config.flatten_config(cfg)
    path_output = config["output_path"]

    # Save config
    cfg = koopa.config.add_versioning(cfg)
    fname_config = os.path.join(path_output, "koopa.cfg")
    koopa.io.save_config(fname_config, cfg)

    # File independent tasks
    if config["alignment_enabled"]:
        tasks_preprocess.align.submit(
            path_in=config["alignment_path"],
            path_out=path_output,
            config=config,
        ).wait()

    # Workflow
    fnames = koopa.util.get_file_list(config["input_path"], config["file_ext"])
    dfs = [workflow_single(fname, config) for fname in fnames]
    tasks_postprocess.merge_all(path_output, dfs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str)
    args = parser.parse_args()
    workflow(args.config)
