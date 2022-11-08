import os

from prefect import flow, get_run_logger
import koopa

import tasks_postprocess
import tasks_preprocess
import tasks_segment
import tasks_spots


@flow(name="Single runner")
def workflow_single(fname: str, config: dict):
    logger = get_run_logger()
    logger.info(f"Started running {fname}")

    # Config
    path_output = config["output_path"]
    do_3d = config["do_3d"]
    do_timeseries = config["do_timeseries"]

    # Preprocess
    tasks_preprocess.preprocess.submit(fname=fname, path=path_output, config=config)

    # Segmentation cells
    if config["brains_enabled"]:
        pass
        # required["cells"] = DilateCells(FileID=fname)
    else:
        tasks_segment.segment_cells.submit(fname=fname, path=path_output, config=config)

    # Segmentation Other
    if config["sego_enabled"]:
        for idx, _ in enumerate(config["sego_channels"]):
            tasks_segment.segment_other.submit(
                fname=fname, path=path_output, index_list=idx, config=config
            )

    # Spot detection
    for idx, val in enumerate(config["detect_channels"]):
        tasks_spots.detect.submit(
            fname=fname, path=path_output, index_list=idx, config=config
        )
        if do_3d or do_timeseries:
            tasks_spots.track.submit(
                fname=fname, path=path_output, index_channel=val, config=config
            )

    # Colocalization
    if config["coloc_enabled"]:
        for reference, transform in config["coloc_channels"]:
            if do_timeseries:
                tasks_spots.colocalize_track.submit(
                    fname=fname,
                    path=path_output,
                    index_reference=reference,
                    index_transform=transform,
                    config=config,
                )
            else:
                tasks_spots.colocalize_frame.submit(
                    fname=fname,
                    path=path_output,
                    index_reference=reference,
                    index_transform=transform,
                    config=config,
                )

    return tasks_postprocess.merge_single.submit(fname, path_output, config)


@flow
def workflow(fname: str):
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
        tasks_preprocess.align(
            path_in=config["alignment_path"],
            path_out=path_output,
            config=config,
        )

    # Workflow
    fnames = koopa.util.get_file_list(config["input_path"], config["file_ext"])
    dfs = [workflow_single(fname, config) for fname in fnames]
    tasks_postprocess.merge_all(path_output, dfs)


workflow("/tungstenfs/scratch/gchao/eichbast/koopa/test.cfg")
