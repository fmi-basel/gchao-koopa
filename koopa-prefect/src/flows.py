from typing import List

from prefect import flow, get_run_logger, unmapped
from prefect_dask.task_runners import DaskTaskRunner
import koopa

import tasks_postprocess
import tasks_preprocess
import tasks_segment
import tasks_spots


# cluster_class="dask_jobqueue.SLURMCluster",
# cluster_kwargs={
#     "n_workers": 1,
#     "account": "ppi",
#     "queue": "cpu_short",
#     "cores": 4,
#     "memory": "16GB",
#     "walltime": "00:30:00",
#     "job_extra_directives": ["--ntasks=1"],
# },
# adapt_kwargs={"minimum": 1, "maximum": 1},


def file_independent(config: dict):
    if not config["alignment_enabled"]:
        return None

    tasks_preprocess.align.submit(
        path_in=config["alignment_path"], path_out=config["output_path"], config=config
    ).wait()


def cell_segmentation(
    fnames: List[str], config: dict, kwargs: dict, dependencies: list
):
    if not config["brains_enabled"]:
        if config["selection"] == "both":
            return tasks_segment.segment_cells_both.map(
                fnames, **kwargs, wait_for=dependencies
            )
        return tasks_segment.segment_cells_single.map(
            fnames, **kwargs, wait_for=dependencies
        )

    brain_1 = tasks_segment.segment_cells_predict.map(
        fnames, **kwargs, wait_for=dependencies
    )
    brain_2 = tasks_segment.segment_cells_merge.map(fnames, **kwargs, wait_for=brain_1)
    return tasks_segment.dilate_cells.map(fnames, **kwargs, wait_for=brain_2)


def other_segmentation(
    fnames: List[str], config: dict, kwargs: dict, dependencies: list
):
    if not config["sego_enabled"]:
        return []

    channels = range(len(config["sego_channels"]))
    fnames_map = [f for f in fnames for _ in channels]
    index_map = [c for _ in fnames for c in channels]

    seg_other = tasks_segment.segment_other.map(
        fnames_map, **kwargs, index_list=index_map, wait_for=dependencies
    )
    return seg_other


def spot_detection(fnames: List[str], config: dict, kwargs: dict, dependencies: list):
    channels = range(len(config["detect_channels"]))
    fnames_map = [f for f in fnames for _ in channels]
    index_map = [c for _ in fnames for c in channels]
    spots = tasks_spots.detect.map(
        fnames_map, **kwargs, index_list=index_map, wait_for=dependencies
    )

    if config["do_3d"] or config["do_timeseries"]:
        fnames_map = [f for f in fnames for _ in config["detect_channels"]]
        index_map = [c for _ in fnames for c in config["detect_channels"]]
        spots = tasks_spots.track.map(
            fnames_map, **kwargs, index_channel=index_map, wait_for=spots
        )
    return spots


def colocalization(fnames: List[str], config: dict, kwargs: dict, dependencies: list):
    if not config["coloc_enabled"]:
        return dependencies

    reference = [i[0] for _ in fnames for i in config["coloc_channels"]]
    transform = [i[1] for _ in fnames for i in config["coloc_channels"]]
    fnames_map = [f for f in fnames for _ in config["coloc_channels"]]

    if config["do_timeseries"]:
        return tasks_spots.colocalize_track.map(
            fnames_map,
            **kwargs,
            index_reference=reference,
            index_transform=transform,
            wait_for=dependencies
        )

    return tasks_spots.colocalize_frame.map(
        fnames_map,
        **kwargs,
        index_reference=reference,
        index_transform=transform,
        wait_for=dependencies
    )


def merging(fnames: List[str], config: dict, kwargs: dict, dependencies: list):
    dfs = tasks_postprocess.merge_single.map(fnames, **kwargs, wait_for=dependencies)
    tasks_postprocess.merge_all.submit(config["output_path"], dfs, wait_for=dfs)


@flow(
    name="Koopa",
    version=koopa.__version__,
    task_runner=DaskTaskRunner,
    persist_result=False,
)
def workflow(config_path: str, force: bool = False):
    """Core koopa workflow.

    Arguments:
        * config_path: Path to koopa configuration file.
            Path must be passed linux-compatible (e.g. /tungstenfs/scratch/...).
            The default configuration file can be viewed and downloaded [here](https://github.com/BBQuercus/koopa/blob/main/koopa-prefect/koopa.cfg).

        * force: If selected, the entire workflow will be re-run.
            Otherwise, only the not yet executed components (missing files) are run.

    All documentation can be found on the koopa wiki (https://github.com/BBQuercus/koopa/wiki).
    """
    logger = get_run_logger()
    logger.info("Started running Koopa!")
    koopa.util.configure_gpu(False)

    # File independent tasks
    config = tasks_preprocess.configuration(config_path)
    file_independent(config)

    # Workflow
    fnames = koopa.util.get_file_list(config["input_path"], config["file_ext"])
    workflow_core(fnames, config)
    logger.info("Koopa finished analyzing everything!")
