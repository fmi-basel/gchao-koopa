import os

from prefect import task
import koopa


@task(name="Segment Cells (Single)")
def segment_cells_single(fname: str, path: os.PathLike, config: dict):
    # Config
    selection = config["selection"]
    fname_image = os.path.join(path, "preprocessed", f"{fname}.tif")
    fname_out = os.path.join(path, f"segmentation_{selection}", f"{fname}.tif")
    if not config["force"] and os.path.exists(fname_out):
        return

    # Input
    image = koopa.io.load_image(fname_image)
    image = image[config[f"channel_{selection}"]]
    if not config["do_3d"] and config["do_timeseries"]:
        image = koopa.segment_cells.preprocess(image)

    # Run
    if selection == "nuclei":
        segmap = koopa.segment_cells.segment_nuclei(image, config)
    else:
        segmap = koopa.segment_cells.segment_cyto(image, config)

    # Save
    koopa.io.save_image(fname_out, segmap)


@task(name="Segment Cells (Both)")
def segment_cells_both(fname: str, path: os.PathLike, config: dict):
    # Config
    fname_image = os.path.join(path, "preprocessed", f"{fname}.tif")
    fname_nuclei = os.path.join(path, "segmentation_nuclei", f"{fname}.tif")
    fname_cyto = os.path.join(path, "segmentation_cyto", f"{fname}.tif")
    if (
        not config["force"]
        and os.path.exists(fname_nuclei)
        and os.path.exists(fname_cyto)
    ):
        return

    # Input
    image = koopa.io.load_image(fname_image)
    image_nuclei = image[config["channel_nuclei"]]
    image_cyto = image[config["channel_cyto"]]
    if not config["do_3d"] and config["do_timeseries"]:
        image_nuclei = koopa.segment_cells.preprocess(image_nuclei)
        image_cyto = koopa.segment_cells.preprocess(image_cyto)

    # Run
    segmap_nuclei, segmap_cyto = koopa.segment_cells.segment_both(
        image_nuclei, image_cyto, config
    )

    # Save
    koopa.io.save_image(fname_cyto, segmap_cyto)
    koopa.io.save_image(fname_nuclei, segmap_nuclei)


@task(name="Segment Cells (Predict)", tags=["GPU"])
def segment_cells_predict(fname: str, path: os.PathLike, config: dict):
    # Config
    fname_image = os.path.join(path, "preprocessed", f"{fname}.tif")
    fname_out = os.path.join(path, "segmentation_nuclei_prediction", f"{fname}.tif")
    if not config["force"] and os.path.exists(fname_out):
        return

    # Input
    image = koopa.io.load_image(fname_image)

    # Run
    image = koopa.segment_flies.normalize_nucleus(image[config["brains_channel"]])
    segmap = koopa.segment_flies.cellpose_predict(image, config["batch_size"])

    # Save
    koopa.io.save_image(fname_out, segmap)


@task(name="Segment Cells (Merge)")
def segment_cells_merge(fname: str, path: os.PathLike, config: dict):
    # Config
    fname_image = os.path.join(path, "preprocessed", f"{fname}.tif")
    fname_pred = os.path.join(path, "segmentation_nuclei_prediction", f"{fname}.tif")
    fname_out = os.path.join(path, "segmentation_nuclei_merge", f"{fname}.tif")
    if not config["force"] and os.path.exists(fname_out):
        return

    # Input
    image = koopa.io.load_image(fname_image)[config["brains_channel"]]
    yf = koopa.io.load_image(fname_pred)

    # Run
    segmap = koopa.segment_flies.merge_masks(yf)
    segmap = koopa.segment_flies.remove_false_objects(
        image,
        segmap,
        min_intensity=config["min_intensity"],
        min_area=config["min_area"],
        max_area=config["max_area"],
    )

    # Save
    koopa.io.save_image(fname_out, segmap)


@task(name="Segment Cells (Dilate)")
def dilate_cells(fname: str, path: os.PathLike, config: dict):
    # Config
    fname_segmap = os.path.join(path, "segmentation_nuclei_merge", f"{fname}.tif")
    fname_out = os.path.join(path, "segmentation_nuclei", f"{fname}.tif")
    if not config["force"] and os.path.exists(fname_out):
        return

    # Input
    segmap = koopa.io.load_image(fname_segmap)

    # Run
    dilated = koopa.segment_flies.dilate_segmap(segmap, dilation=config["dilation"])

    # Save
    koopa.io.save_image(fname_out, dilated)


@task(name="Segment Other", description="")
def segment_other(fname: str, path: os.PathLike, index_list: int, config: dict):
    # Config
    index_channel = config["sego_channels"][index_list]
    fname_image = os.path.join(path, "preprocessed", f"{fname}.tif")
    fname_out = os.path.join(path, f"segmentation_c{index_channel}", f"{fname}.tif")
    if not config["force"] and os.path.exists(fname_out):
        pass

    # Input
    image = koopa.io.load_image(fname_image)

    # Run
    segmap = koopa.segment_other.segment(
        image=image[index_channel], index_list=index_list, config=config
    )

    # Save
    koopa.io.save_image(fname_out, segmap)
