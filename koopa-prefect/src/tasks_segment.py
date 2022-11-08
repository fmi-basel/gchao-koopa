import os

from prefect import task
import koopa


@task(name="Segment Cells", description="")
def segment_cells(fname: str, path: os.PathLike, config: dict):
    # Input
    fname_image = os.path.join(path, "preprocessed", f"{fname}.tif")
    image = koopa.io.load_image(fname_image)

    # Run
    image_nuclei = image[config["channel_nuclei"]]
    image_cyto = image[config["channel_cyto"]]
    if not config["do_3d"] and config["do_timeseries"]:
        image_nuclei = koopa.segment_cells.preprocess(image_nuclei)
        image_cyto = koopa.segment_cells.preprocess(image_cyto)

    # Single output
    selection = config["selection"]
    if selection == "nuclei":
        segmap = koopa.segment_cells.segment_nuclei(image_nuclei, config)
    if selection == "cyto":
        segmap = koopa.segment_cells.segment_cyto(image_cyto, config)
    if selection in ("nuclei", "cyto"):
        fname_out = os.path.join(path, f"segmentation_{selection}", f"{fname}.tif")
        koopa.io.save_image(fname_out, segmap)

    # Dual output
    if selection == "both":
        segmap_nuclei, segmap_cyto = koopa.segment_cells.segment_both(
            image_nuclei, image_cyto, config
        )
        fname_cyto = os.path.join(path, "segmentation_cyto", f"{fname}.tif")
        fname_nuclei = os.path.join(path, "segmentation_nuclei", f"{fname}.tif")
        koopa.io.save_image(fname_cyto, segmap_cyto)
        koopa.io.save_image(fname_nuclei, segmap_nuclei)


@task
def segment_cells_predict(fname: str, path: os.PathLike, config: dict):
    # Input
    fname_image = os.path.join(path, "preprocessed", f"{fname}.tif")
    image = koopa.io.load_image(fname_image)

    # Run
    image = koopa.segment_flies.normalize_nucleus(image[config["brains_channel"]])
    segmap = koopa.segment_flies.cellpose_predict(image, config["batch_size"])

    # Save
    fname_out = os.path.join(path, "segmentation_nuclei_prediction", f"{fname}.tif")
    koopa.io.save_image(fname_out, segmap)


@task
def segment_cells_merge(fname: str, path: os.PathLike, config: dict):
    # Input
    fname_image = os.path.join(path, "preprocessed", f"{fname}.tif")
    fname_pred = os.path.join(path, "segmentation_nuclei_prediction", f"{fname}.tif")
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
    fname_out = os.path.join(path, "segmentation_nuclei_merge", f"{fname}.tif")
    koopa.io.save_image(fname_out, segmap)


@task
def dilate_cells(fname: str, path: os.PathLike, config: dict):
    # Input
    fname_segmap = os.path.join(path, "segmentation_nuclei_merge", f"{fname}.tif")
    segmap = koopa.io.load_image(fname_segmap)

    # Run
    dilated = koopa.segment_flies.dilate_segmap(segmap, dilation=config["dilation"])

    # Save
    fname_out = os.path.join(path, "segmentation_nuclei", f"{fname}.tif")
    koopa.io.save_image(fname_out, dilated)


@task(name="Segment Other", description="")
def segment_other(fname: str, path: os.PathLike, index_list: int, config: dict):
    # Input
    index_channel = config["detect_channels"][index_list]
    fname_image = os.path.join(path, "preprocessed", f"{fname}.tif")
    image = koopa.io.load_image(fname_image)

    # Run
    segmap = koopa.segment_other.segment(
        image=image[index_channel], index_list=index_list, config=config
    )

    # Save
    fname_out = os.path.join(path, f"segmentation_c{index_channel}", f"{fname}.tif")
    koopa.io.save_image(fname_out, segmap)
