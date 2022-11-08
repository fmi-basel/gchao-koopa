import os

from prefect import task, get_run_logger

import koopa


@task(
    name="Align",
    description="Registration for camera or chromatic aberation alignment.",
)
def align(path_in: os.PathLike, path_out: os.PathLike, config: dict):
    # Input
    images_reference, images_transform = koopa.align.load_alignment_images(
        path_in, config["channel_reference"], config["channel_transform"]
    )

    # Run
    if config["alignment_method"] == "pystackreg":
        matrix = koopa.align.register_alignment_pystackreg(
            images_reference, images_transform
        )
    elif config["alignment_method"] == "deepblink":
        matrix = koopa.align.register_alignment_deepblink(
            config["alignment_model"], images_reference, images_transform
        )
    else:
        raise ValueError(f"Unknown alignment method: {config['alignment_method']}")
    sr = koopa.align.get_stackreg(matrix)

    # Save
    fname_pre = os.path.join(path_out, "alignment_pre.tif")
    fname_post = os.path.join(path_out, "alignment_post.tif")
    fname_matrix = os.path.join(path_out, "alignment.npy")
    koopa.align.visualize_alignment(
        sr, images_reference[0], images_transform[0], fname_pre, fname_post
    )
    koopa.io.save_alignment(fname_matrix, sr)


@task(name="Preprocess", description="Task to open, trim, and align images.")
def preprocess(fname: str, path: str, config: dict):
    # Input
    fname_in = koopa.io.find_full_path(config["input_path"], fname, config["file_ext"])
    image = koopa.io.load_raw_image(fname_in, config["file_ext"])

    # Run
    if image.ndim != 4:
        raise ValueError(f"Image {fname} has {image.ndim} dimensions, expected 4.")
    if not config["do_3d"] and not config["do_timeseries"]:
        image = koopa.preprocess.register_3d_image(image, config["registration_method"])
    if config["do_3d"] or config["do_timeseries"]:
        image = koopa.preprocess.trim_image(
            image, config["frame_start"], config["frame_end"]
        )
    if config["crop_start"] or config["crop_end"]:
        image = koopa.preprocess.crop_image(
            image, config["crop_start"], config["crop_end"]
        )
    if config["bin_axes"]:
        image = koopa.preprocess.bin_image(image, config["bin_axes"])
    # if config["alignment_enabled"]:
    #     sr = koopa.io.load_alignment()
    #     image = koopa.align.align_image(image, sr, TODO)

    # Save
    fname_out = os.path.join(path, "preprocessed", f"{fname}.tif")
    koopa.io.save_image(fname_out, image)
