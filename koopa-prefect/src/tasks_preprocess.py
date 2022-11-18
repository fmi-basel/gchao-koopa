import os

from prefect import task, get_run_logger

import koopa


@task(name="Configuration")
def configuration(path: os.PathLike, force: bool):
    logger = get_run_logger()

    # Parse configuration
    cfg = koopa.io.load_config(path)
    koopa.config.validate_config(cfg)
    logger.info("Configuration file validated.")
    config = koopa.config.flatten_config(cfg)

    # Save config
    cfg = koopa.config.add_versioning(cfg)
    fname_config = os.path.join(config["output_path"], "koopa.cfg")
    koopa.io.save_config(fname_config, cfg)

    config["force"] = force
    return config


@task(name="Align")
def align(path_in: os.PathLike, path_out: os.PathLike, config: dict):
    """Registration for camera or chromatic aberation alignment."""
    # Config
    fname_pre = os.path.join(path_out, "alignment_pre.tif")
    fname_post = os.path.join(path_out, "alignment_post.tif")
    fname_matrix = os.path.join(path_out, "alignment.npy")
    if not config["force"] and all(
        os.path.exists(i) for i in (fname_pre, fname_post, fname_matrix)
    ):
        return

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
    koopa.align.visualize_alignment(
        sr, images_reference[0], images_transform[0], fname_pre, fname_post
    )
    koopa.io.save_alignment(fname_matrix, sr)


@task(name="Preprocess")
def preprocess(fname: str, path: str, config: dict):
    """Task to open, trim, and align images."""
    # Config
    logger = get_run_logger()
    fname_in = koopa.io.find_full_path(config["input_path"], fname, config["file_ext"])
    fname_out = os.path.join(path, "preprocessed", f"{fname}.tif")
    if not config["force"] and os.path.exists(fname_out):
        return

    # Input
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
    # TODO add multiple transform channels?
    if config["alignment_enabled"]:
        fname_align = os.path.join(path, "alignment.npy")
        sr = koopa.io.load_alignment(fname_align)
        image = koopa.align.align_image(image, sr, [config["channel_transform"]])
    logger.debug(f"Preprocessed {fname}")

    # Save
    koopa.io.save_image(fname_out, image)
