import os
import shutil
import subprocess


def test_pipeline_2d():
    """Example pipeline basic 2D cellular data."""
    out_path = "./tests/data/test_out_fish/"
    columns = "FileID,y,x,mass,size,eccentricity,signal,frame,channel,cell_id,area_cyto,eccentricity_cyto,area_nuclei,eccentricity_nuclei,num_cells,nuclear"
    files = [
        "detection_raw_c0/20220512_EGFP_3h_20.parq",
        "koopa.cfg",
        "preprocessed/20220512_EGFP_3h_20.tif",
        "segmentation_cyto/20220512_EGFP_3h_20.tif",
        "segmentation_nuclei/20220512_EGFP_3h_20.tif",
        "summary.csv",
    ]

    # Run pipeline
    shutil.rmtree(out_path)
    os.mkdir(out_path)
    subprocess.run(
        ["koopa", "--config", "./tests/config/fish2d.cfg", "--workers", "24"],
        check=True,
    )

    # Check output files
    for fname in files:
        assert os.path.exists(os.path.join(out_path, fname))

    # Check output format
    with open(os.path.join(out_path, "summary.csv"), "r") as f:
        first_line = f.readline().strip()
    assert columns == first_line


def test_pipeline_files():
    """Example pipeline for Jess's flies."""
    out_path = "./tests/data/test_out_flies"
    columns = "FileID,y,x,mass,size,eccentricity,signal,frame,channel,particle,coloc_particle,cell_id,area_cyto,num_cells"
    files = [
        "colocalization_0-1/hr38-24xPP7_hr38633_PP7546_OCT_9.parq",
        "detection_final_c1/hr38-24xPP7_hr38633_PP7546_OCT_9.parq",
        "detection_raw_c0/hr38-24xPP7_hr38633_PP7546_OCT_9.parq",
        "koopa.cfg",
        "preprocessed/hr38-24xPP7_hr38633_PP7546_OCT_9.tif",
        "segmentation_cyto/hr38-24xPP7_hr38633_PP7546_OCT_9.tif",
        "summary.csv",
    ]

    # Run pipeline
    shutil.rmtree(out_path)
    os.mkdir(out_path)
    subprocess.run(
        ["koopa", "--config", "./tests/config/flies.cfg", "--workers", "2"], check=True
    )

    # Check output files
    for fname in files:
        assert os.path.exists(os.path.join(out_path, fname))

    # Check output format
    with open(os.path.join(out_path, "summary.csv"), "r") as f:
        first_line = f.readline().strip()
    assert columns == first_line


def test_pipeline_live():
    """Example pipeline for live cell."""
    out_path = "./tests/data/test_out_live"
    columns = "FileID,y,x,mass,size,eccentricity,signal,frame,channel,particle,coloc_particle,cell_id,area_cyto,eccentricity_cyto,num_cells"
    files = [
        "alignment.npy",
        "alignment_post.tif",
        "alignment_pre.tif",
        "colocalization_0-1/20220518_18xsm_2.parq",
        "detection_final_c0/20220518_18xsm_2.parq",
        "detection_raw_c0/20220518_18xsm_2.parq",
        "koopa.cfg",
        "preprocessed/20220518_18xsm_2.tif",
        "segmentation_cyto/20220518_18xsm_2.tif",
        "summary.csv",
    ]

    # Run pipeline
    shutil.rmtree(out_path)
    os.mkdir(out_path)
    subprocess.run(
        ["koopa", "--config", "./tests/config/live.cfg", "--workers", "24"], check=True
    )

    # Check output files
    for fname in files:
        assert os.path.exists(os.path.join(out_path, fname))

    # Check output format
    with open(os.path.join(out_path, "summary.csv"), "r") as f:
        first_line = f.readline().strip()
    assert columns == first_line
