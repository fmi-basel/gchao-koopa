import os
import shutil
import subprocess


def test_pipeline_2d():
    shutil.rmtree("./tests/data/test_out_fish/")
    subprocess.run(
        ["koopa", "--config", "./tests/config/fish2d.cfg", "--workers", "24"],
        check=True,
    )
    for fname in [
        "summary.csv",
        "koopa.cfg",
        "preprocessed/20220512_EGFP_3h_20.tif",
        "segmentation_primary/20220512_EGFP_3h_20.tif",
        "segmentation_secondary/20220512_EGFP_3h_20.tif",
        "detection_raw_c0/20220512_EGFP_3h_20.parq",
    ]:
        assert os.path.exists(os.path.join("./tests/data/test_out_fish/", fname))
    columns = "FileID,y,x,mass,size,ecc,signal,frame,channel,primary,primary_count,distance_from_primary,secondary"
    with open("./tests/data/test_out_fish/summary.csv", "r") as f:
        first_line = f.readline().strip()
    assert columns == first_line


def test_pipeline_3d():
    pass


def test_pipeline_live():
    """Example pipeline for live cell."""
    shutil.rmtree("./tests/data/test_out_live/")
    subprocess.run(
        ["koopa", "--config", "./tests/config/live.cfg", "--workers", "24"], check=True
    )
    for fname in [
        "summary.csv",
        "alignment.png",
        "alignment.npy",
        "koopa.cfg",
        "preprocessed/20220518_18xsm_2.tif",
        "segmentation_primary/20220518_18xsm_2.tif",
        "detection_raw_c0/20220518_18xsm_2.parq",
        "colocalization_0-1/20220518_18xsm_2.parq",
        "detection_final_c0/20220518_18xsm_2.parq",
    ]:
        assert os.path.exists(os.path.join("./tests/data/test_out_live/", fname))
    columns = "FileID,y,x,mass,size,ecc,signal,frame,channel,particle,coloc_particle,primary,primary_count"
    with open("./tests/data/test_out_live/summary.csv", "r") as f:
        first_line = f.readline().strip()
    assert columns == first_line


def test_pipeline_config():
    """Create default config file."""
    subprocess.run(["koopa", "--create-config"], check=True)
    assert os.path.exists("./koopa.cfg")
    os.remove("./koopa.cfg")


def test_raw():
    assert 1


def test_pipeline_helptext():
    """Help only with and without being explicit."""
    process_1 = subprocess.check_output("koopa").decode()
    process_2 = subprocess.check_output(["koopa", "--help"]).decode()
    assert process_1 == process_2
