import os
import shutil
import subprocess


def test_pipeline_live():
    """Example pipeline for live cell."""
    path = os.path.dirname(os.path.abspath(__file__))

    out_path = os.path.join(path, "data", "test_out_live")
    columns = "FileID,y,x,mass,size,eccentricity,signal,frame,channel,particle,coloc_particle,cell_id,num_cells,area_cyto,eccentricity_cyto"
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
    if os.path.exists(out_path):
        shutil.rmtree(out_path)
    os.makedirs(out_path)
    subprocess.run(
        [
            "python",
            os.path.abspath(os.path.join(path, "..", "src", "cli.py")),
            "--config",
            os.path.join(path, "data", "live.cfg"),
        ],
        check=True,
    )

    # Check output files
    for fname in files:
        assert os.path.exists(os.path.join(out_path, fname))

    # Check output format
    with open(os.path.join(out_path, "summary.csv"), "r") as f:
        first_line = f.readline().strip()
    assert columns == first_line
