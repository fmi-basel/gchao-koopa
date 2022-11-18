import os
import shutil
import subprocess


def test_pipeline_files():
    """Example pipeline for Jess's flies."""
    path = os.path.dirname(os.path.abspath(__file__))

    out_path = os.path.join(path, "data", "test_out_flies")
    columns = "FileID,y,x,mass,size,eccentricity,signal,frame,channel,particle,particle_0-1,coloc_particle_0-1,cell_id,num_cells,area_nuclei,other_c1"
    files = [
        "colocalization_0-1/hr38-24xPP7_hr38633_PP7546_OCT_9.parq",
        "detection_final_c1/hr38-24xPP7_hr38633_PP7546_OCT_9.parq",
        "detection_raw_c0/hr38-24xPP7_hr38633_PP7546_OCT_9.parq",
        "koopa.cfg",
        "preprocessed/hr38-24xPP7_hr38633_PP7546_OCT_9.tif",
        "segmentation_nuclei/hr38-24xPP7_hr38633_PP7546_OCT_9.tif",
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
            os.path.join(path, "data", "flies.cfg"),
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
