import os
import shutil
import subprocess


def test_pipeline_2d():
    """Example pipeline basic 2D cellular data."""
    path = os.path.dirname(os.path.abspath(__file__))

    out_path = os.path.join(path, "data", "test_out_fish")
    columns = "FileID,y,x,mass,size,eccentricity,signal,frame,channel,cell_id,num_cells,area_cyto,eccentricity_cyto,area_nuclei,eccentricity_nuclei,nuclear"
    files = [
        "detection_raw_c0/20220512_EGFP_3h_2.parq",
        "koopa.cfg",
        "preprocessed/20220512_EGFP_3h_2.tif",
        "segmentation_cyto/20220512_EGFP_3h_2.tif",
        "segmentation_nuclei/20220512_EGFP_3h_2.tif",
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
            os.path.join(path, "data", "fish2d.cfg"),
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
