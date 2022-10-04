import os
import subprocess


def test_pipeline_config():
    """Create default config file."""
    if os.path.exists("./koopa.cfg"):
        os.remove("./koopa.cfg")
    subprocess.run(["koopa", "--create-config"])
    assert os.path.exists("./koopa.cfg")


def test_pipeline_helptext():
    """Help only with and without being explicit."""
    process_1 = subprocess.check_output("koopa").decode()
    process_2 = subprocess.check_output(["koopa", "--help"]).decode()
    assert process_1 == process_2


def test_rerun_merge():
    fname = "./tests/data/test_out_flies/summary.csv"
    last_modified = os.stat(fname).st_mtime
    subprocess.run(
        [
            "koopa",
            "--config",
            "./tests/config/flies.cfg",
            "--workers",
            "2",
            "--task",
            "Merge",
            "--force",
        ]
    )
    assert os.stat(fname).st_mtime > last_modified


def test_rerun_colocalize():
    fname = "./tests/data/test_out_flies/colocalization_0-1/hr38-24xPP7_hr38633_PP7546_OCT_9.parq"
    os.remove(fname)
    subprocess.run(
        [
            "koopa",
            "--config",
            "./tests/config/flies.cfg",
            "--workers",
            "2",
            "--task",
            "ColocalizeFrame",
        ],
        check=True,
    )
    assert os.path.exists(fname)
