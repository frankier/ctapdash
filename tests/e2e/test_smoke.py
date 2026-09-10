"""Run the packaged-build self-check against the live CLI process."""
import os
import subprocess
import sys
from pathlib import Path

from tests.dummy_data import write_dataset


def test_smoke(tmp_path: Path) -> None:
    config = write_dataset(tmp_path)
    env = dict(os.environ, MPLCONFIGDIR=str(tmp_path / "matplotlib"))
    env["_MNE_FAKE_HOME_DIR"] = str(tmp_path)
    env["PYTHONPATH"] = str(Path(__file__).resolve().parents[2] / "src")
    process = subprocess.run(
        [sys.executable, "-m", "ctapdash", "--smoke-test", "--config", str(config),
         "--no-window", "--no-browser"],
        env=env, capture_output=True, text=True, timeout=300,
    )
    assert process.returncode == 0, process.stdout + process.stderr
