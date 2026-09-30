"""Importing analysis modules must not start the analysis or require data."""

import os
from pathlib import Path
import shutil
import subprocess
import sys


def test_import_without_data_or_results(tmp_path):
    source = Path(__file__).resolve().parents[1] / "scripts"
    scripts = tmp_path / "scripts"
    shutil.copytree(source, scripts, ignore=shutil.ignore_patterns("__pycache__"))
    shutil.copytree(source.parent / "src", tmp_path / "src",
                    ignore=shutil.ignore_patterns("__pycache__"))
    env = dict(os.environ, PYTHONPATH=str(scripts), MPLBACKEND="Agg",
               MPLCONFIGDIR=str(tmp_path / "mpl"))
    result = subprocess.run(
        [sys.executable, "-c", "import runner"],
        cwd=tmp_path, env=env, capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout == ""
    assert not (tmp_path / "results").exists()
