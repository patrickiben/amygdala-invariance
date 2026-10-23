"""Check the published encoder choice and keep regenerated summaries out of the archive."""
import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]


def test_second_affect_runner_defaults_to_preserved_toisoul(tmp_path):
    fake_python = tmp_path / "python-fixture"
    fake_python.write_text('#!/bin/sh\nprintf "%s\\n" "$@"\n')
    fake_python.chmod(0o755)
    env = {**os.environ, "SP": str(tmp_path), "PYTHON": str(fake_python)}
    env.pop("ENCODER", None)
    result = subprocess.run(["bash", str(ROOT / "real_data/scripts/run_second_affect.sh")],
                            env=env, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert "extract_toisoul.py" in result.stdout
    assert "extract_affect2.py" not in result.stdout
    assert "second_affect_swap.py" in result.stdout


def test_tost_summary_output_does_not_overwrite_archived_evidence(tmp_path):
    preserved = ROOT / "real_data/scripts/reanalysis/single_neuron_tost_result.txt"
    before = preserved.read_bytes()
    result = subprocess.run([sys.executable, str(ROOT / "real_data/scripts/reanalysis/single_neuron_tost.py")],
                            env={**os.environ, "AMYG_OUTPUT": str(tmp_path)}, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert (tmp_path / "single_neuron_tost_result.txt").exists()
    assert "Delta = +0.100" in result.stdout
    assert preserved.read_bytes() == before
