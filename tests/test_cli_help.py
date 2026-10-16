"""Every command-line entry point prints its help and exits with status 0."""

import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]

COMMANDS = [
    ["source.train"],
    ["source.denoise"],
    ["source.train", "experiment=10948/multinoise2noise/default"],
    ["source.denoise", "experiment=10948/multinoise2noise/default"],
    ["source.extract"],
    ["source.correct_motion"],
    ["source.datasets.split_star"],
    ["source.evaluation.make_mask"],
    ["source.evaluation.evaluate_reconstruction"],
    ["source.evaluation.evaluate_local_resolution"],
    ["source.evaluation.make_tables"],
    ["source.evaluation.make_figures"],
    ["source.evaluation.render_locres"],
]


@pytest.mark.parametrize("command", COMMANDS, ids=lambda c: " ".join(c))
def test_help(command):
    result = subprocess.run(
        [sys.executable, "-m", command[0], *command[1:], "--help"],
        cwd=REPO,
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert result.returncode == 0, result.stdout[-2000:] + result.stderr[-2000:]
    assert (
        "usage" in result.stdout.lower() or "powered by hydra" in result.stdout.lower()
    )
