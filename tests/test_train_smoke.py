"""One epoch of experiment=10948/multinoise2noise/default with debug=limit on a synthetic HDF5.

Runs ``python -m source.train`` on the CPU (debug/default.yaml) with the 25/50/25
``split`` column; the test stage must reconstruct both half-maps from the test particles.
``python -m source.denoise`` with the resulting checkpoint must then write one denoised
stack per micrograph and ``denoised.star``.
"""

import os
import subprocess
import sys
from pathlib import Path

import mrcfile
import numpy as np

from synthetic_dataset import write_synthetic_dataset

REPO = Path(__file__).resolve().parents[1]


def run(module, overrides):
    result = subprocess.run(
        [sys.executable, "-m", module, "experiment=10948/multinoise2noise/default"]
        + overrides,
        cwd=REPO,
        env={**os.environ, "PROJECT_ROOT": str(REPO)},
        capture_output=True,
        text=True,
        timeout=1800,
    )
    assert result.returncode == 0, result.stdout[-3000:] + result.stderr[-3000:]


def test_one_epoch_debug_limit_then_denoise(tmp_path):
    dataset = write_synthetic_dataset(tmp_path / "data", n_frames=4, size=32)
    storage = tmp_path / "storage"
    common = [
        f"paths.data_dir={tmp_path / 'data'}",
        f"paths.storage_dir={storage}",
        f"datamodule.dataset_name={dataset.name}",
        "datamodule.split_mode=column",
        f"datamodule.particles_star_path={dataset / f'{dataset.name}_split.star'}",
        f"datamodule.original_reconstruction_path={dataset / 'reference.mrc'}",
        "datamodule.splits=4",
        "datamodule.max_splits=4",
        "datamodule.old_size=32",
        "datamodule.new_size=32",
        "datamodule.bg_radius=12",
        "datamodule.angpix=1.0",
        "datamodule.num_workers=1",
    ]
    run(
        "source.train",
        ["debug=limit", "trainer.max_epochs=1", "trainer.min_epochs=1"] + common,
    )
    checkpoints = list(storage.rglob("checkpoints/*.ckpt"))
    assert len(checkpoints) == 1
    halves_dir = storage / "reconstruction" / dataset.name / "debug" / "denoised_mean"
    for half in (1, 2):
        volume = mrcfile.open(halves_dir / f"half_{half}_reconstruction.mrc").data
        assert volume.shape == (32, 32, 32)
        assert np.isfinite(volume).all() and np.abs(volume).max() > 0

    # Checkpoint names contain '=', so the value is quoted for the Hydra override parser.
    run(
        "source.denoise",
        [
            "debug=default",
            f"ckpt_path='{checkpoints[0]}'",
            "task_name=denoise_smoke",
            "datamodule.batch_size=16",
        ]
        + common,
    )
    stacks = sorted(
        (storage / "reconstruction" / dataset.name / "denoise_smoke").glob("*.mrcs")
    )
    assert len(stacks) == 4
    assert sum(mrcfile.open(p).data.shape[0] for p in stacks) == 400
    assert (dataset / "denoised.star").exists()
