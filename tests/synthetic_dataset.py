"""Tiny synthetic particle-movie dataset in the layout read by MultiNoise2NoiseDatamodule.

``<root>/<name>/`` holds ``extraction/<name>.h5`` (dataset ``data``: particles x frames x
size x size, float32), ``<name>_split.star`` (25/50/25 ``split`` column, test particles in
both rlnRandomSubset halves), ``mask.mrc`` and ``reference.mrc``.
"""

from pathlib import Path

import h5py
import mrcfile
import numpy as np
import pandas as pd
import starfile

SPLIT_FRACTIONS = {"train": 0.25, "val": 0.5, "test": 0.25}


def write_synthetic_dataset(
    root: Path,
    name: str = "synthetic",
    n_particles: int = 400,
    n_frames: int = 4,
    size: int = 32,
    n_micrographs: int = 4,
    seed: int = 0,
) -> Path:
    rng = np.random.default_rng(seed)
    dataset_dir = Path(root) / name
    (dataset_dir / "extraction").mkdir(parents=True, exist_ok=True)

    yy, xx = np.mgrid[:size, :size] - size / 2
    blob = np.exp(-(xx**2 + yy**2) / (2 * (size / 8) ** 2)).astype(np.float32)
    frames = blob[None, None] + rng.normal(0, 1, (n_particles, n_frames, size, size))
    with h5py.File(dataset_dir / "extraction" / f"{name}.h5", "w") as h5:
        h5.create_dataset("data", data=frames.astype(np.float32))

    labels = np.repeat(
        list(SPLIT_FRACTIONS),
        [int(f * n_particles) for f in SPLIT_FRACTIONS.values()],
    )
    if len(labels) != n_particles:
        raise ValueError("n_particles must be divisible by 4")
    labels = rng.permutation(labels)
    micrograph = np.arange(n_particles) % n_micrographs
    particles = pd.DataFrame(
        {
            "rlnCoordinateX": rng.uniform(0, 4000, n_particles).round(1),
            "rlnCoordinateY": rng.uniform(0, 4000, n_particles).round(1),
            "rlnMicrographName": [f"Micrographs/mic_{m:03d}.mrc" for m in micrograph],
            "rlnImageName": [
                f"{i + 1:06d}@particles/mic.mrcs" for i in range(n_particles)
            ],
            "rlnOpticsGroup": 1,
            "rlnDefocusU": rng.uniform(15000, 25000, n_particles),
            "rlnDefocusV": rng.uniform(15000, 25000, n_particles),
            "rlnDefocusAngle": rng.uniform(-180, 180, n_particles),
            "rlnCtfBfactor": 0.0,
            "rlnPhaseShift": 0.0,
            "rlnAngleRot": rng.uniform(-180, 180, n_particles),
            "rlnAngleTilt": rng.uniform(0, 180, n_particles),
            "rlnAnglePsi": rng.uniform(-180, 180, n_particles),
            "rlnOriginXAngst": rng.normal(0, 1, n_particles),
            "rlnOriginYAngst": rng.normal(0, 1, n_particles),
            "rlnMaxValueProbDistribution": 1.0,
            "rlnRandomSubset": np.arange(n_particles) % 2 + 1,
            "rlnNoisyImage": f"extraction/{name}.h5",
            "particle_index": np.arange(n_particles),
            "split": labels,
        }
    )
    optics = pd.DataFrame(
        {
            "rlnOpticsGroup": [1],
            "rlnVoltage": [300.0],
            "rlnSphericalAberration": [2.7],
            "rlnAmplitudeContrast": [0.1],
            "rlnImagePixelSize": [1.0],
            "rlnImageSize": [size],
        }
    )
    starfile.write(
        {"optics": optics, "particles": particles},
        dataset_dir / f"{name}_split.star",
        overwrite=True,
    )

    zz, yy, xx = np.mgrid[:size, :size, :size] - size / 2
    radius = np.sqrt(xx**2 + yy**2 + zz**2)
    for file_name, volume in (
        ("mask.mrc", (radius < size / 3).astype(np.float32)),
        (
            "reference.mrc",
            np.exp(-(radius**2) / (2 * (size / 8) ** 2)).astype(np.float32),
        ),
    ):
        with mrcfile.new(dataset_dir / file_name, overwrite=True) as mrc:
            mrc.set_data(volume)
            mrc.voxel_size = 1.0
    return dataset_dir
