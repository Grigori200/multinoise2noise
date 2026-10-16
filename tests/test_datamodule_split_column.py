import os
from pathlib import Path

import h5py
import hydra
from hydra.errors import InstantiationException
import numpy as np
import pandas as pd
import pytest
import starfile
from hydra import compose, initialize_config_dir

import source.utils  # noqa: F401  (registers OmegaConf resolvers)

REPO = Path(__file__).resolve().parents[1]
N_FRAMES = 4
SIZE = 32
SPLIT_COUNTS = {"train": 6, "val": 12, "test": 6}


def write_synthetic_dataset(root: Path) -> Path:
    rows = []
    for split, count in SPLIT_COUNTS.items():
        for i in range(count):
            rows.append(
                {
                    "rlnMicrographName": f"Movies/mic_{i % 3}.mrc",
                    "rlnCoordinateX": float(len(rows)),
                    "rlnCoordinateY": float(i),
                    "rlnAnglePsi": 0.0,
                    "rlnAngleRot": 0.0,
                    "rlnAngleTilt": 0.0,
                    "rlnOriginXAngst": 0.0,
                    "rlnOriginYAngst": 0.0,
                    "rlnDefocusU": 15000.0,
                    "rlnDefocusV": 15000.0,
                    "rlnDefocusAngle": 0.0,
                    "rlnCtfBfactor": 0.0,
                    "rlnPhaseShift": 0.0,
                    "rlnMaxValueProbDistribution": 1.0,
                    "rlnRandomSubset": 1 + i % 2,
                    "rlnNoisyImage": "extraction/synthetic.h5",
                    "particle_index": len(rows),
                    "split": split,
                }
            )
    particles = pd.DataFrame(rows)
    optics = pd.DataFrame({"rlnOpticsGroup": [1], "rlnImagePixelSize": [1.0]})
    star_path = root / "synthetic_split.star"
    starfile.write({"optics": optics, "particles": particles}, star_path)
    (root / "extraction").mkdir()
    rng = np.random.default_rng(0)
    with h5py.File(root / "extraction" / "synthetic.h5", "w") as f:
        f["data"] = rng.standard_normal((len(particles), N_FRAMES, SIZE, SIZE)).astype(
            np.float32
        )
    return star_path


def compose_datamodule(root: Path, star_path: Path, split_mode: str):
    os.environ.setdefault("PROJECT_ROOT", str(REPO))
    h5 = str(root / "extraction" / "synthetic.h5")
    overrides = [
        "experiment=10948/multinoise2noise/default",
        "logger=csv",
        f"++local.storage_dir={root}",
        "++local.user_name=test",
        f"datamodule.dataset_path={root}",
        f"datamodule.particles_star_path={star_path}",
        f"datamodule.split_mode={split_mode}",
        f"datamodule.splits={N_FRAMES}",
        f"datamodule.max_splits={N_FRAMES}",
        f"datamodule.new_size={SIZE}",
        "datamodule.batch_size=2",
        "datamodule.num_workers=1",
        f"datamodule.datasets.train.h5_path={h5}",
        f"datamodule.datasets.val.h5_path={h5}",
    ]
    with initialize_config_dir(str(REPO / "configs_train"), version_base="1.3"):
        cfg = compose("train.yaml", overrides=overrides)
    return hydra.utils.instantiate(cfg.datamodule, _recursive_=False)


def test_column_mode_loaders(tmp_path):
    star_path = write_synthetic_dataset(tmp_path)
    datamodule = compose_datamodule(tmp_path, star_path, "column")
    datamodule.setup("fit")
    assert len(datamodule.dataset_train) == SPLIT_COUNTS["train"]
    assert len(datamodule.dataset_val) == SPLIT_COUNTS["val"]
    assert [len(d) for d in datamodule.dataset_test] == [3, 3]
    assert set(datamodule.dataset_train.data["split"]) == {"train"}
    assert set(datamodule.dataset_val.data["split"]) == {"val"}
    for dataset in datamodule.dataset_test:
        assert set(dataset.data["split"]) == {"test"}

    batch, _, _ = next(iter(datamodule.train_dataloader()))
    item = batch[0]
    assert item["input"].shape == (2, N_FRAMES, 1, SIZE, SIZE)
    assert item["target"].shape == (2, N_FRAMES, 1, SIZE, SIZE)
    val_batch = next(iter(datamodule.val_loader[0]))
    assert val_batch["full_dose"].shape == (2, 1, 1, SIZE, SIZE)


def test_column_mode_requires_three_labels(tmp_path):
    star_path = write_synthetic_dataset(tmp_path)
    star = starfile.read(star_path)
    star["particles"].loc[star["particles"]["split"] == "val", "split"] = "train"
    starfile.write(star, star_path, overwrite=True)
    with pytest.raises(
        InstantiationException, match="'split' column must contain exactly"
    ):
        compose_datamodule(tmp_path, star_path, "column")


def test_unknown_split_mode_rejected(tmp_path):
    star_path = write_synthetic_dataset(tmp_path)
    with pytest.raises(InstantiationException, match="Unknown split_mode"):
        compose_datamodule(tmp_path, star_path, "bogus")
