#!/usr/bin/env python
"""Re-label released dose-weighted checkpoints with the paper dose-weighting loss.

Network weights are copied unchanged; only ``hyper_parameters["criterion"]`` is
replaced by the criterion of ``experiment=<dataset>/<family>/<experiment>`` (default
``dose_weighted``; the conversion of the released checkpoints was run while the paper
config was still named ``dose_weighted_paper``).
The script refuses to convert a checkpoint whose stored criterion differs from the
new one in anything other than the added dose-weighting switches.

Each converted checkpoint is verified by (1) instantiating its stored criterion and
checking the switches, and (2) instantiating the Lightning module from the new
experiment config and loading the converted state dict with ``strict=True``.
"""

import argparse
import json
import os
import sys
from pathlib import Path

import hydra
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
os.environ.setdefault("PROJECT_ROOT", str(REPO))
import source.utils  # noqa: E402,F401  (registers OmegaConf resolvers)

NEW_KEYS = (
    "prediction_dose_weighting",
    "target_dose_weighting",
    "undo_target_dose_weighting",
    "loss_dose_weighting",
    "target_weight_floor",
)
VARIANTS = {
    "dose_weighted": "multinoise2noise",
    "dose_weighted_ctf": "multinoise2noise_ctf",
}


def compose_cfg(dataset: str, family: str, experiment: str, overrides: list[str]):
    with initialize_config_dir(str(REPO / "configs_train"), version_base="1.3"):
        return compose(
            "train.yaml",
            overrides=[
                f"experiment={dataset}/{family}/{experiment}",
                "logger=csv",
            ]
            + overrides,
        )


def convert(
    ckpt_path: Path,
    out_path: Path,
    dataset: str,
    family: str,
    experiment: str,
    overrides: list[str],
) -> dict:
    cfg = compose_cfg(dataset, family, experiment, overrides)
    new_criterion = OmegaConf.to_container(cfg.lightning_model.criterion, resolve=True)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    old_criterion = OmegaConf.to_container(
        ckpt["hyper_parameters"]["criterion"], resolve=True
    )
    stripped = {k: v for k, v in new_criterion.items() if k not in NEW_KEYS}
    if stripped != old_criterion:
        raise RuntimeError(
            f"{ckpt_path}: stored criterion differs from the new config beyond the new keys\n"
            f"stored: {old_criterion}\nnew:    {new_criterion}"
        )
    ckpt["hyper_parameters"]["criterion"] = OmegaConf.create(new_criterion)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(ckpt, out_path)

    reloaded = torch.load(out_path, map_location="cpu", weights_only=False)
    criterion = hydra.utils.instantiate(reloaded["hyper_parameters"]["criterion"])
    switches = {k: getattr(criterion, k) for k in NEW_KEYS}
    for key in NEW_KEYS:
        if key in new_criterion and switches[key] != new_criterion[key]:
            raise RuntimeError(
                f"{out_path}: {key}={switches[key]} != {new_criterion[key]}"
            )
    model = hydra.utils.instantiate(cfg.lightning_model, _recursive_=False)
    result = model.load_state_dict(reloaded["state_dict"], strict=True)
    weights_equal = all(
        torch.equal(reloaded["state_dict"][k], v) for k, v in ckpt["state_dict"].items()
    )
    return {
        "dataset": dataset,
        "variant": out_path.stem,
        "source": str(ckpt_path),
        "converted": str(out_path),
        "criterion": new_criterion["_target_"],
        "switches": switches,
        "missing_keys": list(result.missing_keys),
        "unexpected_keys": list(result.unexpected_keys),
        "state_dict_unchanged": weights_equal,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-root", type=Path, required=True)
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--datasets", nargs="+", default=["10025", "10474", "10948"])
    parser.add_argument(
        "--override",
        action="append",
        default=[],
        metavar="DATASET:KEY=VALUE",
        help="Hydra override for one dataset, e.g. to point verification at local files.",
    )
    parser.add_argument("--experiment", default="dose_weighted")
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    per_dataset: dict[str, list[str]] = {}
    for item in args.override:
        dataset, override = item.split(":", 1)
        per_dataset.setdefault(dataset, []).append(override)

    rows = []
    for dataset in args.datasets:
        overrides = per_dataset.get(dataset, [])
        for variant, family in VARIANTS.items():
            src = args.checkpoint_root / dataset / f"{variant}.ckpt"
            dst = args.out_root / dataset / f"{variant}.ckpt"
            rows.append(convert(src, dst, dataset, family, args.experiment, overrides))
            print(json.dumps(rows[-1]))
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(rows, indent=1))


if __name__ == "__main__":
    main()
