from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import hydra
import lightning as L
import rootutils
import torch
from lightning import Callback, LightningDataModule, LightningModule, Trainer
from lightning.pytorch.loggers import Logger
from omegaconf import DictConfig, OmegaConf

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
# ------------------------------------------------------------------------------------ #
# the setup_root above is equivalent to:
# - adding project root dir to PYTHONPATH
#       (so you don't need to force user to install project as a package)
#       (necessary before importing any local modules e.g. `from src import utils`)
# - setting up PROJECT_ROOT environment variable
#       (which is used as a base for paths in "configs/paths/default.yaml")
#       (this way all filepaths are the same no matter where you run the code)
# - loading environment variables from ".env" in root dir
#
# you can remove it if you:
# 1. either install project as a package or move entry files to project root dir
# 2. set `root_dir` to "." in "configs/paths/default.yaml"
#
# more info: https://github.com/ashleve/rootutils
# ------------------------------------------------------------------------------------ #

from source.utils import (
    RankedLogger,
    checkpoint_load_kwargs,
    extras,
    get_metric_value,
    instantiate_callbacks,
    instantiate_loggers,
    log_hyperparameters,
    task_wrapper,
)

# Register resolver to sanitize task_name for filenames (replace '/' with '_')
# This must be done before @hydra.main decorator processes the config
OmegaConf.register_new_resolver("sanitize_filename", lambda x: str(x).replace("/", "_"))

log = RankedLogger(__name__, rank_zero_only=True)


def _summarize_weight_sample(module: torch.nn.Module, keys: List[str]) -> str:
    parts: List[str] = []
    state = module.state_dict()
    for key in keys:
        if key not in state:
            parts.append(f"{key}=<missing>")
            continue
        tensor = state[key].detach().float()
        mean = tensor.mean().item()
        std = tensor.std(unbiased=False).item()
        parts.append(f"{key}[mean={mean:.6e},std={std:.6e}]")
    return " | ".join(parts)


def _load_pretrained_weights_if_requested(cfg: DictConfig, model: LightningModule) -> None:
    pretrained_path = cfg.get("pretrained_weights_path")
    if not pretrained_path:
        return

    ckpt_path = Path(str(pretrained_path))
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Pretrained weights not found: {ckpt_path}")

    payload = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    if isinstance(payload, dict) and "params_ema" in payload:
        state = payload["params_ema"]
    elif isinstance(payload, dict) and "params" in payload:
        state = payload["params"]
    elif isinstance(payload, dict) and "state_dict" in payload:
        state = payload["state_dict"]
    elif isinstance(payload, dict):
        state = payload
    else:
        raise ValueError(f"Unsupported pretrained checkpoint format: {type(payload)}")

    clean_state = {}
    for key, value in state.items():
        k = str(key)
        if k.startswith("module."):
            k = k[len("module.") :]
        clean_state[k] = value

    target_module = model.model if hasattr(model, "model") else model
    model_state = target_module.state_dict()
    filtered_state = {}
    skipped_shape = []
    for k, v in clean_state.items():
        if k not in model_state:
            continue
        if model_state[k].shape != v.shape:
            skipped_shape.append((k, tuple(v.shape), tuple(model_state[k].shape)))
            continue
        filtered_state[k] = v

    sample_keys = list(filtered_state.keys())[:5]
    if sample_keys:
        log.info(
            "Pretrained debug (before load): "
            + _summarize_weight_sample(target_module, sample_keys)
        )

    missing, unexpected = target_module.load_state_dict(filtered_state, strict=False)
    if sample_keys:
        log.info(
            "Pretrained debug (after load): "
            + _summarize_weight_sample(target_module, sample_keys)
        )
    log.info(
        f"Loaded pretrained weights from {ckpt_path} "
        f"(loaded={len(filtered_state)}, missing={len(missing)}, unexpected={len(unexpected)}, "
        f"skipped_shape={len(skipped_shape)})"
    )
    if skipped_shape:
        preview = ", ".join(
            [f"{k}: ckpt{src}->model{dst}" for k, src, dst in skipped_shape[:5]]
        )
        log.warning(
            f"Skipped {len(skipped_shape)} checkpoint tensors due to shape mismatch. "
            f"Examples: {preview}"
        )


@task_wrapper
def train(cfg: DictConfig) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Trains the model. Can additionally evaluate on a testset, using best weights obtained during
    training.

    This method is wrapped in optional @task_wrapper decorator, that controls the behavior during
    failure. Useful for multiruns, saving info about the crash, etc.

    :param cfg: A DictConfig configuration composed by Hydra.
    :return: A tuple with metrics and dict with all instantiated objects.
    """
    # set seed for random number generators in pytorch, numpy and python.random
    if cfg.get("seed"):
        L.seed_everything(cfg.seed, workers=True)

    log.info(f"Instantiating datamodule <{cfg.datamodule._target_}>")
    datamodule: LightningDataModule = hydra.utils.instantiate(cfg.datamodule)

    log.info(f"Instantiating model <{cfg.lightning_model._target_}>")
    model: LightningModule = hydra.utils.instantiate(cfg.lightning_model, _recursive_=False)
    _load_pretrained_weights_if_requested(cfg=cfg, model=model)

    log.info("Instantiating callbacks...")
    callbacks: List[Callback] = instantiate_callbacks(cfg.get("callbacks"))

    log.info("Instantiating loggers...")
    logger: List[Logger] = instantiate_loggers(cfg.get("logger"))

    log.info(f"Instantiating trainer <{cfg.trainer._target_}>")
    trainer: Trainer = hydra.utils.instantiate(cfg.trainer, callbacks=callbacks, logger=logger)

    object_dict = {
        "cfg": cfg,
        "datamodule": datamodule,
        "model": model,
        "callbacks": callbacks,
        "logger": logger,
        "trainer": trainer,
    }

    if logger:
        log.info("Logging hyperparameters!")
        log_hyperparameters(object_dict)

    if cfg.get("reconstruct_initial"):
        log.info("Reconstructing initial model!")
        trainer.test(model=model, datamodule=datamodule)

    model.exchange_data(datamodule=datamodule)

    # if cfg.get("reconstruct_initial"):
    #     datamodule.setup(stage="fit")

    if cfg.get("train"):
        log.info("Starting training!")
        trainer.fit(
            model=model,
            datamodule=datamodule,
            ckpt_path=cfg.get("ckpt_path"),
            **checkpoint_load_kwargs(trainer.fit),
        )

    train_metrics = trainer.callback_metrics

    if cfg.get("test"):
        log.info("Starting testing!")
        ckpt_path = trainer.checkpoint_callback.best_model_path
        if ckpt_path == "":
            ckpt_path = cfg.get("ckpt_path")
            if ckpt_path == "":
                log.warning("Best ckpt not found! Using current weights for testing...")
                ckpt_path = None

        trainer.test(
            model=model,
            datamodule=datamodule,
            ckpt_path=ckpt_path,
            **checkpoint_load_kwargs(trainer.test),
        )

        log.info("testing Full Dose denoising reconstruction")
        # model.hparams.reconstruction_mode = "denoised_full_dose"
        # trainer.test(model=model, datamodule=datamodule, ckpt_path=ckpt_path)
        log.info(f"Best ckpt path: {ckpt_path}")

    test_metrics = trainer.callback_metrics

    # merge train and test metrics
    metric_dict = {**train_metrics, **test_metrics}

    return metric_dict, object_dict


@hydra.main(version_base="1.3", config_path="../configs_train", config_name="train.yaml")
def main(cfg: DictConfig) -> Optional[float]:
    """Main entry point for training.

    :param cfg: DictConfig configuration composed by Hydra.
    :return: Optional[float] with optimized metric value.
    """
    # apply extra utilities
    # (e.g. ask for tags if none are provided in cfg, print cfg tree, etc.)
    extras(cfg)

    # train the model
    metric_dict, _ = train(cfg)

    # safely retrieve metric value for hydra-based hyperparameter optimization
    metric_value = get_metric_value(
        metric_dict=metric_dict, metric_name=cfg.get("optimized_metric")
    )

    # return optimized metric
    return metric_value


if __name__ == "__main__":
    main()
