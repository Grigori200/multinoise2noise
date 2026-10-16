from types import SimpleNamespace

import lightning.pytorch as pl
import pytest
from lightning.fabric.utilities.data import AttributeDict

from source._noise2noise.multinoise2noise import MultiNoise2NoiseLightningModel


def make_model(split_mode):
    model = MultiNoise2NoiseLightningModel.__new__(MultiNoise2NoiseLightningModel)
    pl.LightningModule.__init__(model)
    model._hparams = AttributeDict(
        logging=AttributeDict(reconstruction_wait=0, reconstruction_step=1)
    )
    datamodule = None if split_mode is None else SimpleNamespace(split_mode=split_mode)
    model._trainer = SimpleNamespace(datamodule=datamodule, current_epoch=0)
    return model


@pytest.mark.parametrize("split_mode", ["random_subset", None])
def test_validation_reconstructs_outside_column_mode(split_mode):
    model = make_model(split_mode)
    model._on_evaluation_epoch_start("val")
    assert model.reconstruction_condition


def test_validation_never_reconstructs_in_column_mode():
    model = make_model("column")
    model._on_evaluation_epoch_start("val")
    assert not model.reconstruction_condition


@pytest.mark.parametrize("split_mode", ["column", "random_subset"])
def test_test_phase_always_reconstructs(split_mode):
    model = make_model(split_mode)
    model._on_evaluation_epoch_start("test")
    assert model.reconstruction_condition
