import inspect

import lightning

from source.utils import checkpoint_load_kwargs


def test_kwargs_follow_the_signature():
    def with_flag(ckpt_path=None, weights_only=None):
        return None

    def without_flag(ckpt_path=None):
        return None

    assert checkpoint_load_kwargs(with_flag) == {"weights_only": False}
    assert checkpoint_load_kwargs(without_flag) == {}


def test_installed_trainer_methods_accept_the_kwargs():
    trainer = lightning.Trainer(logger=False, enable_checkpointing=False)
    for method in (trainer.fit, trainer.test, trainer.predict):
        kwargs = checkpoint_load_kwargs(method)
        assert kwargs in ({}, {"weights_only": False})
        assert set(kwargs) <= set(inspect.signature(method).parameters)
