#!/usr/bin/env python
"""Editable install of the training and evaluation code: ``pip install -e .[scunet]``.

The entry points (``python -m source.train`` etc.) locate their Hydra configs relative to the
repository (``.project-root``), so install from a checkout with ``-e``. ``requirements.txt``
is the exact environment used for the manuscript; the lower bounds below are those versions.
"""

from setuptools import find_packages, setup

setup(
    name="multinoise2noise",
    version="1.0.0",
    description="Multi-Noise2Noise: physics-aware self-supervised denoising of cryo-EM particle movies",
    author="Mateusz Grzesiuk",
    author_email="mateusz.grzesiuk@pwr.edu.pl",
    license="MIT",
    packages=find_packages(include=["source", "source.*"]),
    install_requires=[
        "h5py>=3.14.0",
        "hydra-colorlog>=1.2.0",
        "hydra-core>=1.3.2",
        "lightning>=2.5.4",
        "lightning-utilities>=0.15.2",
        "matplotlib>=3.10.6",
        "mrcfile>=1.5.4",
        "numpy>=2.3.2",
        "omegaconf>=2.3.0",
        "pandas>=2.3.2",
        "pillow>=11.3.0",
        "rich>=14.1.0",
        "rootutils>=1.0.7",
        "starfile>=0.5.13",
        "tifffile>=2025.8.28",
        "torch>=2.8.0",
        "torchmetrics>=1.8.1",
        "torchvision>=0.23.0",
        "tqdm>=4.67.1",
        "wandb>=0.21.3",
    ],
    extras_require={
        # source/models/scunet.py (configs model=scunet); not part of requirements.txt.
        "scunet": ["einops>=0.8.2", "timm>=1.0.24"],
        # source/evaluation/render_locres.py; not part of requirements.txt.
        "render": ["pyvista>=0.48.4"],
        "test": ["pytest"],
    },
)
