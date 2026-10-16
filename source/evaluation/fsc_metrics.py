"""Scalar summaries of Fourier shell correlation curves as reported in the manuscript."""

import math

import torch


def shell_resolution(shell: int, n_shells: int, angpix: float) -> float:
    """Resolution in Å of shell ``shell`` for a curve with ``n_shells`` shells (0 .. Nyquist)."""
    if shell <= 0:
        return math.inf
    return (n_shells - 1) * 2 * angpix / shell


def first_shell_below(fsc: torch.Tensor, threshold: float) -> int | None:
    """Index of the first shell whose FSC is below ``threshold`` (None if it never is)."""
    below = torch.nonzero(fsc < threshold, as_tuple=False)
    if below.numel() == 0:
        return None
    return int(below[0, 0])


def resolution_at_threshold(fsc: torch.Tensor, angpix: float, threshold: float) -> dict:
    """Resolution at the first shell at which the curve falls below ``threshold``.

    If the curve never falls below the threshold, the Nyquist resolution is reported and
    ``crossed`` is False.
    """
    fsc = torch.as_tensor(fsc).flatten()
    if fsc.ndim != 1 or fsc.numel() < 2:
        raise ValueError(
            f"Expected a 1D FSC curve with at least two shells, got {tuple(fsc.shape)}"
        )
    shell = first_shell_below(fsc, threshold)
    if shell is None:
        return {"shell": fsc.numel() - 1, "resolution": 2 * angpix, "crossed": False}
    return {
        "shell": shell,
        "resolution": shell_resolution(shell, fsc.numel(), angpix),
        "crossed": True,
    }


def mean_fsc(fsc: torch.Tensor) -> float:
    """Mean of the curve over all shells from 0 up to and including Nyquist."""
    return float(torch.as_tensor(fsc, dtype=torch.float64).flatten().mean())
