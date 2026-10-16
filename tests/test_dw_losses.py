import pytest
import torch

from source.loss_functions.ctf_dw_l2 import CTF_DW_MSELoss
from source.loss_functions.dw_l2 import DW_MSELoss
from source.reconstruction.dose_weighter import DoseWeighter

N_FRAMES = 4
SIZE = 16
# The spectra are complex64, the reference formulas below are evaluated in float64.
FLOAT32_RTOL = 1e-5
FLOAT32_ATOL = 1e-6
PAPER_MODES = {
    DW_MSELoss: dict(
        prediction_dose_weighting=False,
        undo_target_dose_weighting=True,
        loss_dose_weighting=True,
    ),
    CTF_DW_MSELoss: dict(
        target_dose_weighting=False,
        undo_target_dose_weighting=True,
        loss_dose_weighting=True,
    ),
}


def make_dose_weighter() -> DoseWeighter:
    return DoseWeighter(
        n_frames=N_FRAMES,
        dose_per_frame=1.5,
        pre_exposure=0.0,
        voltage=300,
        pixel_size=1.0,
        size=(SIZE, SIZE),
    )


def make_batch(seed: int = 0, n_splits: int = N_FRAMES):
    generator = torch.Generator().manual_seed(seed)
    target = torch.randn(2, n_splits, 1, SIZE, SIZE, generator=generator)
    denoised = torch.randn(2, n_splits, 1, SIZE, SIZE, generator=generator)
    ctf = torch.rand(2, 1, 1, SIZE, SIZE // 2 + 1, generator=generator)
    return target, denoised, ctf


def legacy_dw_loss(loss_fn: DW_MSELoss, target, preds) -> torch.Tensor:
    """Verbatim formula of DW_MSELoss.forward before the dose-weighting options were added."""
    f_preds = torch.fft.rfft2(preds, norm="ortho")
    if f_preds.shape[-4] == loss_fn.dose_weighter.weight.shape[-4]:
        f_preds *= loss_fn.dose_weighter.weight
    else:
        f_preds *= loss_fn.frames_weight_sum
    f_target = torch.fft.rfft2(target, norm="ortho")
    loss = torch.nn.functional.mse_loss(
        f_preds.imag, f_target.imag, reduction="none"
    ) + torch.nn.functional.mse_loss(f_preds.real, f_target.real, reduction="none")
    if loss_fn.correct_frequencies:
        loss = loss / loss_fn.frames_weight_sum
    return loss.mean()


def legacy_ctf_dw_loss(loss_fn: CTF_DW_MSELoss, preds, target, ctf) -> torch.Tensor:
    """Verbatim formula of CTF_DW_MSELoss.forward before the dose-weighting options were added."""
    f_preds = torch.fft.rfft2(preds, norm="ortho")
    if f_preds.shape[-4] == loss_fn.dose_weighter.weight.shape[-4]:
        f_preds *= loss_fn.dose_weighter.weight
    else:
        f_preds *= loss_fn.frames_weight_sum
    f_target = torch.fft.rfft2(target, norm="ortho")
    loss = (
        (f_preds.real - f_target.real) ** 2 + (f_preds.imag - f_target.imag) ** 2
    ) * (ctf**loss_fn.n)
    if loss_fn.correct_frequencies:
        loss = loss / loss_fn.loss_div
    return loss.mean()


@pytest.mark.parametrize("n_splits", [N_FRAMES, 1])
def test_dw_defaults_match_legacy(n_splits):
    loss_fn = DW_MSELoss(dose_weighter=make_dose_weighter())
    target, denoised, _ = make_batch(n_splits=n_splits)
    expected = legacy_dw_loss(loss_fn, target, denoised)
    torch.testing.assert_close(loss_fn(target, denoised), expected)


@pytest.mark.parametrize("n_splits", [N_FRAMES, 1])
def test_ctf_dw_defaults_match_legacy(n_splits):
    loss_fn = CTF_DW_MSELoss(dose_weighter=make_dose_weighter())
    target, denoised, ctf = make_batch(n_splits=n_splits)
    expected = legacy_ctf_dw_loss(loss_fn, target, denoised, ctf)
    torch.testing.assert_close(loss_fn(target, denoised, ctf), expected)


def paper_terms(loss_fn, target, denoised):
    weight = loss_fn.dose_weighter.weight
    f_target = torch.fft.rfft2(target, norm="ortho") / weight.clamp_min(
        loss_fn.target_weight_floor
    )
    f_denoised = torch.fft.rfft2(denoised, norm="ortho")
    return weight * (f_denoised - f_target).abs() ** 2


def test_dw_paper_mode_matches_equation():
    loss_fn = DW_MSELoss(
        dose_weighter=make_dose_weighter(),
        prediction_dose_weighting=False,
        undo_target_dose_weighting=True,
        loss_dose_weighting=True,
    )
    target, denoised, _ = make_batch()
    expected = (
        paper_terms(loss_fn, target, denoised) / loss_fn.frames_weight_sum
    ).mean()
    torch.testing.assert_close(
        loss_fn(target, denoised), expected, rtol=1e-5, atol=1e-6, check_dtype=False
    )


def test_ctf_dw_paper_mode_matches_equation():
    loss_fn = CTF_DW_MSELoss(
        dose_weighter=make_dose_weighter(),
        target_dose_weighting=False,
        undo_target_dose_weighting=True,
        loss_dose_weighting=True,
    )
    target, denoised, ctf = make_batch()
    expected = (paper_terms(loss_fn, target, denoised) * ctf / loss_fn.loss_div).mean()
    torch.testing.assert_close(
        loss_fn(target, denoised, ctf), expected, rtol=FLOAT32_RTOL, atol=FLOAT32_ATOL
    )


def test_undo_recovers_unweighted_target():
    """A dose-weighted target with undo enabled gives zero loss for the unweighted signal."""
    dose_weighter = make_dose_weighter()
    loss_fn = DW_MSELoss(
        dose_weighter=dose_weighter,
        prediction_dose_weighting=False,
        undo_target_dose_weighting=True,
        loss_dose_weighting=True,
        target_weight_floor=1e-12,
    )
    _, signal, _ = make_batch(seed=1)
    f_weighted = torch.fft.rfft2(signal, norm="ortho") * loss_fn.dose_weighter.weight
    weighted_target = torch.fft.irfft2(f_weighted, s=(SIZE, SIZE), norm="ortho")
    assert loss_fn(weighted_target, signal).item() < 1e-10


@pytest.mark.parametrize("loss_cls", [DW_MSELoss, CTF_DW_MSELoss])
def test_target_and_undo_are_exclusive(loss_cls):
    with pytest.raises(ValueError, match="mutually exclusive"):
        loss_cls(
            dose_weighter=make_dose_weighter(),
            target_dose_weighting=True,
            undo_target_dose_weighting=True,
        )


@pytest.mark.parametrize("mode", ["legacy", "paper"])
@pytest.mark.parametrize("loss_cls", [DW_MSELoss, CTF_DW_MSELoss])
@pytest.mark.parametrize("n_splits", [N_FRAMES, 1])
def test_backward_with_float32_inputs(loss_cls, mode, n_splits):
    """Training passes float32 frames; float64 dose weights must not break autograd."""
    kwargs = PAPER_MODES[loss_cls] if mode == "paper" else {}
    loss_fn = loss_cls(dose_weighter=make_dose_weighter(), **kwargs)
    target, denoised, ctf = make_batch(n_splits=n_splits)
    denoised.requires_grad_(True)
    args = (target, denoised, ctf) if loss_cls is CTF_DW_MSELoss else (target, denoised)
    loss = loss_fn(*args)
    loss.backward()
    assert torch.isfinite(loss)
    assert denoised.grad is not None
    assert denoised.grad.dtype == torch.float32
    assert torch.isfinite(denoised.grad).all()


@pytest.mark.parametrize("loss_cls", [DW_MSELoss, CTF_DW_MSELoss])
def test_defaults_keep_legacy_dtype(loss_cls):
    loss_fn = loss_cls(dose_weighter=make_dose_weighter())
    target, denoised, ctf = make_batch()
    args = (target, denoised, ctf) if loss_cls is CTF_DW_MSELoss else (target, denoised)
    legacy = legacy_ctf_dw_loss if loss_cls is CTF_DW_MSELoss else legacy_dw_loss
    assert loss_fn(*args).dtype == legacy(loss_fn, *args).dtype


@pytest.mark.parametrize("loss_cls", [DW_MSELoss, CTF_DW_MSELoss])
def test_floor_must_be_positive(loss_cls):
    with pytest.raises(ValueError, match="target_weight_floor"):
        loss_cls(
            dose_weighter=make_dose_weighter(),
            target_dose_weighting=False,
            target_weight_floor=0.0,
        )
