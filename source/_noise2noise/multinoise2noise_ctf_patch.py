import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau

from source._noise2noise.multinoise2noise_ctf import MultiNoise2NoiseCTFLightningModel
from source._noise2noise.patching import (
    _build_patch_grid_no_pad,
    _center_weight,
    _crop_pred_to_patch_size,
    match_ctf_to_rfft_shape,
)


class MultiNoise2NoiseCTFPatchLightningModel(MultiNoise2NoiseCTFLightningModel):
    """
    Patch-wise CTF variant of MultiNoise2NoiseCTFLightningModel.

    Only _shared_step is changed: model inference is patch-based and CTF tensor
    is adjusted to the stitched output FFT shape before criterion evaluation.
    """

    def __init__(self, *args, **kwargs) -> None:
        super(MultiNoise2NoiseCTFPatchLightningModel, self).__init__(*args, **kwargs)
        self.automatic_optimization = False
        self._patch_info_logged = False

    def _log_patch_info(self, h: int, w: int) -> None:
        """Log patch configuration info once at the start of training."""
        if self._patch_info_logged:
            return
        self._patch_info_logged = True
        
        patch_size = int(getattr(self.hparams, "patch_size", 128))
        overlap = int(getattr(self.hparams, "patch_overlap", 16))
        patch_batch_size = int(getattr(self.hparams, "patch_batch_size", 1) or 1)
        
        y_starts, x_starts = _build_patch_grid_no_pad(h, w, patch_size, overlap)
        n_patches_y = len(y_starts)
        n_patches_x = len(x_starts)
        total_patches = n_patches_y * n_patches_x
        
        stride = patch_size - overlap
        
        print(f"\n{'='*60}")
        print(f"PATCH CONFIGURATION")
        print(f"{'='*60}")
        print(f"  Input image size:     {h} x {w}")
        print(f"  Patch size:           {patch_size} x {patch_size}")
        print(f"  Overlap:              {overlap}")
        print(f"  Stride:               {stride}")
        print(f"  Patches per image:    {n_patches_y} x {n_patches_x} = {total_patches}")
        print(f"  Patch batch size:     {patch_batch_size}")
        print(f"  Forward passes/image: {(total_patches + patch_batch_size - 1) // patch_batch_size}")
        print(f"{'='*60}\n")

    def exchange_data(self, **kwargs):
        patch_size = int(getattr(self.hparams, "patch_size", 128))
        self.criterion.loss_div = self.criterion.loss_div * match_ctf_to_rfft_shape(
            kwargs["datamodule"].ctf_mean(self.reconstructor.extractor.ctf),
            patch_size,
            patch_size,
        )

    def _shared_step(
        self, x1: torch.Tensor, x2: torch.Tensor, ctf: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        loss, denoised = self._patchwise_pass(
            x1=x1,
            x2=x2,
            ctf=ctf,
            backward=False,
            return_denoised=True,
        )
        return loss, denoised

    @staticmethod
    def _correct_ctf(data: torch.Tensor, ctf: torch.Tensor) -> torch.Tensor:
        f_data = torch.fft.rfft2(data, norm="forward") * ctf
        return torch.fft.irfft2(f_data, norm="forward")

    def _patchwise_pass(
        self,
        x1: torch.Tensor,
        x2: torch.Tensor,
        ctf: torch.Tensor,
        *,
        backward: bool,
        return_denoised: bool,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if x1.ndim != 5 or x2.ndim != 5:
            raise ValueError(
                f"Expected x1/x2 with shape (B,S,C,H,W), got {tuple(x1.shape)} and {tuple(x2.shape)}"
            )
        bsz, n_splits, c, h, w = x1.shape
        
        # Log patch configuration once at start of training
        self._log_patch_info(h, w)
        
        patch_size = int(getattr(self.hparams, "patch_size", 128))
        overlap = int(getattr(self.hparams, "patch_overlap", 16))
        patch_batch_size = int(
            getattr(
                self.hparams,
                "patch_batch_size" if backward else "eval_patch_batch_size",
                getattr(self.hparams, "patch_batch_size", 1),
            )
            or 1
        )
        if patch_batch_size <= 0:
            raise ValueError(f"patch_batch_size must be > 0, got {patch_batch_size}")
        y_starts, x_starts = _build_patch_grid_no_pad(h, w, patch_size, overlap)
        if not y_starts or not x_starts:
            raise ValueError(
                "No training patches fit image shape "
                f"{(h, w)} with patch_size={patch_size}, overlap={overlap}"
            )

        if ctf.ndim == 4:
            ctf = ctf.unsqueeze(2)
        if ctf.ndim != 5:
            raise ValueError(f"Expected ctf with 4 or 5 dimensions, got {ctf.ndim}")

        total_patches = bsz * len(y_starts) * len(x_starts)
        loss_sum = torch.zeros((), device=x1.device)
        output = None
        norm = None
        weight_expanded = None
        if return_denoised:
            output = torch.zeros_like(x1)
            norm = torch.zeros_like(x1)
            weight = _center_weight(patch_size, device=x1.device, dtype=x1.dtype)
            weight_expanded = weight.expand(1, c, patch_size, patch_size)
        patch_specs = [(batch_idx, y, x_start) for batch_idx in range(bsz) for y in y_starts for x_start in x_starts]
        for start_idx in range(0, len(patch_specs), patch_batch_size):
            chunk_specs = patch_specs[start_idx : start_idx + patch_batch_size]
            patch_in = torch.cat(
                [x1[b : b + 1, :, :, y : y + patch_size, x_start : x_start + patch_size] for b, y, x_start in chunk_specs],
                dim=0,
            )
            patch_target = torch.cat(
                [x2[b : b + 1, :, :, y : y + patch_size, x_start : x_start + patch_size] for b, y, x_start in chunk_specs],
                dim=0,
            )
            ctf_chunk = torch.cat([ctf[b : b + 1] for b, _, _ in chunk_specs], dim=0)
            ctf_chunk = match_ctf_to_rfft_shape(ctf_chunk, patch_size, patch_size)
            m_chunk = patch_in.shape[0]
            patch_in_flat = patch_in.reshape(m_chunk * n_splits, c, patch_size, patch_size)
            pred_flat = self.model(patch_in_flat)
            pred_flat = _crop_pred_to_patch_size(pred_flat, patch_size, c)
            pred = pred_flat.reshape(m_chunk, n_splits, c, patch_size, patch_size)
            if pred.shape != patch_target.shape:
                raise ValueError(
                    "Patch prediction and target shapes must match, got "
                    f"{tuple(pred.shape)} and {tuple(patch_target.shape)}"
                )
            if pred.shape[1] != n_splits:
                raise ValueError(
                    f"Expected split axis={n_splits} in prediction, got {pred.shape[1]}"
                )
            patch_loss = self.criterion(patch_target, pred, ctf_chunk)
            scaled_patch_loss = patch_loss * (m_chunk / total_patches)
            if backward:
                self.manual_backward(scaled_patch_loss)
                loss_sum = loss_sum + scaled_patch_loss.detach()
            else:
                loss_sum = loss_sum + scaled_patch_loss
            if return_denoised:
                assert output is not None and norm is not None and weight_expanded is not None
                for patch_idx, (batch_idx, y, x_start) in enumerate(chunk_specs):
                    pred_weighted = (pred[patch_idx] * weight_expanded).to(output.dtype)
                    output[batch_idx, :, :, y : y + patch_size, x_start : x_start + patch_size].add_(
                        pred_weighted
                    )
                    norm[batch_idx, :, :, y : y + patch_size, x_start : x_start + patch_size].add_(
                        weight_expanded
                    )
        denoised = None
        if return_denoised:
            assert output is not None and norm is not None
            mask = norm > 1e-6
            denoised = torch.where(mask, output / norm.clamp(min=1e-6), x1)
        return loss_sum, denoised

    def _patchwise_backward(self, x1: torch.Tensor, x2: torch.Tensor, ctf: torch.Tensor) -> torch.Tensor:
        loss, _ = self._patchwise_pass(
            x1=x1,
            x2=x2,
            ctf=ctf,
            backward=True,
            return_denoised=False,
        )
        return loss

    def training_step(
        self,
        batch: list[dict[str, torch.Tensor]],
        *args: object,
        **kwargs: object,
    ) -> torch.Tensor:
        optimizer = self.optimizers()
        optimizer.zero_grad(set_to_none=True)

        loss = torch.zeros((), device=self.device)
        for i in range(len(batch)):
            ctf_weights = self.reconstructor.extractor.ctf.get_fftw_image(**batch[i]["ctf_params"]).detach().float().abs()[:, None, ...]
            if self.hparams.ctf_premultiply:
                batch[i]["input"] = self._correct_ctf(batch[i]["input"], ctf_weights)
            loss = loss + self._patchwise_backward(x1=batch[i]["input"], x2=batch[i]["target"], ctf=ctf_weights)
        loss = loss / len(batch)

        optimizer.step()
        scheduler = self.lr_schedulers()
        if scheduler is not None and not isinstance(scheduler, ReduceLROnPlateau):
            scheduler.step()

        loss_value = float(loss.detach().cpu())
        self.log_dict(
            {f"train/loss": loss_value},
            on_step=True,
            on_epoch=False,
            prog_bar=True,
        )
        return loss.detach()

    def on_validation_epoch_end(self) -> None:
        super().on_validation_epoch_end()
        scheduler = self.lr_schedulers()
        if scheduler is None:
            return
        if isinstance(scheduler, ReduceLROnPlateau):
            val_loss = self.trainer.callback_metrics.get("val/loss")
            if val_loss is not None:
                scheduler.step(val_loss)