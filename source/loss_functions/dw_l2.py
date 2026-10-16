import torch

from source.reconstruction.dose_weighter import DoseWeighter


class DW_MSELoss(torch.nn.Module):
    """
    Mean Squared Error loss function with dose weighting for electron microscopy images.

    This class applies dose weighting to the predicted and target images in the Fourier domain
    before computing the loss. Optionally, regularization can be applied to encourage similarity
    between frames.

    Attributes:
        dose_weighter (DoseWeighter): Module for dose weighting.
        correct_frequencies (bool): Whether to apply frequency correction.
        regularization (float): Regularization strength for inter-frame similarity.
        frames_weight_sum (torch.Tensor): Sum of absolute dose weights across frames.
    """
    def __init__(
        self,
        dose_weighter: DoseWeighter,
        correct_frequencies: bool = True,
        regularization: float = 0.,
        prediction_dose_weighting: bool = True,
        target_dose_weighting: bool = False,
        undo_target_dose_weighting: bool = False,
        loss_dose_weighting: bool = False,
        target_weight_floor: float = 1e-3,
        *args,
        **kwargs,
    ):
        """
        Initialize the DW_MSELoss module.

        Args:
            dose_weighter (DoseWeighter): Dose weighting module.
            correct_frequencies (bool, optional): Whether to apply frequency correction. Defaults to True.
            regularization (float, optional): Regularization strength for inter-frame similarity. Defaults to 0.
            prediction_dose_weighting (bool, optional): Multiply the prediction spectrum by the frame
                dose weights. Defaults to True (historical behaviour).
            target_dose_weighting (bool, optional): Multiply the target spectrum by the frame dose
                weights. Defaults to False.
            undo_target_dose_weighting (bool, optional): Divide the target spectrum by the frame dose
                weights (clamped from below by ``target_weight_floor``) so that the loss compares
                against the unweighted frame. Defaults to False.
            loss_dose_weighting (bool, optional): Multiply the per-pixel squared spectral error by
                the frame dose weights. Defaults to False.
            target_weight_floor (float, optional): Lower bound on the dose weights used when undoing
                target dose weighting. Defaults to 1e-3.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.
        """
        super().__init__()
        if target_dose_weighting and undo_target_dose_weighting:
            raise ValueError(
                "target_dose_weighting and undo_target_dose_weighting are mutually exclusive"
            )
        if target_weight_floor <= 0:
            raise ValueError(f"target_weight_floor must be positive, got {target_weight_floor}")
        self.correct_frequencies = correct_frequencies
        self.regularization = regularization
        self.prediction_dose_weighting = prediction_dose_weighting
        self.target_dose_weighting = target_dose_weighting
        self.undo_target_dose_weighting = undo_target_dose_weighting
        self.loss_dose_weighting = loss_dose_weighting
        self.target_weight_floor = target_weight_floor
        self.dose_weighter = dose_weighter

        self.dose_weighter.weight = self.dose_weighter.weight[None, :, None, ...]
        self.register_buffer("frames_weight_sum", 
                             self.dose_weighter.weight.abs().sum(dim=-4, keepdim=True), 
                             persistent=False)

    def forward_split(self, preds: torch.Tensor, target: torch.Tensor, 
                      split_idx: int, total_splits: int) -> torch.Tensor:
        """
        Compute dose-weighted MSE loss for a single split/frame.
        
        This method enables gradient accumulation by computing loss per-split
        with proper dose weighting. The caller should accumulate gradients
        and divide by total_splits.
        
        Note: Parameter naming follows the original forward() convention where
        'preds' receives the noisy target and 'target' receives model output.
        
        Args:
            preds: Noisy target images for this split, shape (B, 1, C, H, W)
            target: Model output (denoised) for this split, shape (B, 1, C, H, W)
            split_idx: Index of current split (0-indexed)
            total_splits: Total number of splits (N)
            
        Returns:
            Scalar loss value for this split (not yet normalized by total_splits)
        """
        if preds.dtype == torch.complex64:
            f_preds = preds
        else:
            f_preds = torch.fft.rfft2(preds, norm="ortho")
        
        if target.dtype == torch.complex64:
            f_target = target
        else:
            f_target = torch.fft.rfft2(target, norm="ortho")
        
        # Get weight for this specific split
        if split_idx < self.dose_weighter.weight.shape[-4]:
            split_weight = self.dose_weighter.weight[:, split_idx:split_idx+1, ...]
        else:
            split_weight = self.frames_weight_sum
        
        f_preds_weighted = f_preds * split_weight
        
        loss = (f_preds_weighted.real - f_target.real) ** 2 + \
               (f_preds_weighted.imag - f_target.imag) ** 2
        
        if self.correct_frequencies:
            loss = loss / self.frames_weight_sum
        
        return loss.mean()
    
    def frame_weights(self, n_splits: int) -> torch.Tensor:
        """
        Return per-frame dose weights when the split dimension matches the number of frames,
        otherwise the summed weights (used for single-split full-dose evaluation).

        Args:
            n_splits (int): Size of the split dimension (dim -4) of the compared tensors.

        Returns:
            torch.Tensor: Weights broadcastable to (B, S, C, H, W // 2 + 1).
        """
        if n_splits == self.dose_weighter.weight.shape[-4]:
            return self.dose_weighter.weight
        return self.frames_weight_sum

    def forward(self, target: torch.Tensor, preds: torch.Tensor) -> torch.Tensor:
        """
        Compute the dose-weighted MSE loss between predictions and targets.

        Args:
            preds (torch.Tensor): Predicted images (real or complex tensor).
            target (torch.Tensor): Target images (real or complex tensor).

        Returns:
            torch.Tensor: Scalar loss value.
        """
        if preds.dtype == torch.complex64:
            f_preds = preds
        else:
            f_preds = torch.fft.rfft2(preds, norm="ortho")

        # The weights are float64; results are cast back to the operand dtype (as the
        # in-place multiply does) because mse_loss cannot backpropagate through mixed
        # float32/float64 inputs.
        weight = self.frame_weights(f_preds.shape[-4])
        if self.prediction_dose_weighting:
            f_preds *= weight
        if target.dtype == torch.complex64:
            f_target = target
        else:
            f_target = torch.fft.rfft2(target, norm="ortho")
        if self.target_dose_weighting:
            f_target = (f_target * weight).to(f_target.dtype)
        if self.undo_target_dose_weighting:
            f_target = (f_target / weight.clamp_min(self.target_weight_floor)).to(
                f_target.dtype
            )
        loss = torch.nn.functional.mse_loss(f_preds.imag, f_target.imag, reduction='none') + torch.nn.functional.mse_loss(f_preds.real, f_target.real, reduction='none')
        if self.loss_dose_weighting:
            loss = (loss * weight).to(loss.dtype)

        if self.correct_frequencies:
            loss = loss / self.frames_weight_sum
        loss = loss.mean()
        if self.regularization > 0.:
            for i in range(f_preds.shape[-4]):
                for j in range(i+1, f_preds.shape[-4]):
                    loss += (torch.nn.functional.mse_loss(f_preds[..., i, :, :, :].real, f_preds[..., j, :, :, :].real, reduction='mean') + torch.nn.functional.mse_loss(f_preds[..., i, :, :, :].imag, f_preds[..., j, :, :, :].imag, reduction='mean')) * self.regularization / (f_preds.shape[-4] * (f_preds.shape[-4] - 1) * 2)
        return loss
