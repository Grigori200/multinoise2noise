import torch


class CTF_MSELoss(torch.nn.Module):
    """
    Mean Squared Error loss function with CTF weighting for electron microscopy images.

    Applies CTF weighting to compute the loss in the Fourier domain.

    Attributes:
        n (int): Power to raise CTF to in the loss.
        correct_frequencies (bool): Whether to apply frequency correction.
        loss_div (torch.Tensor): Frequency correction divisor.
    """
    def __init__(self, n: int = 1, correct_frequencies: bool = True, **kwargs):
        """
        Initialize the CTF_MSELoss module.

        Args:
            n (int, optional): Power to raise CTF to in the loss. Defaults to 1.
            correct_frequencies (bool, optional): Whether to apply frequency correction. Defaults to True.
            **kwargs: Additional keyword arguments.
        """
        super().__init__()
        self.n = n
        self.correct_frequencies = correct_frequencies
        self.register_buffer("loss_div", torch.ones((1,1,1,1,1)), persistent=False)

    def forward_split(self, preds: torch.Tensor, target: torch.Tensor, ctf: torch.Tensor,
                      split_idx: int, total_splits: int) -> torch.Tensor:
        """
        Compute CTF-weighted MSE loss for a single split/frame.
        
        This method enables gradient accumulation by computing loss per-split.
        The caller should accumulate gradients and divide by total_splits.
        
        Args:
            preds: Noisy target images for this split, shape (B, 1, C, H, W)
            target: Model output (denoised) for this split, shape (B, 1, C, H, W)
            ctf: CTF weighting tensor for this split
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
        
        loss = ((f_preds.real - f_target.real) ** 2 + 
                (f_preds.imag - f_target.imag) ** 2) * (ctf ** self.n)
        
        if self.correct_frequencies:
            loss = loss / self.loss_div
        
        return loss.mean()
    
    def forward(self, preds: torch.Tensor, target: torch.Tensor, ctf: torch.Tensor) -> torch.Tensor:
        """
        Compute the dose-weighted CTF MSE loss between predictions and targets.

        Args:
            preds (torch.Tensor): Predicted images (real or complex tensor).
            target (torch.Tensor): Target images (real or complex tensor).
            ctf (torch.Tensor): CTF weighting tensor.

        Returns:
            torch.Tensor: Scalar loss value.
        """
        if preds.dtype == torch.complex64:
            f_preds = preds
        else:
            f_preds = torch.fft.rfft2(preds, norm="ortho")
        if target.dtype == torch.complex64:
            f_target = target
        else:
            f_target = torch.fft.rfft2(target, norm="ortho")
        loss = ((f_preds.real - f_target.real) ** 2 + 
                (f_preds.imag - f_target.imag) ** 2) * (ctf ** self.n)
        if self.correct_frequencies:
            loss = loss / self.loss_div
        loss = loss.mean()
        return loss
