"""
CTF-aware split-wise variant of MultiNoise2NoiseCTFLightningModel.

Processes each frame independently through the model while properly
handling CTF weighting in the loss computation.
"""

import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau

from source._noise2noise.multinoise2noise_ctf import MultiNoise2NoiseCTFLightningModel


class MultiNoise2NoiseCTFSplitWiseLightningModel(MultiNoise2NoiseCTFLightningModel):
    """
    Split-wise CTF processing variant that handles large N (number of frames per group).
    
    Key differences from base class:
    - Processes frames through the model in smaller chunks (split_batch_size)
    - Accumulates outputs, then computes loss once with CTF weighting
    - Supports gradient checkpointing for additional memory savings
    
    New hyperparameters:
    - split_batch_size: Number of splits to process at once (default: 1)
    - use_gradient_checkpointing: Whether to use gradient checkpointing (default: False)
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        # self.automatic_optimization = False
        self._split_info_logged = False

    def _log_split_info(self, n_splits: int) -> None:
        """Log split configuration once at start of training."""
        if self._split_info_logged:
            return
        self._split_info_logged = True
        
        split_batch_size = int(getattr(self.hparams, "split_batch_size", 1))
        use_checkpointing = bool(getattr(self.hparams, "use_gradient_checkpointing", False))
        
        n_forward_passes = (n_splits + split_batch_size - 1) // split_batch_size
        
        print(f"\n{'='*60}")
        print(f"CTF SPLIT-WISE CONFIGURATION")
        print(f"{'='*60}")
        print(f"  Number of splits (N):     {n_splits}")
        print(f"  Split batch size:         {split_batch_size}")
        print(f"  Forward passes per batch: {n_forward_passes}")
        print(f"  Gradient checkpointing:   {use_checkpointing}")
        print(f"{'='*60}\n")

    def _forward_splitwise(self, x1: torch.Tensor) -> torch.Tensor:
        """
        Forward pass processing splits in chunks to reduce memory.
        
        Args:
            x1: Input tensor of shape (B, S, C, H, W)
            
        Returns:
            Denoised output of shape (B, S, C, H, W)
        """
        B, S, C, H, W = x1.shape
        split_batch_size = int(getattr(self.hparams, "split_batch_size", 1))
        use_checkpointing = bool(getattr(self.hparams, "use_gradient_checkpointing", False))
        
        outputs = []
        
        for start_s in range(0, S, split_batch_size):
            end_s = min(start_s + split_batch_size, S)
            chunk_size = end_s - start_s
            
            # Extract chunk: (B, chunk_size, C, H, W)
            x1_chunk = x1[:, start_s:end_s]
            
            # Flatten for model: (B * chunk_size, C, H, W)
            x1_flat = x1_chunk.reshape(B * chunk_size, C, H, W)
            
            # Forward through model
            if use_checkpointing and self.training:
                out_flat = torch.utils.checkpoint.checkpoint(
                    self.model, x1_flat, use_reentrant=False
                )
            else:
                out_flat = self.model(x1_flat)
            
            # Handle model output shape
            if out_flat.shape[1] != C:
                out_flat = out_flat.expand(-1, C, -1, -1)
            
            # Reshape back: (B, chunk_size, C, H, W)
            out_chunk = out_flat.reshape(B, chunk_size, C, H, W)
            outputs.append(out_chunk)
        
        # Concatenate all outputs: (B, S, C, H, W)
        return torch.cat(outputs, dim=1)

    def _shared_step(self, x1: torch.Tensor, x2: torch.Tensor, ctf: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Override shared step to use split-wise forward."""
        if x1.ndim != 5:
            raise ValueError(f"Expected 5D input (B,S,C,H,W), got shape {x1.shape}")
        
        self._log_split_info(x1.shape[1])
        
        denoised = self._forward_splitwise(x1)
        loss = self.criterion(x2, denoised, ctf)
        return loss, denoised

    @staticmethod
    def _correct_ctf(data: torch.Tensor, ctf: torch.Tensor) -> torch.Tensor:
        f_data = torch.fft.rfft2(data, norm="forward") * ctf
        return torch.fft.irfft2(f_data, norm="forward")

    def training_step(
        self,
        batch: list[dict[str, torch.Tensor]],
        *args: object,
        **kwargs: object,
    ) -> torch.Tensor:
        # optimizer = self.optimizers()
        # optimizer.zero_grad(set_to_none=True)

        total_loss = torch.zeros((), device=self.device)
        
        for i in range(len(batch)):
            ctf_weights = self.reconstructor.extractor.ctf.get_fftw_image(
                **batch[i]["ctf_params"]
            ).detach().float().abs()[:, None, ...]
            
            if self.hparams.ctf_premultiply:
                batch[i]["input"] = self._correct_ctf(batch[i]["input"], ctf_weights)
            
            loss, _ = self._shared_step(
                x1=batch[i]["input"], 
                x2=batch[i]["target"], 
                ctf=ctf_weights
            )
            scaled_loss = loss / len(batch)
            # self.manual_backward(scaled_loss)
            total_loss = total_loss + scaled_loss#.detach()

        # optimizer.step()

        # scheduler = self.lr_schedulers()
        # if scheduler is not None and not isinstance(scheduler, ReduceLROnPlateau):
        #     scheduler.step()

        self.log_dict(
            {"train/loss": float(total_loss.cpu())},
            on_step=True,
            on_epoch=False,
            prog_bar=True,
        )
        return total_loss

    # def on_validation_epoch_end(self) -> None:
    #     super().on_validation_epoch_end()
    #     scheduler = self.lr_schedulers()
    #     if scheduler is None:
    #         return
    #     if isinstance(scheduler, ReduceLROnPlateau):
    #         val_loss = self.trainer.callback_metrics.get("val/loss")
    #         if val_loss is not None:
    #             scheduler.step(val_loss)


class MultiNoise2NoiseCTFSplitWiseAccumLightningModel(MultiNoise2NoiseCTFLightningModel):
    """
    Gradient accumulation CTF variant that processes splits one at a time.
    
    This is the most memory-efficient approach but requires loss functions
    that support split-wise computation.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.automatic_optimization = False
        self._split_info_logged = False

    def _log_split_info(self, n_splits: int) -> None:
        if self._split_info_logged:
            return
        self._split_info_logged = True
        
        print(f"\n{'='*60}")
        print(f"CTF SPLIT-WISE GRADIENT ACCUMULATION")
        print(f"{'='*60}")
        print(f"  Number of splits (N):     {n_splits}")
        print(f"  Processing one split at a time with gradient accumulation")
        print(f"{'='*60}\n")

    @staticmethod
    def _correct_ctf(data: torch.Tensor, ctf: torch.Tensor) -> torch.Tensor:
        f_data = torch.fft.rfft2(data, norm="forward") * ctf
        return torch.fft.irfft2(f_data, norm="forward")

    def _accumulate_backward(self, x1: torch.Tensor, x2: torch.Tensor, 
                              ctf: torch.Tensor) -> torch.Tensor:
        """
        Process each split independently and accumulate gradients with CTF weighting.
        """
        if x1.ndim != 5 or x2.ndim != 5:
            raise ValueError(f"Expected 5D tensors, got {x1.shape} and {x2.shape}")
        
        B, S, C, H, W = x1.shape
        self._log_split_info(S)
        
        # Ensure CTF has correct dimensions
        if ctf.ndim == 4:
            ctf = ctf.unsqueeze(2)  # (B, 1, 1, H, W//2+1)
        
        total_loss = torch.zeros((), device=x1.device)
        
        for s in range(S):
            # Extract single split: (B, 1, C, H, W)
            x1_s = x1[:, s:s+1]
            x2_s = x2[:, s:s+1]
            
            # Flatten: (B, C, H, W)
            x1_flat = x1_s.squeeze(1)
            
            # Forward
            out_flat = self.model(x1_flat)
            
            # Handle output shape
            if out_flat.dim() == 4 and out_flat.shape[1] == 1:
                out_s = out_flat.unsqueeze(1)
            elif out_flat.dim() == 4:
                out_s = out_flat.unsqueeze(1)
            else:
                out_s = out_flat
            
            if out_s.shape != x2_s.shape:
                out_s = out_s.expand_as(x2_s)
            
            # Compute loss for this split with CTF
            if hasattr(self.criterion, 'forward_split'):
                loss_s = self.criterion.forward_split(
                    x2_s, out_s, ctf, split_idx=s, total_splits=S
                )
            else:
                # Fallback: use standard forward (will compute full CTF-weighted loss)
                loss_s = self.criterion(x2_s, out_s, ctf)
            
            # Scale and accumulate gradient
            scaled_loss = loss_s / S
            self.manual_backward(scaled_loss)
            
            total_loss = total_loss + scaled_loss.detach()
            
            del x1_s, x2_s, x1_flat, out_flat, out_s, loss_s
        
        return total_loss

    def training_step(
        self,
        batch: list[dict[str, torch.Tensor]],
        *args: object,
        **kwargs: object,
    ) -> torch.Tensor:
        optimizer = self.optimizers()
        optimizer.zero_grad(set_to_none=True)

        total_loss = torch.zeros((), device=self.device)
        
        for i in range(len(batch)):
            ctf_weights = self.reconstructor.extractor.ctf.get_fftw_image(
                **batch[i]["ctf_params"]
            ).detach().float().abs()[:, None, ...]
            
            if self.hparams.ctf_premultiply:
                batch[i]["input"] = self._correct_ctf(batch[i]["input"], ctf_weights)
            
            loss = self._accumulate_backward(
                x1=batch[i]["input"], 
                x2=batch[i]["target"], 
                ctf=ctf_weights
            )
            total_loss = total_loss + loss / len(batch)

        optimizer.step()

        scheduler = self.lr_schedulers()
        if scheduler is not None and not isinstance(scheduler, ReduceLROnPlateau):
            scheduler.step()

        self.log_dict(
            {"train/loss": float(total_loss.cpu())},
            on_step=True,
            on_epoch=False,
            prog_bar=True,
        )
        return total_loss

    def on_validation_epoch_end(self) -> None:
        super().on_validation_epoch_end()
        scheduler = self.lr_schedulers()
        if scheduler is None:
            return
        if isinstance(scheduler, ReduceLROnPlateau):
            val_loss = self.trainer.callback_metrics.get("val/loss")
            if val_loss is not None:
                scheduler.step(val_loss)
