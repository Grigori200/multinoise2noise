from __future__ import annotations

from dataclasses import dataclass
import math
from typing import List, Sequence, Tuple

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class PatchGrid:
    y_starts: Sequence[int]
    x_starts: Sequence[int]
    padded_h: int
    padded_w: int


def _build_starts(length: int, patch_size: int, stride: int) -> List[int]:
    if length <= patch_size:
        return [0]
    n_steps = int(math.ceil((length - patch_size) / stride))
    return [i * stride for i in range(n_steps + 1)]


def _build_patch_grid(h: int, w: int, patch_size: int, overlap: int) -> PatchGrid:
    if patch_size <= 0:
        raise ValueError(f"patch_size must be > 0, got {patch_size}")
    if overlap < 0:
        raise ValueError(f"overlap must be >= 0, got {overlap}")
    if overlap >= patch_size:
        raise ValueError(f"overlap ({overlap}) must be smaller than patch_size ({patch_size})")

    stride = patch_size - overlap
    y_starts = _build_starts(h, patch_size, stride)
    x_starts = _build_starts(w, patch_size, stride)
    padded_h = y_starts[-1] + patch_size
    padded_w = x_starts[-1] + patch_size
    return PatchGrid(y_starts=y_starts, x_starts=x_starts, padded_h=padded_h, padded_w=padded_w)


def _build_patch_grid_no_pad(
    h: int, w: int, patch_size: int, overlap: int
) -> Tuple[List[int], List[int]]:
    """
    Returns (y_starts, x_starts) where every patch fits within the image:
    y + patch_size <= h and x + patch_size <= w. No padding is used.
    """
    if patch_size <= 0:
        raise ValueError(f"patch_size must be > 0, got {patch_size}")
    if overlap < 0:
        raise ValueError(f"overlap must be >= 0, got {overlap}")
    if overlap >= patch_size:
        raise ValueError(f"overlap ({overlap}) must be smaller than patch_size ({patch_size})")

    stride = patch_size - overlap
    n_y = max(0, (h - patch_size) // stride + 1) if h >= patch_size else 0
    n_x = max(0, (w - patch_size) // stride + 1) if w >= patch_size else 0
    y_starts = [i * stride for i in range(n_y)]
    x_starts = [i * stride for i in range(n_x)]
    return y_starts, x_starts


def _center_weight(patch_size: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    # Separable triangular ramp with maximum in patch center.
    coord = torch.linspace(-1.0, 1.0, steps=patch_size, device=device, dtype=dtype)
    ramp = 1.0 - coord.abs()
    ramp = torch.clamp(ramp, min=1e-3)
    weight_2d = ramp[:, None] * ramp[None, :]
    return weight_2d[None, None, ...]


def _crop_pred_to_patch_size(
    pred: torch.Tensor, patch_size: int, c: int
) -> torch.Tensor:
    """Center-crop pred to (patch_size, patch_size) if model returned larger."""
    if pred.shape[1] != c:
        raise ValueError(
            f"Model channel mismatch: input C={c}, predicted C={pred.shape[1]}"
        )
    pred_h, pred_w = int(pred.shape[-2]), int(pred.shape[-1])
    if (pred_h, pred_w) == (patch_size, patch_size):
        return pred
    if pred_h >= patch_size and pred_w >= patch_size:
        y0 = (pred_h - patch_size) // 2
        x0 = (pred_w - patch_size) // 2
        return pred[..., y0 : y0 + patch_size, x0 : x0 + patch_size]
    raise ValueError(
        "Patch model output is smaller than requested patch_size. "
        f"Requested {(patch_size, patch_size)}, got {(pred_h, pred_w)}"
    )


def infer_with_overlap(
    model: torch.nn.Module,
    x: torch.Tensor,
    patch_size: int = 16,
    overlap: int = 0,
    patch_batch_size: int | None = 1,
) -> torch.Tensor:
    """
    Run patch-wise model inference with overlap and center-weighted blending.

    Memory-efficient: no padding, view-based patch extraction, in-place accumulation
    into a preallocated output buffer.

    Args:
        model: Denoising model operating on (N, C, H, W).
        x: Input tensor of shape (N, C, H, W).
        patch_size: Total patch side length used for model inference.
            The overlap region is included within this size.
        overlap: Overlap between adjacent patches.
        patch_batch_size: Optional mini-batch size for patch inference.

    Returns:
        Tensor with the same shape as x.
    """
    if x.ndim != 4:
        raise ValueError(f"Expected x with shape (N, C, H, W), got {tuple(x.shape)}")

    n, c, h, w = x.shape
    stride = patch_size - overlap
    y_starts, x_starts = _build_patch_grid_no_pad(h, w, patch_size, overlap)

    # Preallocate output and norm
    output = torch.zeros(n, c, h, w, device=x.device, dtype=x.dtype)
    norm = torch.zeros(n, c, h, w, device=x.device, dtype=x.dtype)

    weight = _center_weight(patch_size, device=x.device, dtype=x.dtype)
    # Expand for (1, C, P, P) to match pred * weight
    weight_expanded = weight.expand(1, c, patch_size, patch_size)

    if not y_starts or not x_starts:
        # No patches fit (e.g. image smaller than patch_size); return input copy
        return x.clone()

    batch_size = patch_batch_size if (patch_batch_size is not None and patch_batch_size > 0) else 1

    # Build flat list of (img_idx, y, x) for iteration
    patch_specs: List[Tuple[int, int, int]] = []
    for img_idx in range(n):
        for y in y_starts:
            for x_start in x_starts:
                patch_specs.append((img_idx, y, x_start))

    if batch_size == 1:
        # Zero-copy path: use view for each patch, no allocation for inputs
        for img_idx, y, x_start in patch_specs:
            patch = x[img_idx : img_idx + 1, :, y : y + patch_size, x_start : x_start + patch_size]
            pred = model(patch)
            pred = _crop_pred_to_patch_size(pred, patch_size, c)
            pred_weighted = (pred * weight_expanded).to(output.dtype)
            output[img_idx, :, y : y + patch_size, x_start : x_start + patch_size].add_(
                pred_weighted[0]
            )
            norm[img_idx, :, y : y + patch_size, x_start : x_start + patch_size].add_(
                weight_expanded[0]
            )
    else:
        # Batched path: collect patches, stack, model, scatter
        for i in range(0, len(patch_specs), batch_size):
            batch_specs = patch_specs[i : i + batch_size]
            patches = torch.stack(
                [
                    x[img_idx, :, y : y + patch_size, x_start : x_start + patch_size]
                    for img_idx, y, x_start in batch_specs
                ],
                dim=0,
            )
            preds = model(patches)
            preds = _crop_pred_to_patch_size(preds, patch_size, c)
            for j, (img_idx, y, x_start) in enumerate(batch_specs):
                pred_weighted = (preds[j : j + 1] * weight_expanded).to(output.dtype)
                output[img_idx, :, y : y + patch_size, x_start : x_start + patch_size].add_(
                    pred_weighted[0]
                )
                norm[img_idx, :, y : y + patch_size, x_start : x_start + patch_size].add_(
                    weight_expanded[0]
                )

    # Normalize; for uncovered pixels (norm=0) copy from input (Option B)
    mask = norm > 1e-6
    output = torch.where(mask, output / norm.clamp(min=1e-6), x)
    return output


def match_ctf_to_rfft_shape(ctf: torch.Tensor, spatial_h: int, spatial_w: int) -> torch.Tensor:
    """
    Ensure ctf tensor last two dimensions match torch.rfft2 output shape.

    The target frequency shape is (spatial_h, spatial_w // 2 + 1).
    If input is larger, it is cropped; if smaller, it is zero-padded.
    """
    target_h = int(spatial_h)
    target_wf = int(spatial_w // 2 + 1)
    if ctf.shape[-2] == target_h and ctf.shape[-1] == target_wf:
        return ctf

    out = ctf
    if out.shape[-2] > target_h:
        out = out[..., :target_h, :]
    if out.shape[-1] > target_wf:
        out = out[..., :, :target_wf]

    pad_h = max(target_h - out.shape[-2], 0)
    pad_w = max(target_wf - out.shape[-1], 0)
    if pad_h > 0 or pad_w > 0:
        out = F.pad(out, (0, pad_w, 0, pad_h), mode="constant", value=0.0)
    return out
