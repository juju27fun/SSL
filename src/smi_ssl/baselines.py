from __future__ import annotations
import numpy as np
import torch

def interpolation_baseline(signals: torch.Tensor, hidden_masks: torch.Tensor) -> torch.Tensor:
    outputs = []
    for signal, hidden in zip(signals[:, 0].cpu().numpy(), hidden_masks.cpu().numpy()):
        indices = np.arange(signal.size)
        visible = ~hidden
        if int(np.count_nonzero(visible)) < 2:
            reconstructed = np.zeros_like(signal)
        else:
            reconstructed = signal.copy()
            reconstructed[hidden] = np.interp(indices[hidden], indices[visible], signal[visible])
        outputs.append(reconstructed)
    return torch.from_numpy(np.stack(outputs)).to(signals.device).unsqueeze(1)



def nearest_baseline(signals: torch.Tensor, hidden_masks: torch.Tensor) -> torch.Tensor:
    outputs = []
    for signal, hidden in zip(signals[:, 0].cpu().numpy(), hidden_masks.cpu().numpy()):
        indices = np.arange(signal.size)
        visible_indices = indices[~hidden]
        if visible_indices.size == 0:
            outputs.append(np.zeros_like(signal))
            continue
        insertion = np.searchsorted(visible_indices, indices)
        left = visible_indices[np.clip(insertion - 1, 0, visible_indices.size - 1)]
        right = visible_indices[np.clip(insertion, 0, visible_indices.size - 1)]
        nearest = np.where(indices - left <= right - indices, left, right)
        reconstructed = signal.copy()
        reconstructed[hidden] = signal[nearest[hidden]]
        outputs.append(reconstructed)
    return torch.from_numpy(np.stack(outputs)).to(signals.device).unsqueeze(1)



def visible_mean_baseline(signals: torch.Tensor, hidden_masks: torch.Tensor) -> torch.Tensor:
    """Predict every sample with the per-signal mean of the visible context."""
    if signals.ndim != 3 or signals.shape[1] != 1:
        raise ValueError("signals must have shape (batch, 1, length)")
    if hidden_masks.shape != signals.shape[:1] + signals.shape[2:]:
        raise ValueError("hidden_masks must have shape (batch, length)")
    visible = (~hidden_masks.to(signals.device)).unsqueeze(1)
    count = visible.sum(dim=-1, keepdim=True).clamp_min(1)
    mean = (signals * visible).sum(dim=-1, keepdim=True) / count
    return mean.expand_as(signals)

