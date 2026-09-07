from __future__ import annotations
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

@torch.no_grad()
def extract_embeddings(
    model: torch.nn.Module,
    signals: np.ndarray,
    *,
    device: torch.device,
    batch_size: int = 128,
) -> np.ndarray:
    loader = DataLoader(
        TensorDataset(torch.from_numpy(np.asarray(signals, dtype=np.float32)).unsqueeze(1)),
        batch_size=batch_size,
        shuffle=False,
    )
    rows = []
    for (batch,) in loader:
        rows.append(
            model.global_embedding(batch.to(device), pool="mean").cpu().numpy()
        )
    values = np.concatenate(rows)
    if not np.all(np.isfinite(values)):
        raise ValueError("Embedding extraction produced non-finite values")
    return values

