from __future__ import annotations
import math
import numpy as np
RAW_CROP_LENGTH=4096
INPUT_LENGTH=512

def centered_reflect_crop(signal: np.ndarray, center: int, length: int = RAW_CROP_LENGTH) -> np.ndarray:
    values = np.asarray(signal, dtype=np.float32).reshape(-1)
    if values.size < 2:
        raise ValueError("reflect cropping requires a signal with at least two samples")
    left = int(center) - int(length) // 2
    right = left + int(length)
    pad_left = max(0, -left)
    pad_right = max(0, right - values.size)
    crop = values[max(0, left) : min(values.size, right)]
    if pad_left or pad_right:
        crop = np.pad(crop, (pad_left, pad_right), mode="reflect")
    if crop.size != length:
        raise ValueError(f"crop length {crop.size} differs from {length}")
    return np.asarray(crop, dtype=np.float32)



def prepare_event_signal(
    signal: np.ndarray,
    center_index: int,
    *,
    center_offset: int = 0,
    amplitude_scale: float = 1.0,
    noise_snr_db: float | None = None,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    crop = centered_reflect_crop(signal, center_index + center_offset)
    decimated = crop.reshape(INPUT_LENGTH, RAW_CROP_LENGTH // INPUT_LENGTH).mean(axis=1)
    decimated = np.asarray(decimated * float(amplitude_scale), dtype=np.float32)
    if noise_snr_db is not None:
        generator = rng if rng is not None else np.random.default_rng()
        power = float(np.mean(np.square(decimated, dtype=np.float64)))
        if power > 0.0:
            noise_std = math.sqrt(power / (10.0 ** (float(noise_snr_db) / 10.0)))
            decimated += generator.normal(0.0, noise_std, decimated.shape).astype(np.float32)
    std = float(decimated.std())
    normalized = (decimated - float(decimated.mean())) / (std + 1.0e-8)
    if normalized.shape != (INPUT_LENGTH,) or not np.isfinite(normalized).all():
        raise ValueError("prepared event signal is invalid")
    return normalized.astype(np.float32, copy=False)

