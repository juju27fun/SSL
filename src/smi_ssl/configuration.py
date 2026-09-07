"""Portable exact-preset loader; architecture ownership belongs to Models."""
from importlib.resources import files
from pathlib import Path
import json
from typing import Any
from smi_models import UNet1DConfig, UNet1DReconstructor


def load_config(path: Path | None = None) -> dict[str, Any]:
    source = path if path is not None else files("smi_ssl").joinpath("presets/unet_mad4d.json")
    config = json.loads(source.read_text())
    if config["model"]["architecture"] != "unet1d_reconstructor":
        raise ValueError("This extracted SSL pipeline owns U-Net reconstruction only")
    if config["data"]["input_length"] != 4096 or config["data"]["sampling_frequency_hz"] != 2000000:
        raise ValueError("The retained MAD4d protocol requires 4096 samples at 2 MHz")
    if config["data"]["normalization"] != "window_zscore":
        raise ValueError("The retained protocol uses per-window z-score normalization")
    if config["masking"]["training_policy"] not in {"P25", "CYCLIC25"}:
        raise ValueError("Unknown masking policy")
    if config["training"]["checkpoint_selection"] != "fixed_final":
        raise ValueError("The retained experiment selects the fixed final checkpoint")
    return config


def make_model(config: dict[str, Any]) -> UNet1DReconstructor:
    model = config["model"]
    if model["architecture"] != "unet1d_reconstructor":
        raise ValueError("Unsupported reconstruction architecture")
    return UNet1DReconstructor(UNet1DConfig(
        input_length=int(config["data"]["input_length"]),
        channels=tuple(int(v) for v in model["channels"]),
        kernel_size=int(model["kernel_size"]),
        bottleneck_hidden_channels=int(model["bottleneck_hidden_channels"]),
        activation=str(model["activation"]),
    ))
