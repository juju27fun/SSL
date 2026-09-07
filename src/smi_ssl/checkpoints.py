"""Strict loading for the original dictionary checkpoint format."""
from pathlib import Path
import torch
from .configuration import make_model


def load_checkpoint(path: Path, device: str = 'cpu'):
    payload = torch.load(path, map_location=device, weights_only=True)
    if not isinstance(payload, dict) or not {'model_state_dict','config','epoch'}.issubset(payload):
        raise ValueError('Expected model_state_dict, config and epoch in checkpoint')
    model = make_model(payload['config'])
    model.load_state_dict(payload['model_state_dict'], strict=True)
    return model.to(device).eval(), payload
