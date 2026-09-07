"""Explicit portable event-array contract, independent of workspace registries."""
from pathlib import Path
import csv
import hashlib

import numpy as np
import torch
from torch.utils.data import Dataset

from .decimation import normalize_signal

SPLITS = ('train', 'val', 'real_val')


class EventArrays(Dataset):
    def __init__(self, payload: dict[str, np.ndarray], split: str, limit: int | None = None, per_class_limit: int | None = None):
        if split not in SPLITS:
            raise ValueError(f'Unsupported split: {split}')
        self.payload = payload
        self.indices = np.flatnonzero(payload['split'] == split)
        if per_class_limit is not None:
            if per_class_limit <= 0:
                raise ValueError("per_class_limit must be positive")
            labels = payload["labels"][self.indices]
            self.indices = np.concatenate([self.indices[labels == label][:per_class_limit]
                                           for label in np.unique(labels)])
        if limit is not None:
            self.indices = self.indices[:limit]
        if not self.indices.size:
            raise ValueError(f'Empty split: {split}')

    def __len__(self):
        return self.indices.size

    def __getitem__(self, index):
        i = self.indices[index]
        values = normalize_signal(self.payload['signals'][i], mode='window_zscore')
        return {'signal': torch.from_numpy(values).unsqueeze(0),
                'event_mask': torch.from_numpy(self.payload['event_masks'][i].copy()),
                'sample_index': int(i), 'sample_id': str(self.payload['ids'][i]),
                'class_name': str(self.payload['labels'][i])}


def load_arrays(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        payload = {name: archive[name] for name in archive.files}
    return validate_arrays(payload)


def validate_arrays(payload: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    required = {'signals', 'event_masks', 'split', 'ids', 'groups', 'labels', 'sampling_frequency_hz'}
    if not required.issubset(payload):
        raise ValueError(f'Missing array keys: {sorted(required-set(payload))}')
    signals = np.asarray(payload['signals'], dtype=np.float32)
    if signals.ndim != 2 or signals.shape[1] != 4096 or not np.isfinite(signals).all():
        raise ValueError('signals must be a finite (N, 4096) matrix')
    n = len(signals)
    if payload['event_masks'].shape != signals.shape or payload['event_masks'].dtype != np.bool_:
        raise ValueError('event_masks must be a boolean matrix matching signals')
    if np.any(payload['event_masks'].sum(axis=1) == 0) or np.any(payload['event_masks'].all(axis=1)):
        raise ValueError('Every sample must have event and background support')
    if float(payload['sampling_frequency_hz']) != 2_000_000:
        raise ValueError('Expected sampling_frequency_hz=2000000')
    for key in ('split', 'ids', 'groups', 'labels'):
        if payload[key].shape != (n,):
            raise ValueError(f'{key} must have shape (N,)')
    if not np.issubdtype(payload['labels'].dtype, np.integer):
        raise ValueError('labels must be integers, with -1 for unlabeled samples')
    if set(payload['split']) != set(SPLITS):
        raise ValueError(f'Exactly these development splits are required: {SPLITS}')
    if len(set(payload['ids'])) != n:
        raise ValueError('Sample IDs must be unique across splits')
    if np.any(payload['groups'].astype(str) == ''):
        raise ValueError('Source groups must be nonempty')
    for group in set(payload['groups']):
        if len(set(payload['split'][payload['groups'] == group])) != 1:
            raise ValueError(f'Source group crosses splits: {group}')
    payload['signals'] = signals
    return payload


def prepare_manifest(manifest: Path, signal_root: Path, output: Path) -> None:
    """Clamped 4096 crop and support transform from the original real loader.

    Input traces must already have the intended acquisition preprocessing.
    No additional filtering or resampling is applied by the MAD4d real loader.
    A stable group identifies the whole acquisition/noise carrier, not an event.
    """
    signal_root = signal_root.resolve()
    rows = list(csv.DictReader(manifest.open(newline='')))
    signals, masks, groups = [], [], []
    for row in rows:
        path = (signal_root / row['path']).resolve()
        if not path.is_relative_to(signal_root):
            raise ValueError('Signal path escapes signal root')
        source = np.load(path, allow_pickle=False)
        if source.ndim != 1 or source.size < 4096 or not np.isfinite(source).all():
            raise ValueError('Source must be one finite trace of at least 4096 samples')
        center = int(round(float(row['center_sample'])))
        start, end = int(row['start_sample']), int(row['end_sample'])
        if not 0 <= start < end <= source.size or not 0 <= center < source.size:
            raise ValueError('Invalid source event bounds or center')
        left = min(max(center - 2048, 0), source.size - 4096)
        signal = np.asarray(source[left:left+4096], dtype=np.float32)
        # Original MAD4d background-margin invariant: 8% at either edge.
        margin = int(round(4096 * 0.08))
        a, b = max(margin, start-left), min(4096-margin, end-left)
        if b <= a:
            raise ValueError(f'Event outside crop: {row["id"]}')
        mask = np.zeros(4096, dtype=bool); mask[a:b] = True
        signals.append(signal); masks.append(mask)
        groups.append(row.get('group') or hashlib.sha256(path.read_bytes()).hexdigest())
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(output)
    payload = dict(signals=np.stack(signals), event_masks=np.stack(masks),
                        split=np.asarray([r['split'] for r in rows]), ids=np.asarray([r['id'] for r in rows]),
                        groups=np.asarray(groups), labels=np.asarray([int(r.get('label',-1)) for r in rows]),
                        sampling_frequency_hz=np.asarray(2_000_000.0))
    validate_arrays(payload)
    with output.open('xb') as stream:
        np.savez_compressed(stream, **payload)


def make_demo(path: Path, seed: int = 42) -> None:
    """Generate toy arrays for software checks; not a model of research data."""
    if path.exists():
        raise FileExistsError(path)
    rng = np.random.default_rng(seed)
    time = np.arange(4096) / 2_000_000
    signals, labels, splits, ids = [], [], [], []
    for split in SPLITS:
        for i in range(6):
            label = i % 3
            frequency = (20+10*label)*1000
            clean = np.exp(-0.5*((time-0.001024)/0.00015)**2)*np.sin(2*np.pi*frequency*time)
            signals.append((clean+rng.normal(0,0.03,time.size)).astype(np.float32))
            labels.append(label);splits.append(split);ids.append(f'toy-{split}-{i}')
    masks = np.zeros((len(signals),4096),dtype=bool);masks[:,1400:2700]=True
    path.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(path, signals=np.stack(signals),event_masks=masks,labels=np.asarray(labels),
                        split=np.asarray(splits),ids=np.asarray(ids),groups=np.asarray(ids),
                        sampling_frequency_hz=np.asarray(2_000_000.0),purpose=np.asarray('synthetic software smoke only'))
