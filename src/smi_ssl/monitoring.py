"""Fixed ID-hashed, class-proportional monitors from the retained source run.

Labels are used only to allocate monitoring rows, never to optimize the model.
0/1/2 correspond to the retained 2um/4um/10um populations. Unknown labels
(including -1) form explicit additional strata; no physical class is inferred.
"""
import hashlib
import math
from typing import Any
from .data import EventArrays

PHYSICAL_CLASSES = ("2um", "4um", "10um")
CLASS_NAMES = dict(enumerate(PHYSICAL_CLASSES))

def fixed_class_proportional_monitor_indices(
    dataset: EventArrays,
    *,
    max_samples: int,
    seed: int,
    split_tag: str,
) -> tuple[list[int], dict[str, Any]]:
    """Select an immutable class-proportional monitor from dataset metadata."""
    if max_samples < 1:
        raise ValueError("monitor size must be positive")
    target = min(int(max_samples), len(dataset))
    if not len(dataset):
        raise ValueError("monitor dataset must be nonempty")
    labels = dataset.payload["labels"][dataset.indices]
    names = [CLASS_NAMES.get(int(label), str(int(label))) for label in labels]
    classes = tuple(name for name in PHYSICAL_CLASSES if name in names) + tuple(sorted(set(names) - set(PHYSICAL_CLASSES)))
    grouped: dict[str, list[tuple[int, str]]] = {name: [] for name in classes}
    for dataset_index, (index, class_name) in enumerate(zip(dataset.indices, names, strict=True)):
        grouped[class_name].append((dataset_index, str(dataset.payload["ids"][index])))
    exact = {
        name: target * len(rows) / max(len(dataset), 1)
        for name, rows in grouped.items()
    }
    quotas = {name: int(math.floor(value)) for name, value in exact.items()}
    remaining = target - sum(quotas.values())
    remainder_order = sorted(
        classes,
        key=lambda name: (-(exact[name] - quotas[name]), name),
    )
    for name in remainder_order[:remaining]:
        quotas[name] += 1

    selected: list[tuple[int, str]] = []
    for class_name in classes:
        ranked = sorted(
            grouped[class_name],
            key=lambda item: hashlib.sha256(
                f"{seed}|{split_tag}|{item[1]}".encode("utf-8")
            ).hexdigest(),
        )
        selected.extend(ranked[: quotas[class_name]])
    if len(selected) != target:
        raise ValueError("class-proportional monitor selection is incomplete")
    indices = sorted(index for index, _ in selected)
    selected_ids = sorted(sample_id for _, sample_id in selected)
    ids_sha256 = hashlib.sha256("\n".join(selected_ids).encode("utf-8")).hexdigest()
    return indices, {
        "protocol": "class-proportional-sample-id-sha256-v1",
        "split": split_tag,
        "seed": int(seed),
        "count": len(indices),
        "class_counts": quotas,
        "selected_ids_sha256": ids_sha256,
    }
