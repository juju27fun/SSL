"""Leakage-safe clustering evaluation for frozen MAD latent spaces."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, confusion_matrix
from sklearn.preprocessing import Normalizer, StandardScaler


@dataclass(frozen=True)
class PreparedLatents:
    train: np.ndarray
    validation: np.ndarray
    validation_pca: np.ndarray
    explained_variance_ratio: np.ndarray


def prepare_latents(train: np.ndarray, validation: np.ndarray) -> PreparedLatents:
    """Fit every representation transform on unlabeled train rows only."""
    train = np.asarray(train, dtype=np.float64)
    validation = np.asarray(validation, dtype=np.float64)
    if train.ndim != 2 or validation.ndim != 2 or train.shape[1] != validation.shape[1]:
        raise ValueError("train and validation latents must be aligned 2-D matrices")
    if not np.isfinite(train).all() or not np.isfinite(validation).all():
        raise ValueError("latents contain non-finite values")
    scaler = StandardScaler().fit(train)
    normalizer = Normalizer(norm="l2").fit(scaler.transform(train))
    train_normalized = normalizer.transform(scaler.transform(train))
    validation_normalized = normalizer.transform(scaler.transform(validation))
    pca = PCA(n_components=2, svd_solver="full").fit(train_normalized)
    return PreparedLatents(
        train=train_normalized,
        validation=validation_normalized,
        validation_pca=pca.transform(validation_normalized),
        explained_variance_ratio=pca.explained_variance_ratio_,
    )


def hungarian_mapping(
    train_labels: np.ndarray, train_clusters: np.ndarray, *, n_classes: int
) -> dict[int, int]:
    """Map cluster IDs to class IDs using train labels for interpretation only."""
    labels = np.asarray(train_labels, dtype=np.int64)
    clusters = np.asarray(train_clusters, dtype=np.int64)
    contingency = np.zeros((n_classes, n_classes), dtype=np.int64)
    for cluster_id in range(n_classes):
        for class_id in range(n_classes):
            contingency[cluster_id, class_id] = np.sum(
                (clusters == cluster_id) & (labels == class_id)
            )
    rows, columns = linear_sum_assignment(-contingency)
    return {int(row): int(column) for row, column in zip(rows, columns, strict=True)}


def evaluate_partition(
    prepared: PreparedLatents,
    train_labels: np.ndarray,
    validation_labels: np.ndarray,
    *,
    seed: int,
    n_classes: int = 3,
    n_init: int = 20,
) -> dict[str, object]:
    """Fit K-means on train and score the untouched validation partition."""
    model = KMeans(n_clusters=n_classes, n_init=n_init, random_state=seed)
    train_clusters = model.fit_predict(prepared.train)
    validation_clusters = model.predict(prepared.validation)
    mapping = hungarian_mapping(train_labels, train_clusters, n_classes=n_classes)
    mapped = np.asarray([mapping[int(value)] for value in validation_clusters])
    counts = confusion_matrix(
        validation_labels, mapped, labels=np.arange(n_classes)
    )
    denominators = counts.sum(axis=1, keepdims=True)
    normalized = np.divide(
        counts,
        denominators,
        out=np.zeros_like(counts, dtype=np.float64),
        where=denominators != 0,
    )
    return {
        "seed": int(seed),
        "ari": float(adjusted_rand_score(validation_labels, validation_clusters)),
        "mapping": mapping,
        "validation_clusters": validation_clusters,
        "validation_predictions": mapped,
        "validation_correct": mapped == np.asarray(validation_labels),
        "confusion_counts": counts,
        "confusion_row_normalized": normalized,
    }


def summarize_partitions(partitions: list[dict[str, object]]) -> dict[str, object]:
    if not partitions:
        raise ValueError("at least one partition is required")
    ari = np.asarray([row["ari"] for row in partitions], dtype=np.float64)
    confusion = np.stack(
        [np.asarray(row["confusion_row_normalized"]) for row in partitions]
    )
    correctness = np.stack(
        [np.asarray(row["validation_correct"], dtype=bool) for row in partitions]
    )
    return {
        "ari_mean": float(ari.mean()),
        "ari_std": float(ari.std(ddof=1)) if len(ari) > 1 else 0.0,
        "confusion_mean": confusion.mean(axis=0),
        "confusion_std": confusion.std(axis=0, ddof=1) if len(confusion) > 1 else np.zeros_like(confusion[0]),
        "correct_consensus": correctness.sum(axis=0) >= (len(partitions) // 2 + 1),
    }
