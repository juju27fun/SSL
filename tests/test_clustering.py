from __future__ import annotations

from pathlib import Path

import numpy as np

from smi_ssl.clustering import (
    PreparedLatents,
    evaluate_partition,
    hungarian_mapping,
    prepare_latents,
    summarize_partitions,
)


ROOT = Path(__file__).parents[1]


def test_preprocessing_and_pca_are_fit_on_train_only() -> None:
    train = np.asarray([[0.0, 1.0], [1.0, 0.0], [2.0, 1.0], [1.0, 2.0]])
    first = prepare_latents(train, np.asarray([[3.0, 3.0]]))
    second = prepare_latents(train, np.asarray([[3000.0, -9000.0]]))
    np.testing.assert_allclose(first.train, second.train)
    np.testing.assert_allclose(
        first.explained_variance_ratio, second.explained_variance_ratio
    )


def test_hungarian_mapping_uses_train_correspondence() -> None:
    labels = np.asarray([0, 0, 1, 1, 2, 2])
    clusters = np.asarray([2, 2, 0, 0, 1, 1])
    assert hungarian_mapping(labels, clusters, n_classes=3) == {0: 1, 1: 2, 2: 0}


def test_ari_is_scored_on_unmapped_cluster_ids() -> None:
    train = np.vstack(
        [
            np.repeat([[-5.0, 0.0]], 4, axis=0),
            np.repeat([[0.0, 5.0]], 4, axis=0),
            np.repeat([[5.0, 0.0]], 4, axis=0),
        ]
    )
    labels = np.repeat(np.arange(3), 4)
    prepared = prepare_latents(train, train)
    result = evaluate_partition(prepared, labels, labels, seed=41, n_init=5)
    assert result["ari"] == 1.0
    assert np.asarray(result["validation_correct"]).all()


def test_consensus_requires_a_majority_of_cluster_seeds() -> None:
    base = {
        "ari": 0.0,
        "confusion_row_normalized": np.eye(3),
    }
    partitions = [
        {**base, "validation_correct": np.asarray([True, False])},
        {**base, "validation_correct": np.asarray([True, True])},
        {**base, "validation_correct": np.asarray([False, False])},
    ]
    summary = summarize_partitions(partitions)
    np.testing.assert_array_equal(summary["correct_consensus"], [True, False])


def test_evaluation_accepts_prepared_latents_without_validation_fit() -> None:
    prepared = PreparedLatents(
        train=np.asarray([[-1.0], [-0.9], [0.0], [0.1], [1.0], [1.1]]),
        validation=np.asarray([[-1.0], [0.0], [1.0]]),
        validation_pca=np.zeros((3, 2)),
        explained_variance_ratio=np.asarray([1.0, 0.0]),
    )
    result = evaluate_partition(
        prepared,
        np.repeat(np.arange(3), 2),
        np.arange(3),
        seed=43,
        n_init=5,
    )
    assert np.asarray(result["confusion_counts"]).sum() == 3
