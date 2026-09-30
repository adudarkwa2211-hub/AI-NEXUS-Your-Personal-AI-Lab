"""Training and evaluation helpers for the TF Flowers classifier."""

from __future__ import annotations

import random
from collections import defaultdict


def stratified_split(
    samples: list[tuple[str, int]],
    seed: int = 42,
    train_fraction: float = 0.8,
    validation_fraction: float = 0.1,
) -> tuple[list[int], list[int], list[int]]:
    if not 0 < train_fraction < 1 or not 0 < validation_fraction < 1:
        raise ValueError("Split fractions must be between 0 and 1")
    if train_fraction + validation_fraction >= 1:
        raise ValueError("Train and validation fractions must sum to less than 1")

    indices_by_class: dict[int, list[int]] = defaultdict(list)
    for index, (_, class_index) in enumerate(samples):
        indices_by_class[class_index].append(index)

    rng = random.Random(seed)
    train_indices: list[int] = []
    validation_indices: list[int] = []
    test_indices: list[int] = []
    for indices in indices_by_class.values():
        if len(indices) < 3:
            raise ValueError("Each class needs at least three images for stratified splits")
        rng.shuffle(indices)
        train_count = max(1, int(len(indices) * train_fraction))
        validation_count = max(1, int(len(indices) * validation_fraction))
        if train_count + validation_count >= len(indices):
            train_count = len(indices) - 2
            validation_count = 1
        train_indices.extend(indices[:train_count])
        validation_indices.extend(indices[train_count : train_count + validation_count])
        test_indices.extend(indices[train_count + validation_count :])

    rng.shuffle(train_indices)
    rng.shuffle(validation_indices)
    rng.shuffle(test_indices)
    return train_indices, validation_indices, test_indices
