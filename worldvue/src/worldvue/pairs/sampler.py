from __future__ import annotations

import random
from collections import defaultdict
from typing import Iterable, List, Sequence

import numpy as np

from worldvue.data.types import Pair


def _group_by_cluster(pairs: Iterable[Pair]) -> dict[str, List[Pair]]:
    grouped: dict[str, List[Pair]] = defaultdict(list)
    for pair in pairs:
        grouped[pair.cluster_id].append(pair)
    return grouped


def _score_uncertainty(pair: Pair, labelers: Sequence) -> float:
    if not labelers:
        return 0.0
    votes: List[float] = []
    for labeler in labelers:
        label_map = labeler.label(pair)
        votes.extend(label_map.values())
    if not votes:
        return 0.0
    return float(np.std(votes))


def select_validation_pairs(
    all_pairs: Iterable[Pair],
    *,
    budget: int = 200,
    calibration: int = 100,
    uncertainty: int = 50,
    test: int = 50,
    min_pairs_per_cluster: int = 2,
    random_seed: int = 17,
    labelers: Sequence | None = None,
) -> List[Pair]:
    pairs = list(all_pairs)
    rng = random.Random(random_seed)
    grouped = _group_by_cluster(pairs)
    selected: List[Pair] = []

    # Calibration sample: ensure coverage across clusters
    clusters_sorted = sorted(grouped.items(), key=lambda item: len(item[1]), reverse=True)
    for cluster_id, cluster_pairs in clusters_sorted:
        pool = list(cluster_pairs)
        sample_size = min(min_pairs_per_cluster, len(pool))
        if sample_size == 0:
            continue
        selected.extend(rng.sample(pool, sample_size))
        if len(selected) >= calibration:
            break
    selected = selected[:calibration]

    # Uncertainty sample: high disagreement pairs
    remaining = [pair for pair in pairs if pair not in selected]
    if uncertainty > 0 and remaining:
        scored = sorted(
            remaining,
            key=lambda pair: _score_uncertainty(pair, labelers or []),
            reverse=True,
        )
        selected.extend(scored[:uncertainty])

    # Test hold-out: random sample from leftovers
    remaining = [pair for pair in pairs if pair not in selected]
    if test > 0 and remaining:
        sample = rng.sample(remaining, min(test, len(remaining)))
        selected.extend(sample)

    return selected[:budget]


__all__ = ['select_validation_pairs']
