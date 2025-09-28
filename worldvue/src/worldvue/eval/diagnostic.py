from __future__ import annotations

from typing import Dict, Iterable, Sequence

import numpy as np

from worldvue.data.types import Pair
from worldvue.weak.labelers import DEFAULT_LABELERS, WeakLabeler


def evaluate_labelers(pairs: Iterable[Pair], labelers: Sequence[WeakLabeler] | None = None) -> Dict[str, Dict[str, float]]:
    labelers = list(labelers or DEFAULT_LABELERS)
    pairs = list(pairs)
    results: Dict[str, Dict[str, float]] = {}
    for labeler in labelers:
        agreements = []
        coverage = 0
        total = 0
        for pair in pairs:
            total += 1
            result = labeler.label_with_confidence(pair)
            if result.score == 0:
                continue
            coverage += 1
            weak_score = pair.weak_labels.get(labeler.axis, 0.0)
            agreements.append(1.0 if np.sign(weak_score) == np.sign(result.score) else 0.0)
        accuracy = float(np.mean(agreements)) if agreements else 0.0
        results[labeler.name] = {
            'axis': labeler.axis,
            'accuracy': accuracy,
            'coverage': coverage / total if total else 0.0,
        }
    return results


__all__ = ['evaluate_labelers']
