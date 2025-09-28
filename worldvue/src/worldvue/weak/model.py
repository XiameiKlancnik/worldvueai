from __future__ import annotations

from collections import defaultdict
from typing import Dict, Iterable, List

import numpy as np

from worldvue.data.types import Pair, STYLE_AXES
from worldvue.weak.labelers import WeakLabeler


class EMLabelModel:
    def __init__(self, labelers: Iterable[WeakLabeler], axes: Iterable[str] | None = None) -> None:
        self.labelers = list(labelers)
        self.axes = list(axes or STYLE_AXES)
        self.weights: Dict[str, float] = {labeler.name: 1.0 for labeler in self.labelers}

    def fit(self, pairs: Iterable[Pair], iterations: int = 3) -> None:
        pairs = list(pairs)
        if not pairs:
            return
        label_cache: Dict[str, Dict[str, float]] = defaultdict(dict)
        for _ in range(iterations):
            # E-step: aggregate weak labels with current weights
            for pair in pairs:
                axis_scores: Dict[str, float] = defaultdict(float)
                axis_weight: Dict[str, float] = defaultdict(float)
                for labeler in self.labelers:
                    result = labeler.label_with_confidence(pair)
                    label_cache[labeler.name][pair.pair_id] = result.score
                    weight = self.weights[labeler.name]
                    weighted = weight * result.score * result.confidence
                    axis_scores[result.axis] += weighted
                    axis_weight[result.axis] += weight * max(result.confidence, 0.1)
                for axis in self.axes:
                    if axis_weight[axis]:
                        score = axis_scores[axis] / axis_weight[axis]
                    else:
                        score = 0.0
                    pair.weak_labels[axis] = float(np.clip(score, -1.0, 1.0))
                    pair.weak_confidence[axis] = float(min(1.0, axis_weight[axis] / (len(self.labelers) or 1)))

            # M-step: update labeler weights based on agreement
            for labeler in self.labelers:
                axis = labeler.axis
                agreements: List[float] = []
                for pair in pairs:
                    aggregated = pair.weak_labels.get(axis, 0.0)
                    label_score = label_cache[labeler.name].get(pair.pair_id, 0.0)
                    if not aggregated or not label_score:
                        continue
                    same_direction = np.sign(aggregated) == np.sign(label_score)
                    agreements.append(1.0 if same_direction else 0.0)
                weight = (np.mean(agreements) if agreements else 0.5) + 0.1
                self.weights[labeler.name] = float(np.clip(weight, 0.1, 2.0))

    def predict(self, pair: Pair) -> Dict[str, float]:
        axis_scores: Dict[str, float] = defaultdict(float)
        axis_weight: Dict[str, float] = defaultdict(float)
        for labeler in self.labelers:
            result = labeler.label_with_confidence(pair)
            weight = self.weights[labeler.name]
            weighted = weight * result.score * result.confidence
            axis_scores[result.axis] += weighted
            axis_weight[result.axis] += weight * max(result.confidence, 0.1)
        predictions: Dict[str, float] = {}
        for axis in self.axes:
            if axis_weight[axis]:
                predictions[axis] = float(np.clip(axis_scores[axis] / axis_weight[axis], -1.0, 1.0))
            else:
                predictions[axis] = 0.0
        return predictions


__all__ = ['EMLabelModel']
