from __future__ import annotations

from typing import Dict, Iterable, Optional

import numpy as np
from sklearn.isotonic import IsotonicRegression

from worldvue.data.types import Pair, STYLE_AXES
from worldvue.training.ranker import predict_axis
from worldvue.weak.features import extract_pair_features, diff_vector
from worldvue.weak.labelers import winner_to_score


def _build_features(pair: Pair) -> np.ndarray:
    pair.ensure_articles()
    _, _, diff = extract_pair_features(pair.article_a.text, pair.article_b.text)
    return diff_vector(diff)


class IsotonicCalibrator:
    def __init__(self, axes: Iterable[str] | None = None) -> None:
        self.axes = list(axes or STYLE_AXES)
        self.models: Dict[str, IsotonicRegression] = {}

    def fit(self, models: Dict[str, object], calibration_pairs: Iterable[Pair]) -> None:
        pairs = list(calibration_pairs)
        if not pairs:
            return
        for axis, model in models.items():
            scores = []
            targets = []
            for pair in pairs:
                vector = _build_features(pair)
                score = predict_axis(model, vector)
                if pair.llm_labels and axis in pair.llm_labels:
                    target = winner_to_score(str(pair.llm_labels[axis].get('winner', 'tie')))
                else:
                    target = pair.weak_labels.get(axis, 0.0)
                if target == 0:
                    continue
                scores.append(score)
                targets.append(1 if target > 0 else 0)
            if len(set(targets)) < 2:
                continue
            iso = IsotonicRegression(out_of_bounds='clip')
            iso.fit(scores, targets)
            self.models[axis] = iso

    def calibrate(self, axis: str, score: float) -> float:
        model = self.models.get(axis)
        if not model:
            return score
        probability = model.transform([score])[0]
        return float(2 * probability - 1)


__all__ = ['IsotonicCalibrator']
