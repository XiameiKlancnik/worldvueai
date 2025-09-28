from __future__ import annotations

from typing import Dict, Iterable, Optional

import numpy as np

from worldvue.data.types import Pair, STYLE_AXES
from worldvue.weak.labelers import DEFAULT_LABELERS, score_to_winner
from worldvue.weak.model import EMLabelModel


def weak_vs_llm_agreement(pairs: Iterable[Pair], axes: Iterable[str] | None = None) -> Dict[str, Dict[str, float]]:
    axes = list(axes or STYLE_AXES)
    results: Dict[str, Dict[str, float]] = {}
    for axis in axes:
        agreements = []
        confidences = []
        for pair in pairs:
            if not pair.llm_labels or axis not in pair.llm_labels:
                continue
            weak_score = pair.weak_labels.get(axis, 0.0)
            weak_winner = score_to_winner(weak_score)
            llm_winner = pair.llm_labels[axis].get('winner')
            if not isinstance(llm_winner, str):
                continue
            agreements.append(1.0 if weak_winner == llm_winner else 0.0)
            confidence = float(pair.weak_confidence.get(axis, 0.0))
            confidences.append(confidence)
        if not agreements:
            results[axis] = {'agreement_rate': 0.0, 'confidence': 0.0}
        else:
            results[axis] = {
                'agreement_rate': float(np.mean(agreements)),
                'confidence': float(np.mean(confidences) if confidences else 0.0),
            }
    return results


def calibrate_labeling_functions(
    pairs: Iterable[Pair],
    *,
    label_model: Optional[EMLabelModel] = None,
    axes: Iterable[str] | None = None,
) -> EMLabelModel:
    model = label_model or EMLabelModel(DEFAULT_LABELERS, axes=axes)
    model.fit(list(pairs))
    return model


__all__ = ['weak_vs_llm_agreement', 'calibrate_labeling_functions']
