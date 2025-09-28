from __future__ import annotations

from typing import Dict, Iterable, Tuple

import numpy as np
from scipy.stats import kendalltau

from worldvue.data.types import Pair, STYLE_AXES


def agreement_rate(pairs: Iterable[Pair], axis: str) -> float:
    votes = []
    for pair in pairs:
        score = pair.weak_labels.get(axis)
        if score is None:
            continue
        votes.append(np.sign(score))
    if not votes:
        return 0.0
    ref = votes[0]
    agreements = [1.0 if vote == ref else 0.0 for vote in votes[1:]]
    return float(np.mean(agreements)) if agreements else 1.0


def coverage(pairs: Iterable[Pair], axis: str) -> float:
    total = 0
    covered = 0
    for pair in pairs:
        total += 1
        if axis in pair.weak_labels:
            covered += 1
    return float(covered / total) if total else 0.0


def kendall_tau_alignment(pairs: Iterable[Pair], axis: str) -> Tuple[float, float]:
    weak_scores = []
    llm_scores = []
    for pair in pairs:
        if not pair.llm_labels or axis not in pair.llm_labels:
            continue
        weak_scores.append(pair.weak_labels.get(axis, 0.0))
        winner = pair.llm_labels[axis].get('winner', 'tie')
        mapped = {'A': -1.0, 'B': 1.0}.get(str(winner), 0.0)
        llm_scores.append(mapped)
    if len(weak_scores) < 3:
        return 0.0, 1.0
    tau, p_value = kendalltau(weak_scores, llm_scores)
    return float(tau), float(p_value)


def summarize_axes(pairs: Iterable[Pair]) -> Dict[str, Dict[str, float]]:
    pairs = list(pairs)
    summary: Dict[str, Dict[str, float]] = {}
    for axis in STYLE_AXES:
        summary[axis] = {
            'agreement_rate': agreement_rate(pairs, axis),
            'coverage': coverage(pairs, axis),
        }
    return summary


__all__ = ['agreement_rate', 'coverage', 'kendall_tau_alignment', 'summarize_axes']
