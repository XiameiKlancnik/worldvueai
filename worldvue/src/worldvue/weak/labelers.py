from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np

from worldvue.data.types import Pair
from worldvue.weak.features import extract_pair_features


def score_to_winner(score: float) -> str:
    if score > 0:
        return 'B'
    if score < 0:
        return 'A'
    return 'tie'


def winner_to_score(winner: str) -> float:
    mapping = {'A': -1.0, 'B': 1.0, 'tie': 0.0}
    return mapping.get(winner, 0.0)


@dataclass
class WeakLabel:
    axis: str
    score: float
    confidence: float


class WeakLabeler(ABC):
    axis: str
    name: str
    threshold: float = 0.1

    def label(self, pair: Pair) -> Dict[str, float]:
        label = self.label_with_confidence(pair)
        return {label.axis: label.score}

    def label_with_confidence(self, pair: Pair) -> WeakLabel:
        pair.ensure_articles()
        score, confidence = self._score(pair)
        return WeakLabel(axis=self.axis, score=score, confidence=confidence)

    @abstractmethod
    def _score(self, pair: Pair) -> Tuple[float, float]:
        raise NotImplementedError

    def _diff_to_score(self, diff: float) -> float:
        if abs(diff) < self.threshold:
            return 0.0
        return float(np.sign(diff))

    def _confidence_from_diff(self, diff: float) -> float:
        return float(min(1.0, abs(diff)))


class HypeLabeler(WeakLabeler):
    axis = 'hype'
    name = 'punctuation_intensity'
    threshold = 0.05

    def _score(self, pair: Pair) -> Tuple[float, float]:
        _, _, diff = extract_pair_features(pair.article_a.text, pair.article_b.text)
        metric = diff['exclamation_ratio'] + diff['caps_ratio']
        score = self._diff_to_score(-metric)
        confidence = self._confidence_from_diff(metric)
        return score, confidence


class QuoteBalanceLabeler(WeakLabeler):
    axis = 'one_sidedness'
    name = 'quote_balance'
    threshold = 0.05

    def _score(self, pair: Pair) -> Tuple[float, float]:
        _, _, diff = extract_pair_features(pair.article_a.text, pair.article_b.text)
        metric = diff['quote_density']
        score = self._diff_to_score(-metric)
        confidence = self._confidence_from_diff(metric)
        return score, confidence


class HedgingLabeler(WeakLabeler):
    axis = 'certain_vs_caution'
    name = 'hedge_presence'
    threshold = 0.2

    def _score(self, pair: Pair) -> Tuple[float, float]:
        _, _, diff = extract_pair_features(pair.article_a.text, pair.article_b.text)
        metric = diff['hedge_count']
        score = self._diff_to_score(metric)
        confidence = self._confidence_from_diff(metric)
        return score, confidence


class CertaintyLabeler(WeakLabeler):
    axis = 'certain_vs_caution'
    name = 'certainty_terms'
    threshold = 0.2

    def _score(self, pair: Pair) -> Tuple[float, float]:
        _, _, diff = extract_pair_features(pair.article_a.text, pair.article_b.text)
        metric = diff['certainty_count']
        score = self._diff_to_score(-metric)
        confidence = self._confidence_from_diff(metric)
        return score, confidence


class SourcingLabeler(WeakLabeler):
    axis = 'sourcing'
    name = 'citation_density'
    threshold = 0.05

    SOURCE_CUES = ['according to', 'reports', 'told', 'said', 'cited', 'stated']

    def _score(self, pair: Pair) -> Tuple[float, float]:
        cues_a = self._count_cues(pair.article_a.text)
        cues_b = self._count_cues(pair.article_b.text)
        diff = float(cues_b - cues_a)
        score = self._diff_to_score(diff)
        confidence = self._confidence_from_diff(diff / 5 if diff else 0)
        return score, confidence

    def _count_cues(self, text: str) -> int:
        lowered = text.lower()
        return sum(lowered.count(cue) for cue in self.SOURCE_CUES)


class FixItLabeler(WeakLabeler):
    axis = 'fight_vs_fix'
    name = 'actionability'
    threshold = 0.1

    FIX_CUES = ['plan', 'policy', 'proposal', 'program', 'initiative']
    FIGHT_CUES = ['blame', 'clash', 'criticize', 'attack', 'accuse']

    def _score(self, pair: Pair) -> Tuple[float, float]:
        score_a = self._score_text(pair.article_a.text)
        score_b = self._score_text(pair.article_b.text)
        diff = float(score_b - score_a)
        score = self._diff_to_score(diff)
        confidence = self._confidence_from_diff(abs(diff) / 5)
        return score, confidence

    def _score_text(self, text: str) -> float:
        lowered = text.lower()
        fix = sum(lowered.count(cue) for cue in self.FIX_CUES)
        fight = sum(lowered.count(cue) for cue in self.FIGHT_CUES)
        return fix - fight


DEFAULT_LABELERS = [
    QuoteBalanceLabeler(),
    HedgingLabeler(),
    CertaintyLabeler(),
    HypeLabeler(),
    SourcingLabeler(),
    FixItLabeler(),
]


__all__ = [
    'WeakLabeler',
    'WeakLabel',
    'DEFAULT_LABELERS',
    'HypeLabeler',
    'QuoteBalanceLabeler',
    'HedgingLabeler',
    'CertaintyLabeler',
    'SourcingLabeler',
    'FixItLabeler',
    'score_to_winner',
    'winner_to_score',
]
