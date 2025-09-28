from __future__ import annotations

import re
from typing import Dict, Tuple

import numpy as np

SENTENCE_RE = re.compile(r'[.!?]+')

HEDGE_WORDS = ['may', 'might', 'could', 'possibly', 'perhaps', 'reportedly']
CERTAINTY_WORDS = ['must', 'definitely', 'always', 'certainly', 'undeniably']


def _safe_divide(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def extract_features(text: str) -> Dict[str, float]:
    text = text or ''
    tokens = text.split()
    token_count = len(tokens)
    char_count = len(text)
    sentences = [segment for segment in SENTENCE_RE.split(text) if segment.strip()]
    sentence_lengths = [len(sentence.split()) for sentence in sentences] or [0]
    exclamations = text.count('!')
    caps = sum(1 for char in text if char.isupper())
    quotes = len(re.findall(r'"[^"]*"', text))
    hedges = sum(text.lower().count(word) for word in HEDGE_WORDS)
    certainty = sum(text.lower().count(word) for word in CERTAINTY_WORDS)
    return {
        'token_count': float(token_count),
        'char_count': float(char_count),
        'exclamation_ratio': _safe_divide(exclamations, char_count),
        'caps_ratio': _safe_divide(caps, max(char_count, 1)),
        'quote_density': _safe_divide(quotes, max(token_count, 1)),
        'hedge_count': float(hedges),
        'certainty_count': float(certainty),
        'mean_sentence_length': float(np.mean(sentence_lengths)),
        'sentence_count': float(len(sentences)),
    }


def extract_pair_features(text_a: str, text_b: str) -> Tuple[Dict[str, float], Dict[str, float], Dict[str, float]]:
    features_a = extract_features(text_a)
    features_b = extract_features(text_b)
    diff = {key: features_a[key] - features_b[key] for key in features_a.keys()}
    return features_a, features_b, diff


def diff_vector(diff_features: Dict[str, float]) -> np.ndarray:
    keys = sorted(diff_features)
    return np.asarray([diff_features[key] for key in keys], dtype=float)


__all__ = ['extract_features', 'extract_pair_features', 'diff_vector']
