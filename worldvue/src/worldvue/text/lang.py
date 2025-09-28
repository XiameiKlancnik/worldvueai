from __future__ import annotations

from functools import lru_cache
from typing import Iterable, Optional

try:
    from fasttext_langdetect import detect as fasttext_detect
except Exception:  # pragma: no cover - optional dependency
    fasttext_detect = None

try:
    from langdetect import DetectorFactory, detect

    DetectorFactory.seed = 13
except Exception as error:  # pragma: no cover - defensive guard
    raise RuntimeError('langdetect must be installed') from error


def _safe_detect(text: str) -> str:
    try:
        return detect(text)
    except Exception:
        return 'und'


@lru_cache(maxsize=2048)
def detect_language(text: str, *, fallback: str = 'und') -> str:
    normalized = text.strip() if text else ''
    if not normalized:
        return fallback
    if fasttext_detect is not None:
        try:
            language = fasttext_detect(normalized)
            if language:
                return language[:5]
        except Exception:
            pass
    language = _safe_detect(normalized)
    return language[:5]


def most_common_language(values: Iterable[str]) -> str:
    counts: dict[str, int] = {}
    for value in values:
        lang = value or 'und'
        counts[lang] = counts.get(lang, 0) + 1
    if not counts:
        return 'und'
    return max(counts, key=counts.get)


__all__ = ['detect_language', 'most_common_language']
