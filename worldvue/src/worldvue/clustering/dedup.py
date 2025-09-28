from __future__ import annotations

from typing import Iterable, List, Tuple

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from worldvue.data.types import Article


def remove_near_duplicates(
    articles: Iterable[Article],
    *,
    threshold: float = 0.95,
) -> Tuple[List[Article], List[str]]:
    unique: List[Article] = []
    dropped: List[str] = []
    embeddings: List[np.ndarray] = []
    for article in articles:
        if not article.embedding:
            unique.append(article)
            continue
        vector = np.asarray(article.embedding).reshape(1, -1)
        if embeddings:
            existing = np.vstack(embeddings)
            similarities = cosine_similarity(existing, vector).flatten()
            if np.any(similarities >= threshold):
                dropped.append(article.id)
                continue
        embeddings.append(vector)
        unique.append(article)
    return unique, dropped


__all__ = ['remove_near_duplicates']
