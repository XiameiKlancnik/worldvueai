from __future__ import annotations

from collections import defaultdict
from typing import Dict, Iterable, List

import numpy as np

from worldvue.data.types import Article

try:  # pragma: no cover - heavy optional dependency
    import hdbscan
except Exception:
    hdbscan = None

try:
    from sklearn.cluster import DBSCAN
except Exception as error:  # pragma: no cover
    raise RuntimeError('scikit-learn must be installed for clustering') from error


def _run_hdbscan(embeddings: np.ndarray, min_cluster_size: int) -> np.ndarray:
    if hdbscan is None:
        raise RuntimeError('hdbscan extra is required for HDBSCAN clustering')
    clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, metric='cosine')
    labels = clusterer.fit_predict(embeddings)
    return labels


def _run_dbscan(embeddings: np.ndarray, min_cluster_size: int) -> np.ndarray:
    eps = 0.3
    clusterer = DBSCAN(eps=eps, min_samples=min_cluster_size, metric='cosine')
    labels = clusterer.fit_predict(embeddings)
    return labels


def cluster_articles(
    articles: Iterable[Article],
    *,
    min_cluster_size: int = 3,
    per_country_threshold: int = 30,
    use_hdbscan: bool = True,
) -> Dict[str, List[str]]:
    items = [article for article in articles if article.embedding]
    if not items:
        return {}
    clusters: Dict[str, List[str]] = {}
    grouped: Dict[str, List[Article]] = defaultdict(list)
    for article in items:
        key = article.source_country if len(items) >= per_country_threshold else 'global'
        grouped[key].append(article)
    cluster_index = 0
    for key, group in grouped.items():
        embeddings = np.asarray([article.embedding for article in group if article.embedding])
        if embeddings.size == 0:
            continue
        if use_hdbscan and hdbscan is not None:
            labels = _run_hdbscan(embeddings, min_cluster_size)
        else:
            labels = _run_dbscan(embeddings, min_cluster_size)
        for label, article in zip(labels, group):
            if label == -1:
                continue
            cluster_id = f'{key}-{label}' if key != 'global' else f'cluster-{cluster_index + label}'
            clusters.setdefault(cluster_id, []).append(article.id)
        cluster_index += len(set(labels))
    return {cluster_id: members for cluster_id, members in clusters.items() if len(members) >= min_cluster_size}


__all__ = ['cluster_articles']
