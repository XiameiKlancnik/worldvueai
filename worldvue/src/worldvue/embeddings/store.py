from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Mapping

import joblib

DEFAULT_STORE_PATH = Path('artifacts/article_embeddings.joblib')


def load_store(path: Path | str = DEFAULT_STORE_PATH) -> Dict[str, List[float]]:
    store_path = Path(path)
    if not store_path.exists():
        return {}
    return joblib.load(store_path)


def persist_store(store: Mapping[str, Iterable[float]], path: Path | str = DEFAULT_STORE_PATH) -> None:
    store_path = Path(path)
    store_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({k: list(v) for k, v in store.items()}, store_path)


def attach_embeddings(articles, store: Mapping[str, Iterable[float]]) -> None:
    for article in articles:
        if article.id in store:
            article.embedding = list(store[article.id])


__all__ = ['DEFAULT_STORE_PATH', 'load_store', 'persist_store', 'attach_embeddings']
