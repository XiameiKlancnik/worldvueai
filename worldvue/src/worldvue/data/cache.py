from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List, Mapping

import joblib

DEFAULT_CACHE_PATH = Path('artifacts/embeddings.pkl')


def load_embedding_cache(path: Path | str = DEFAULT_CACHE_PATH) -> Dict[str, List[float]]:
    cache_path = Path(path)
    if not cache_path.exists():
        return {}
    return joblib.load(cache_path)


def save_embedding_cache(
    cache: Mapping[str, List[float]],
    path: Path | str = DEFAULT_CACHE_PATH,
) -> None:
    cache_path = Path(path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(dict(cache), cache_path)


def merge_embedding_cache(
    cache: Mapping[str, List[float]],
    new_items: Mapping[str, Iterable[float]],
) -> Dict[str, List[float]]:
    merged = dict(cache)
    for key, value in new_items.items():
        merged[key] = list(value)
    return merged


def log_cache_metadata(path: Path | str = DEFAULT_CACHE_PATH) -> dict:
    cache_path = Path(path)
    if not cache_path.exists():
        return {'path': str(cache_path), 'exists': False}
    cache = load_embedding_cache(cache_path)
    return {
        'path': str(cache_path),
        'exists': True,
        'size': len(cache),
    }


def dump_costs_snapshot(costs_path: Path | str, entries: List[dict]) -> None:
    path = Path(costs_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8') as handle:
        for entry in entries:
            handle.write(json.dumps(entry) + '
')


__all__ = [
    'DEFAULT_CACHE_PATH',
    'load_embedding_cache',
    'save_embedding_cache',
    'merge_embedding_cache',
    'log_cache_metadata',
    'dump_costs_snapshot',
]
