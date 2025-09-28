from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, List, Optional

import pandas as pd

from worldvue.data.types import Article
from worldvue.text.lang import detect_language


SUPPORTED_EXTENSIONS = {'.csv', '.json', '.jsonl', '.parquet'}


def _read_frame(path: Path) -> pd.DataFrame:
    if path.suffix not in SUPPORTED_EXTENSIONS:
        raise ValueError(f'Unsupported file type: {path.suffix}')
    if path.suffix == '.parquet':
        return pd.read_parquet(path)
    if path.suffix == '.csv':
        return pd.read_csv(path)
    if path.suffix == '.jsonl':
        return pd.read_json(path, lines=True)
    if path.suffix == '.json':
        return pd.read_json(path)
    raise ValueError(f'Unable to read file: {path}')


def load_articles(
    path: Path | str,
    *,
    limit: Optional[int] = None,
    detect_missing_language: bool = True,
) -> List[Article]:
    file_path = Path(path)
    frame = _read_frame(file_path)
    if limit is not None:
        frame = frame.head(limit)
    articles: List[Article] = []
    for record in frame.to_dict(orient='records'):
        language = record.get('language') or record.get('lang') or 'und'
        text = record.get('text') or record.get('body') or ''
        if detect_missing_language and (language == 'und' or not language.strip()):
            language = detect_language(text)
        published_at = pd.to_datetime(record.get('published_at') or record.get('date'))
        article = Article(
            id=str(record.get('id') or record.get('article_id') or record.get('uuid')),
            title=record.get('title') or record.get('headline') or '',
            text=text,
            source_name=record.get('source_name') or record.get('source') or '',
            source_country=(record.get('source_country') or record.get('country') or 'UN')[:5],
            language=language,
            published_at=published_at.to_pydatetime(),
            embedding=record.get('embedding'),
        )
        articles.append(article)
    return articles


def load_cached_pairs(path: Path | str) -> Iterable[dict]:
    file_path = Path(path)
    if not file_path.exists():
        return []
    with file_path.open('r', encoding='utf-8') as handle:
        return [json.loads(line) for line in handle if line.strip()]


__all__ = ['load_articles', 'load_cached_pairs', 'SUPPORTED_EXTENSIONS']
