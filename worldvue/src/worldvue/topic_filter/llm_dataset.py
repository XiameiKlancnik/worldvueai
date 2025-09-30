from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, List, Dict, Any

import pandas as pd


def load_llm_labels(jsonl_path: Path | str, min_conf: float = 0.85) -> pd.DataFrame:
    """Load topic labels produced by the LLM judge (JSONL)."""
    records: List[Dict[str, Any]] = []
    jsonl_path = Path(jsonl_path)
    with jsonl_path.open('r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                continue
            primary = data.get('primary')
            conf = data.get('confidence', 0.0)
            if not primary or conf < min_conf:
                continue
            secondaries = data.get('secondaries') or []
            if isinstance(secondaries, str):
                secondaries = [secondaries]
            records.append({
                'article_id': str(data.get('article_id')),
                'label': primary,
                'confidence': float(conf),
                'secondaries': secondaries
            })
    return pd.DataFrame(records)


def build_multiclass_dataset(labels_df: pd.DataFrame,
                             articles_df: pd.DataFrame,
                             *,
                             max_per_class: int = 3000,
                             embedding_col: str = 'embedding',
                             id_col: str = 'article_id') -> pd.DataFrame:
    """Merge LLM labels with article embeddings and balance per class."""
    articles_df = articles_df.copy()
    if id_col not in articles_df.columns:
        if 'id' in articles_df.columns:
            articles_df[id_col] = articles_df['id'].astype(str)
        else:
            raise ValueError(f'Missing id column: {id_col}')

    if 'secondaries' not in labels_df.columns:
        labels_df = labels_df.copy()
        labels_df['secondaries'] = [[] for _ in range(len(labels_df))]

    merged = labels_df.merge(articles_df[[id_col, embedding_col]], on=id_col, how='inner')
    parts = []
    for label, group in merged.groupby('label'):
        n = min(len(group), max_per_class)
        sampled = group.sample(n=n, random_state=42) if len(group) > n else group
        parts.append(sampled)
    dataset = pd.concat(parts, ignore_index=True)
    dataset = dataset.sample(frac=1.0, random_state=42).reset_index(drop=True)
    return dataset[[id_col, 'label', 'confidence', 'secondaries', embedding_col]]


__all__ = ['load_llm_labels', 'build_multiclass_dataset']
