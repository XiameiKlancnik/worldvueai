import re
from typing import List, Tuple
import pandas as pd

POLICY_INCLUDE_DEFAULT = [
    'parliament','senate','congress','minister','ministry','cabinet','president','prime minister',
    'election','referendum','coalition','policy','bill','law','legislation','amendment',
    'court','supreme court','constitutional','regulator','commission','authority','sanction',
    'budget','deficit','tax','tariff','central bank','interest rate','immigration','asylum',
    'diplomatic','treaty','summit','union','eu','united nations','nato','security council'
]

OMIT_EXCLUDE_DEFAULT = [
    'celebrity','entertainment','hollywood','showbiz','festival','concert','trailer','premiere',
    'review','recap','box office','netflix','hbo','disney+','prime video','streaming','series',
    'sports','football','soccer','basketball','cricket','tennis','golf','olympics','world cup',
    'fashion','beauty','lifestyle','gossip'
]


def _contains_any(text: str, terms: List[str]) -> bool:
    t = text.lower()
    return any(term in t for term in terms)


def build_balanced_dataset(
    articles: pd.DataFrame,
    *,
    include_terms: List[str] = None,
    exclude_terms: List[str] = None,
    text_cols: List[str] = None,
    id_col: str = 'article_id',
    embedding_col: str = 'embedding',
    max_per_class: int = None
) -> pd.DataFrame:
    """Heuristically label and balance dataset for topic filter training.

    - Positive label (y=1) = omit (entertainment/sports/lifestyle/etc.)
    - Negative label (y=0) = keep (politics/policy-context)

    Ambiguous rows (both include and exclude or neither) are dropped.
    """
    if include_terms is None:
        include_terms = POLICY_INCLUDE_DEFAULT
    if exclude_terms is None:
        exclude_terms = OMIT_EXCLUDE_DEFAULT
    if text_cols is None:
        text_cols = [c for c in ['keywords','frames','category','section','topic','title','summary','text'] if c in articles.columns]

    if id_col not in articles.columns:
        if 'id' in articles.columns:
            articles = articles.copy()
            articles[id_col] = articles['id'].astype(str)
        else:
            raise ValueError(f'Missing id column: {id_col}')
    if embedding_col not in articles.columns:
        raise ValueError(f'Missing embedding column: {embedding_col}')

    df = articles[[id_col, embedding_col] + text_cols].copy()
    combined = df[text_cols].fillna('').astype(str).agg(' '.join, axis=1)

    include_mask = combined.apply(lambda s: _contains_any(s, include_terms))
    exclude_mask = combined.apply(lambda s: _contains_any(s, exclude_terms))

    # Keep only confident heuristics: include XOR exclude
    pos = df[ exclude_mask & ~include_mask ].copy()
    neg = df[ include_mask & ~exclude_mask ].copy()

    # Balance
    n = min(len(pos), len(neg))
    if max_per_class:
        n = min(n, max_per_class)
    pos = pos.sample(n=n, random_state=42) if len(pos) > n else pos
    neg = neg.sample(n=n, random_state=42) if len(neg) > n else neg

    pos['y'] = 1
    neg['y'] = 0

    out = pd.concat([pos, neg], ignore_index=True)
    out = out.sample(frac=1.0, random_state=42).reset_index(drop=True)
    return out
