from __future__ import annotations

from datetime import datetime
from typing import Dict, List, Optional, Union

from pydantic import BaseModel, Field, validator

STYLE_AXES = [
    'one_sidedness',
    'hype',
    'sourcing',
    'fight_vs_fix',
    'certain_vs_caution',
]

POLICY_AXES = [
    'econ_left_right',
    'social_lib_cons',
]


class Article(BaseModel):
    id: str
    title: str = ''
    text: str = ''
    source_name: str = ''
    source_country: str = Field('UN', min_length=2, max_length=5)
    language: str = Field('und', min_length=2, max_length=5)
    published_at: datetime
    embedding: Optional[List[float]] = None

    class Config:
        allow_mutation = True
        anystr_strip_whitespace = True

    @validator('text', pre=True, always=True)
    def default_text(cls, value: Optional[str]) -> str:  # noqa: N805
        return value or ''

    @validator('title', pre=True, always=True)
    def default_title(cls, value: Optional[str]) -> str:  # noqa: N805
        return value or ''


class Pair(BaseModel):
    pair_id: str
    article_a_id: str
    article_b_id: str
    cluster_id: str
    weak_labels: Dict[str, float] = Field(default_factory=dict)
    weak_confidence: Dict[str, float] = Field(default_factory=dict)
    llm_labels: Optional[Dict[str, Dict[str, Union[str, float]]]] = None
    cost_usd: float = 0.0
    article_a: Optional[Article] = None
    article_b: Optional[Article] = None

    class Config:
        arbitrary_types_allowed = True
        allow_mutation = True

    def attach_articles(self, article_a: Article, article_b: Article) -> None:
        self.article_a = article_a
        self.article_b = article_b

    def as_tuple(self) -> tuple[str, str]:
        return self.article_a_id, self.article_b_id

    def ensure_articles(self) -> None:
        if not self.article_a or not self.article_b:
            raise ValueError('Pair requires attached Article instances for processing')


__all__ = ['Article', 'Pair', 'STYLE_AXES', 'POLICY_AXES']
