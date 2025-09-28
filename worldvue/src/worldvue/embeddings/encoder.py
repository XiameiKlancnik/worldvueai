from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional

import numpy as np

from worldvue.data.types import Article
from worldvue.embeddings.store import attach_embeddings, load_store, persist_store

try:
    from sentence_transformers import SentenceTransformer
except Exception:  # pragma: no cover - optional heavy dependency
    SentenceTransformer = None  # type: ignore

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
except Exception:  # pragma: no cover
    TfidfVectorizer = None  # type: ignore


DEFAULT_MODEL = 'paraphrase-multilingual-MiniLM-L12-v2'


@dataclass
class EncodingResult:
    article_id: str
    embedding: List[float]


class EmbeddingEncoder:
    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        *,
        cache_path: str | None = None,
    ) -> None:
        self.model_name = model_name
        self.cache_path = cache_path
        self._model: Optional[SentenceTransformer] = None
        self._vectorizer = None
        self._store = load_store(cache_path) if cache_path else load_store()

    @property
    def model(self):
        if self._model is None:
            if SentenceTransformer is None:
                raise RuntimeError('sentence-transformers is required for EmbeddingEncoder')
            self._model = SentenceTransformer(self.model_name)
        return self._model

    def _vectorizer_model(self):  # fallback for tests without transformer model
        if TfidfVectorizer is None:
            raise RuntimeError('scikit-learn is required for TF-IDF fallback encoder')
        if self._vectorizer is None:
            self._vectorizer = TfidfVectorizer(max_features=768)
        return self._vectorizer

    def encode_texts(self, texts: Iterable[str], batch_size: int = 32) -> np.ndarray:
        texts = [text or '' for text in texts]
        if SentenceTransformer is None:
            vectorizer = self._vectorizer_model()
            matrix = vectorizer.fit_transform(texts)
            return matrix.toarray()
        return np.asarray(self.model.encode(texts, batch_size=batch_size, show_progress_bar=False))

    def encode_articles(
        self,
        articles: Iterable[Article],
        *,
        batch_size: int = 32,
        refresh: bool = False,
    ) -> List[EncodingResult]:
        article_list = list(articles)
        attach_embeddings(article_list, self._store)
        missing = [article for article in article_list if refresh or not article.embedding]
        if not missing:
            return [EncodingResult(article.id, article.embedding or []) for article in article_list]
        texts = [article.text or article.title for article in missing]
        embeddings = self.encode_texts(texts, batch_size=batch_size)
        results: List[EncodingResult] = []
        for article, embedding in zip(missing, embeddings):
            vector = embedding.tolist()
            article.embedding = vector
            self._store[article.id] = vector
            results.append(EncodingResult(article.id, vector))
        persist_store(self._store, self.cache_path) if self.cache_path else persist_store(self._store)
        for article in article_list:
            if article.id in self._store:
                article.embedding = self._store[article.id]
        if not results:
            results = [EncodingResult(article.id, article.embedding or []) for article in article_list]
        return results


__all__ = ['EmbeddingEncoder', 'EncodingResult', 'DEFAULT_MODEL']
