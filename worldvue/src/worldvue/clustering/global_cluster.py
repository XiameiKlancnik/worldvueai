"""Global multilingual topic clustering for cross-country comparisons."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from pathlib import Path
import json

try:
    import hdbscan
    HAS_HDBSCAN = True
except ImportError:
    HAS_HDBSCAN = False

try:
    from sklearn.cluster import KMeans
    from sklearn.feature_extraction.text import TfidfVectorizer
except ImportError as e:
    raise RuntimeError('scikit-learn must be installed for clustering') from e


@dataclass
class ClusterInfo:
    cluster_id: str
    size: int
    keywords: List[str]
    summary: str
    centroid: Optional[np.ndarray] = None


class GlobalTopicClusterer:
    """
    Clusters articles globally (not per-country) to enable cross-country pairing
    within the same topic.
    """

    def __init__(self, min_cluster_size: int = 6, max_clusters: int = 100,
                 use_hdbscan: bool = True):
        self.min_cluster_size = min_cluster_size
        self.max_clusters = max_clusters
        self.use_hdbscan = use_hdbscan and HAS_HDBSCAN
        self.clusters_: Dict[str, ClusterInfo] = {}
        self.labels_: Optional[np.ndarray] = None

    def fit(self, embeddings: np.ndarray, texts: List[str]) -> 'GlobalTopicClusterer':
        """
        Fit clustering on embeddings with text for keyword extraction.

        Args:
            embeddings: Article embeddings (n_articles, embedding_dim)
            texts: Article texts for keyword extraction
        """
        if self.use_hdbscan:
            clusterer = hdbscan.HDBSCAN(
                min_cluster_size=self.min_cluster_size,
                metric='cosine',
                cluster_selection_method='eom',
                prediction_data=True
            )
            self.labels_ = clusterer.fit_predict(embeddings)
        else:
            # Fallback to k-means
            n_clusters = min(self.max_clusters, len(embeddings) // self.min_cluster_size)
            n_clusters = max(2, n_clusters)
            clusterer = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            self.labels_ = clusterer.fit_predict(embeddings)

        # Extract keywords and summaries for each cluster
        self._extract_cluster_info(embeddings, texts)

        return self

    def _extract_cluster_info(self, embeddings: np.ndarray, texts: List[str]):
        """Extract keywords and create summaries for each cluster."""
        unique_labels = np.unique(self.labels_)
        valid_labels = [l for l in unique_labels if l != -1]

        # TF-IDF for keyword extraction
        vectorizer = TfidfVectorizer(
            max_features=100,
            stop_words='english',
            ngram_range=(1, 2),
            max_df=0.8,
            min_df=2
        )

        for label in valid_labels:
            mask = self.labels_ == label
            cluster_texts = [texts[i] for i, m in enumerate(mask) if m]
            cluster_embeddings = embeddings[mask]

            if len(cluster_texts) < self.min_cluster_size:
                continue

            # Extract keywords
            try:
                tfidf_matrix = vectorizer.fit_transform(cluster_texts)
                feature_names = vectorizer.get_feature_names_out()
                scores = np.array(tfidf_matrix.sum(axis=0)).flatten()
                top_indices = scores.argsort()[-10:][::-1]
                keywords = [feature_names[i] for i in top_indices]
            except:
                keywords = []

            # Compute centroid
            centroid = cluster_embeddings.mean(axis=0)

            # Create neutral summary (placeholder - would use LLM in production)
            summary = f"Topic cluster {label}: {', '.join(keywords[:3])}"

            cluster_id = f"global_{label}"
            self.clusters_[cluster_id] = ClusterInfo(
                cluster_id=cluster_id,
                size=len(cluster_texts),
                keywords=keywords,
                summary=summary,
                centroid=centroid
            )

    def predict(self, embeddings: np.ndarray) -> np.ndarray:
        """Assign new articles to existing clusters."""
        if self.labels_ is None:
            raise ValueError("Clusterer must be fitted first")

        # Assign to nearest cluster centroid
        predictions = []
        cluster_centroids = {
            cid: info.centroid
            for cid, info in self.clusters_.items()
            if info.centroid is not None
        }

        for emb in embeddings:
            min_dist = float('inf')
            best_cluster = -1

            for cid, centroid in cluster_centroids.items():
                # Cosine distance
                dist = 1 - np.dot(emb, centroid) / (np.linalg.norm(emb) * np.linalg.norm(centroid))
                if dist < min_dist:
                    min_dist = dist
                    best_cluster = int(cid.split('_')[1])

            predictions.append(best_cluster)

        return np.array(predictions)

    def to_dataframe(self, article_ids: List[str]) -> pd.DataFrame:
        """Convert clustering results to DataFrame."""
        if self.labels_ is None:
            raise ValueError("Clusterer must be fitted first")

        records = []
        for idx, (aid, label) in enumerate(zip(article_ids, self.labels_)):
            if label == -1:
                continue

            cluster_id = f"global_{label}"
            if cluster_id in self.clusters_:
                info = self.clusters_[cluster_id]
                records.append({
                    'article_id': aid,
                    'cluster_id': cluster_id,
                    'cluster_size': info.size,
                    'cluster_keywords': ', '.join(info.keywords[:5]),
                    'cluster_summary': info.summary
                })

        return pd.DataFrame(records)

    def save(self, path: Path):
        """Save cluster information to file."""
        data = {
            'clusters': {
                cid: {
                    'cluster_id': info.cluster_id,
                    'size': info.size,
                    'keywords': info.keywords,
                    'summary': info.summary,
                    'centroid': info.centroid.tolist() if info.centroid is not None else None
                }
                for cid, info in self.clusters_.items()
            },
            'min_cluster_size': self.min_cluster_size,
            'max_clusters': self.max_clusters
        }

        with open(path, 'w') as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load(cls, path: Path) -> 'GlobalTopicClusterer':
        """Load cluster information from file."""
        with open(path, 'r') as f:
            data = json.load(f)

        clusterer = cls(
            min_cluster_size=data['min_cluster_size'],
            max_clusters=data['max_clusters']
        )

        for cid, info in data['clusters'].items():
            clusterer.clusters_[cid] = ClusterInfo(
                cluster_id=info['cluster_id'],
                size=info['size'],
                keywords=info['keywords'],
                summary=info['summary'],
                centroid=np.array(info['centroid']) if info['centroid'] else None
            )

        return clusterer