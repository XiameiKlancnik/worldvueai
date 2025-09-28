"""Hybrid pair sampling for within and cross-country comparisons."""

from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional, Set
import pandas as pd
import numpy as np
from collections import defaultdict
import random
from pathlib import Path

from ..budget.config import BudgetConfig


@dataclass
class PairSample:
    pair_id: str
    a_id: str
    b_id: str
    a_country: str
    b_country: str
    a_lang: str
    b_lang: str
    cluster_id: str
    is_cross_country: bool
    a_outlet: Optional[str] = None
    b_outlet: Optional[str] = None
    pair_type: str = "normal"  # normal, paraphrase_check, entity_swap_check


class HybridPairSampler:
    """
    Samples article pairs for comparison with budget constraints.
    Creates both within-country and cross-country pairs.
    """

    def __init__(self, config: BudgetConfig):
        self.config = config
        self.pairs: List[PairSample] = []
        self.article_degrees: Dict[str, int] = defaultdict(int)
        random.seed(config.seed)
        np.random.seed(config.seed)
        self._pair_sequence = 0

    def _next_pair_id(self, prefix: str, context: Optional[str] = None) -> str:
        """Generate a globally unique pair identifier."""
        components = [prefix]
        if context:
            sanitized = str(context).replace(" ", "_")
            components.append(sanitized)
        components.append(f"{self._pair_sequence:06d}")
        self._pair_sequence += 1
        return "_".join(components)

    def sample_pairs(self, articles_df: pd.DataFrame, clusters_df: pd.DataFrame) -> pd.DataFrame:
        """
        Sample pairs according to budget constraints.

        Args:
            articles_df: DataFrame with article metadata and embeddings
            clusters_df: DataFrame with cluster assignments
        """
        # Merge articles with clusters
        df = articles_df.merge(clusters_df[['article_id', 'cluster_id']], on='article_id')

        # Calculate target numbers
        n_within = int(self.config.target_pairs_total * self.config.within_country_ratio)
        n_cross = self.config.target_pairs_total - n_within

        # Sample within-country pairs
        within_pairs = self._sample_within_country(df, n_within)
        self.pairs.extend(within_pairs)

        # Sample cross-country pairs
        cross_pairs = self._sample_cross_country(df, n_cross)
        self.pairs.extend(cross_pairs)

        # Add validation pairs
        validation_pairs = self._create_validation_pairs(df, max_pairs=100)
        self.pairs.extend(validation_pairs)

        return self._to_dataframe()

    def _sample_within_country(self, df: pd.DataFrame, target_pairs: int) -> List[PairSample]:
        """Sample pairs within the same country."""
        pairs = []
        pairs_per_group = []

        # Group by country and cluster
        for (country, cluster_id), group in df.groupby(['country', 'cluster_id']):
            if len(group) < 2:
                continue

            # Calculate how many pairs to sample from this group
            max_pairs = min(
                self.config.pairs_per_cluster_max,
                len(group) * (len(group) - 1) // 2
            )
            n_pairs = min(max_pairs, self.config.pairs_per_cluster_min)

            if n_pairs > 0:
                pairs_per_group.append((country, cluster_id, group, n_pairs))

        # Distribute target_pairs across groups
        total_available = sum(n for _, _, _, n in pairs_per_group)
        if total_available == 0:
            return pairs

        for country, cluster_id, group, base_pairs in pairs_per_group:
            # Scale pairs proportionally
            n_pairs = int(base_pairs * target_pairs / total_available)
            n_pairs = max(1, min(n_pairs, len(group) * (len(group) - 1) // 2))

            # Sample pairs from this group
            articles = group.to_dict('records')
            sampled = self._sample_from_group(articles, n_pairs, country, cluster_id, False)
            pairs.extend(sampled)

            if len(pairs) >= target_pairs:
                break

        return pairs[:target_pairs]

    def _sample_cross_country(self, df: pd.DataFrame, target_pairs: int) -> List[PairSample]:
        """Sample pairs across countries within the same cluster."""
        pairs = []

        # Group by cluster (ignoring country)
        for cluster_id, cluster_group in df.groupby('cluster_id'):
            countries = cluster_group['country'].unique()
            if len(countries) < 2:
                continue

            # Sample pairs between countries
            n_pairs = min(
                self.config.pairs_per_cluster_max // 2,
                target_pairs - len(pairs)
            )

            for _ in range(n_pairs):
                if len(pairs) >= target_pairs:
                    break

                # Sample two different countries
                c1, c2 = random.sample(list(countries), 2)
                g1 = cluster_group[cluster_group['country'] == c1]
                g2 = cluster_group[cluster_group['country'] == c2]

                if len(g1) == 0 or len(g2) == 0:
                    continue

                # Sample one article from each country
                a1 = g1.sample(n=1).iloc[0]
                a2 = g2.sample(n=1).iloc[0]

                # Check degree constraints
                if (self.article_degrees[a1['article_id']] >= self.config.max_pairs_per_article or
                    self.article_degrees[a2['article_id']] >= self.config.max_pairs_per_article):
                    continue

                pair = PairSample(
                    pair_id=self._next_pair_id("cross", f"{cluster_id}_{c1}_{c2}"),
                    a_id=a1['article_id'],
                    b_id=a2['article_id'],
                    a_country=a1['country'],
                    b_country=a2['country'],
                    a_lang=a1.get('language', 'en'),
                    b_lang=a2.get('language', 'en'),
                    cluster_id=cluster_id,
                    is_cross_country=True,
                    a_outlet=a1.get('outlet'),
                    b_outlet=a2.get('outlet')
                )
                pairs.append(pair)
                self.article_degrees[a1['article_id']] += 1
                self.article_degrees[a2['article_id']] += 1

        return pairs[:target_pairs]

    def _sample_from_group(self, articles: List[dict], n_pairs: int,
                           country: str, cluster_id: str,
                           is_cross: bool) -> List[PairSample]:
        """Sample n_pairs from a group of articles."""
        pairs = []
        article_ids = [a['article_id'] for a in articles]

        # Generate all possible pairs
        possible_pairs = []
        for i in range(len(articles)):
            for j in range(i + 1, len(articles)):
                a1, a2 = articles[i], articles[j]

                # Skip if same outlet and balance_by_outlet is True
                if self.config.balance_by_outlet and a1.get('outlet') == a2.get('outlet'):
                    continue

                # Check degree constraints
                if (self.article_degrees[a1['article_id']] < self.config.max_pairs_per_article and
                    self.article_degrees[a2['article_id']] < self.config.max_pairs_per_article):
                    possible_pairs.append((a1, a2))

        # Sample from possible pairs
        if possible_pairs:
            sampled = random.sample(possible_pairs, min(n_pairs, len(possible_pairs)))
            for a1, a2 in sampled:
                pair = PairSample(
                    pair_id=self._next_pair_id("within", f"{country}_{cluster_id}"),
                    a_id=a1['article_id'],
                    b_id=a2['article_id'],
                    a_country=a1.get('country', country),
                    b_country=a2.get('country', country),
                    a_lang=a1.get('language', 'en'),
                    b_lang=a2.get('language', 'en'),
                    cluster_id=cluster_id,
                    is_cross_country=is_cross,
                    a_outlet=a1.get('outlet'),
                    b_outlet=a2.get('outlet')
                )
                pairs.append(pair)
                self.article_degrees[a1['article_id']] += 1
                self.article_degrees[a2['article_id']] += 1

        return pairs

    def _create_validation_pairs(self, df: pd.DataFrame, max_pairs: int = 100) -> List[PairSample]:
        """Create validation pairs for robustness checks."""
        pairs = []

        # Sample articles for validation
        sample_size = min(max_pairs, len(df) // 10)
        if sample_size == 0:
            return pairs

        sample_df = df.sample(n=sample_size, random_state=self.config.seed)

        for _, row in sample_df.iterrows():
            # Paraphrase check (A vs paraphrase of A)
            pair = PairSample(
                pair_id=self._next_pair_id("validation", row.get('cluster_id', 'unknown')),
                a_id=row['article_id'],
                b_id=row['article_id'],  # Same article
                a_country=row['country'],
                b_country=row['country'],
                a_lang=row.get('language', 'en'),
                b_lang=row.get('language', 'en'),
                cluster_id=row.get('cluster_id', 'unknown'),
                is_cross_country=False,
                pair_type="paraphrase_check"
            )
            pairs.append(pair)

            if len(pairs) >= max_pairs:
                break

        return pairs

    def _to_dataframe(self) -> pd.DataFrame:
        """Convert pairs to DataFrame."""
        records = []
        for pair in self.pairs:
            records.append({
                'pair_id': pair.pair_id,
                'a_id': pair.a_id,
                'b_id': pair.b_id,
                'a_country': pair.a_country,
                'b_country': pair.b_country,
                'a_lang': pair.a_lang,
                'b_lang': pair.b_lang,
                'cluster_id': pair.cluster_id,
                'is_cross_country': pair.is_cross_country,
                'a_outlet': pair.a_outlet,
                'b_outlet': pair.b_outlet,
                'pair_type': pair.pair_type
            })

        return pd.DataFrame(records)

    def save(self, path: Path):
        """Save pairs to parquet file."""
        df = self._to_dataframe()
        df.to_parquet(path, index=False)

    def get_statistics(self) -> Dict[str, any]:
        """Get statistics about sampled pairs."""
        df = self._to_dataframe()
        stats = {
            'total_pairs': len(df),
            'within_country': len(df[~df['is_cross_country']]),
            'cross_country': len(df[df['is_cross_country']]),
            'validation_pairs': len(df[df['pair_type'] != 'normal']),
            'unique_articles': len(set(df['a_id']) | set(df['b_id'])),
            'unique_countries': len(set(df['a_country']) | set(df['b_country'])),
            'unique_clusters': df['cluster_id'].nunique(),
            'avg_degree': np.mean(list(self.article_degrees.values())) if self.article_degrees else 0,
            'max_degree': max(self.article_degrees.values()) if self.article_degrees else 0
        }
        return stats