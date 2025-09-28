from __future__ import annotations

from itertools import combinations
from typing import Dict, Iterable, List, Sequence

from worldvue.data.types import Article, Pair


def generate_pairs(
    articles: Iterable[Article],
    clusters: Dict[str, Sequence[str]],
    *,
    max_pairs_per_cluster: int | None = None,
) -> List[Pair]:
    article_index = {article.id: article for article in articles}
    pairs: List[Pair] = []
    for cluster_id, member_ids in clusters.items():
        id_list = [id_ for id_ in member_ids if id_ in article_index]
        combo_iter = combinations(sorted(id_list), 2)
        count = 0
        for article_a_id, article_b_id in combo_iter:
            pair_id = f'{cluster_id}:{article_a_id}:{article_b_id}'
            pair = Pair(
                pair_id=pair_id,
                article_a_id=article_a_id,
                article_b_id=article_b_id,
                cluster_id=cluster_id,
            )
            pair.attach_articles(article_index[article_a_id], article_index[article_b_id])
            pairs.append(pair)
            count += 1
            if max_pairs_per_cluster and count >= max_pairs_per_cluster:
                break
    return pairs


__all__ = ['generate_pairs']
