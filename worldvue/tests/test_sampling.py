from datetime import datetime

from worldvue.data.types import Article, Pair
from worldvue.pairs.sampler import select_validation_pairs


class StubLabeler:
    axis = 'hype'

    def label(self, pair: Pair):
        return {self.axis: (-1.0 if pair.pair_id.endswith('0') else 1.0)}


def _make_pair(idx: int) -> Pair:
    article_a = Article(
        id=f'A{idx}',
        title='A',
        text='A',
        source_name='S',
        source_country='US',
        language='en',
        published_at=datetime.utcnow(),
    )
    article_b = Article(
        id=f'B{idx}',
        title='B',
        text='B',
        source_name='S',
        source_country='US',
        language='en',
        published_at=datetime.utcnow(),
    )
    pair = Pair(
        pair_id=f'pair-{idx}',
        article_a_id=article_a.id,
        article_b_id=article_b.id,
        cluster_id=f'cluster-{idx % 3}',
    )
    pair.attach_articles(article_a, article_b)
    return pair


def test_select_validation_pairs_respects_budget():
    pairs = [_make_pair(i) for i in range(30)]
    selected = select_validation_pairs(pairs, budget=12, labelers=[StubLabeler()])
    assert len(selected) == 12
    cluster_ids = {pair.cluster_id for pair in selected}
    assert len(cluster_ids) >= 3
