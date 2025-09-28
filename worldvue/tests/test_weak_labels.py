from datetime import datetime

from worldvue.data.types import Article, Pair
from worldvue.weak.labelers import HypeLabeler, QuoteBalanceLabeler
from worldvue.weak.model import EMLabelModel


def _make_pair(text_a: str, text_b: str) -> Pair:
    article_a = Article(
        id='A',
        title='Article A',
        text=text_a,
        source_name='Source A',
        source_country='US',
        language='en',
        published_at=datetime.utcnow(),
    )
    article_b = Article(
        id='B',
        title='Article B',
        text=text_b,
        source_name='Source B',
        source_country='US',
        language='en',
        published_at=datetime.utcnow(),
    )
    pair = Pair(pair_id='pair-A-B', article_a_id='A', article_b_id='B', cluster_id='cluster-1')
    pair.attach_articles(article_a, article_b)
    return pair


def test_hype_labeler_detects_excitement():
    pair = _make_pair('Calm coverage.', 'BREAKING NEWS!!! SHOCKING!')
    labeler = HypeLabeler()
    result = labeler.label_with_confidence(pair)
    assert result.axis == 'hype'
    assert result.score > 0  # Article B more hype
    assert 0 <= result.confidence <= 1


def test_quote_balance_labeler_prefers_quotes():
    pair = _make_pair('Statement with no quotes.', '"Expert" said it is fine.')
    labeler = QuoteBalanceLabeler()
    result = labeler.label_with_confidence(pair)
    assert result.axis == 'one_sidedness'
    assert result.score > 0


def test_em_label_model_assigns_scores():
    pair = _make_pair('Calm text.', 'Another calm text but with "sources"!')
    model = EMLabelModel([HypeLabeler(), QuoteBalanceLabeler()])
    model.fit([pair])
    assert 'hype' in pair.weak_labels
    assert 'one_sidedness' in pair.weak_labels
