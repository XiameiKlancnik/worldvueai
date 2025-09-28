import json
from datetime import datetime

import pytest

from worldvue.data.types import Article, Pair
from worldvue.judge.client import BudgetAwareJudge, BudgetExceededError
from worldvue.judge.mock import MockOpenAI


def _make_pair() -> Pair:
    article_a = Article(
        id='A1',
        title='Neutral',
        text='Calm reporting with quotes: "source".',
        source_name='SourceA',
        source_country='US',
        language='en',
        published_at=datetime.utcnow(),
    )
    article_b = Article(
        id='B1',
        title='Excited',
        text='BREAKING NEWS!!! Action plan announced.',
        source_name='SourceB',
        source_country='US',
        language='en',
        published_at=datetime.utcnow(),
    )
    pair = Pair(pair_id='pair-1', article_a_id='A1', article_b_id='B1', cluster_id='cluster-1')
    pair.attach_articles(article_a, article_b)
    return pair


def test_budget_judge_with_mock(tmp_path):
    pair = _make_pair()
    cost_log = tmp_path / 'costs.jsonl'
    judge = BudgetAwareJudge(budget_usd=0.50, client=MockOpenAI(), cost_log_path=cost_log)
    result = judge.judge_pair(pair)
    assert set(result.keys()) == {'one_sidedness', 'hype', 'sourcing', 'fight_vs_fix', 'certain_vs_caution'}
    assert cost_log.exists()
    logged = json.loads(cost_log.read_text().strip())
    assert logged['pair_id'] == pair.pair_id


def test_budget_exceeded_raises(tmp_path):
    pair = _make_pair()
    judge = BudgetAwareJudge(budget_usd=1e-6, client=MockOpenAI(), cost_log_path=tmp_path / 'costs.jsonl')
    with pytest.raises(BudgetExceededError):
        judge.judge_pair(pair)
