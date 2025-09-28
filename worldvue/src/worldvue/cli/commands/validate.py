from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable, List

from rich.console import Console

from worldvue.data.loaders import dump_pairs, load_articles, load_cached_pairs
from worldvue.data.types import Pair
from worldvue.judge.client import BudgetAwareJudge, BudgetExceededError
from worldvue.judge.mock import MockOpenAI
from worldvue.pairs.sampler import select_validation_pairs
from worldvue.weak.labelers import DEFAULT_LABELERS
from worldvue.weak.validator import weak_vs_llm_agreement

console = Console()


def _load_pairs(path: Path) -> List[Pair]:
    payloads = load_cached_pairs(path)
    return [Pair(**payload) for payload in payloads]


def _attach_articles(pairs: Iterable[Pair], articles):
    index = {article.id: article for article in articles}
    for pair in pairs:
        if pair.article_a_id in index and pair.article_b_id in index:
            pair.attach_articles(index[pair.article_a_id], index[pair.article_b_id])


def _client_from_env():
    if os.getenv('WORLDVUE_JUDGE_MOCK', '0') == '1':
        return MockOpenAI()
    return None


def run_validation(
    *,
    sample: int,
    budget: float,
    pairs_path: Path,
    articles_path: Path,
    output: Path,
) -> None:
    console.log('Preparing validation run...')
    pairs = _load_pairs(pairs_path)
    articles = load_articles(articles_path)
    _attach_articles(pairs, articles)
    selected = select_validation_pairs(
        pairs,
        budget=sample,
        calibration=min(sample, sample // 2),
        uncertainty=min(50, sample // 2),
        test=max(0, sample - sample // 2 - min(50, sample // 2)),
        labelers=DEFAULT_LABELERS,
    )
    console.log(f'Selected {len(selected)} pairs for validation.')
    judge = BudgetAwareJudge(budget_usd=budget, client=_client_from_env(), cost_log_path=Path('costs.jsonl'))
    judged = 0
    for pair in selected:
        try:
            result = judge.judge_pair(pair)
        except BudgetExceededError as error:
            console.log(f'[red]Budget exhausted early:[/red] {error}')
            break
        pair.llm_labels = result
        judged += 1
    console.log(f'Judged {judged} pairs, spent ${judge.spent:.4f}')
    metrics = weak_vs_llm_agreement(selected)
    console.print('[bold]Validation results[/bold]')
    for axis, payload in metrics.items():
        console.print(f'- {axis}: agreement={payload["agreement_rate"]:.2%} confidence={payload["confidence"]:.2f}')
    dump_pairs(output, [pair.dict(exclude={'article_a', 'article_b'}) for pair in selected])
    console.log(f'Validation pairs saved to {output}')


__all__ = ['run_validation']
