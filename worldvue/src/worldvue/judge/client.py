from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

from worldvue.data.types import Pair
from worldvue.judge.prompt import STYLE_AXES, build_prompt
from worldvue.text.truncate import truncate_pair

try:  # pragma: no cover - optional when running tests with mock
    from openai import OpenAI
except Exception:  # pragma: no cover
    OpenAI = None  # type: ignore


class BudgetExceededError(RuntimeError):
    pass


class BudgetAwareJudge:
    def __init__(
        self,
        *,
        budget_usd: float = 10.0,
        model: str = 'gpt-4o-mini',
        temperature: float = 0.0,
        max_tokens: int = 200,
        cost_log_path: Path | str = Path('costs.jsonl'),
        client: Optional[object] = None,
        prompt_rate: float = 0.15 / 1_000_000,
        completion_rate: float = 0.60 / 1_000_000,
    ) -> None:
        self.budget = budget_usd
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.cost_log_path = Path(cost_log_path)
        self.cost_log_path.parent.mkdir(parents=True, exist_ok=True)
        self.prompt_rate = prompt_rate
        self.completion_rate = completion_rate
        self.spent = 0.0
        self.client = client or (OpenAI() if OpenAI else None)
        if self.client is None:
            raise RuntimeError('OpenAI client is required unless a mock client is provided')

    def judge_pair(self, pair: Pair) -> Dict[str, Dict[str, float | str]]:
        if self.spent >= self.budget:
            raise BudgetExceededError(f'Spent ${self.spent:.2f} of ${self.budget:.2f}')
        pair.ensure_articles()
        text_a, text_b = truncate_pair(pair.article_a.text[:1000], pair.article_b.text[:1000])
        payload = build_prompt(text_a, text_b)
        response = self.client.chat.completions.create(
            model=self.model,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            response_format={'type': 'json_object'},
            messages=[
                {'role': 'system', 'content': 'Compare news style. Return JSON only.'},
                {'role': 'user', 'content': payload},
            ],
        )
        tokens_in = getattr(response.usage, 'prompt_tokens', 0)
        tokens_out = getattr(response.usage, 'completion_tokens', 0)
        cost = tokens_in * self.prompt_rate + tokens_out * self.completion_rate
        if self.spent + cost > self.budget + 1e-9:
            raise BudgetExceededError(f'Cost ${cost:.4f} would exceed remaining budget ${self.budget - self.spent:.4f}')
        self.spent += cost
        pair.cost_usd += cost
        result = self._parse_response(response)
        self._log_cost(pair.pair_id, tokens_in, tokens_out, cost)
        pair.llm_labels = result
        return result

    def _parse_response(self, response) -> Dict[str, Dict[str, float | str]]:
        content = response.choices[0].message.content
        data = json.loads(content)
        normalized: Dict[str, Dict[str, float | str]] = {}
        for axis in STYLE_AXES:
            axis_payload = data.get(axis) or {}
            winner = axis_payload.get('winner', 'tie')
            confidence = float(axis_payload.get('confidence', 0.5))
            normalized[axis] = {'winner': winner, 'confidence': confidence}
        return normalized

    def _log_cost(self, pair_id: str, tokens_in: int, tokens_out: int, cost: float) -> None:
        record = {
            'timestamp': datetime.utcnow().isoformat(),
            'pair_id': pair_id,
            'model': self.model,
            'tokens_in': tokens_in,
            'tokens_out': tokens_out,
            'cost_usd': cost,
            'total_spent': self.spent,
        }
        with self.cost_log_path.open('a', encoding='utf-8') as handle:
            handle.write(json.dumps(record) + '\n')


__all__ = ['BudgetAwareJudge', 'BudgetExceededError']
