from __future__ import annotations

import json
import re
from dataclasses import dataclass

from worldvue.judge.prompt import STYLE_AXES
from worldvue.weak.features import extract_features


@dataclass
class _Message:
    content: str


@dataclass
class _Choice:
    message: _Message


@dataclass
class _Usage:
    prompt_tokens: int
    completion_tokens: int


@dataclass
class MockResponse:
    choices: list[_Choice]
    usage: _Usage


class MockOpenAI:
    def __init__(self) -> None:
        self.chat = self.MockChat(self)

    class MockChat:
        def __init__(self, outer: 'MockOpenAI') -> None:
            self.completions = outer.MockCompletions()

    class MockCompletions:
        def create(self, *, messages, **_) -> MockResponse:
            prompt = messages[-1]['content']
            text_a, text_b = self._extract_texts(prompt)
            features_a = extract_features(text_a)
            features_b = extract_features(text_b)
            result = {}
            for axis in STYLE_AXES:
                winner, confidence = self._decide_axis(axis, features_a, features_b)
                result[axis] = {'winner': winner, 'confidence': confidence}
            content = json.dumps(result)
            usage = _Usage(prompt_tokens=len(prompt.split()), completion_tokens=len(content.split()))
            return MockResponse(choices=[_Choice(message=_Message(content=content))], usage=usage)

        def _extract_texts(self, prompt: str) -> tuple[str, str]:
            match = re.search(r'A:\s*(.*)\nB:\s*(.*)', prompt, flags=re.S)
            if not match:
                return '', ''
            return match.group(1).strip(), match.group(2).strip()

        def _decide_axis(self, axis: str, features_a, features_b):
            diff = features_b['token_count'] - features_a['token_count']
            if axis == 'hype':
                metric_a = features_a['exclamation_ratio'] + features_a['caps_ratio']
                metric_b = features_b['exclamation_ratio'] + features_b['caps_ratio']
            elif axis == 'sourcing':
                metric_a = features_a['quote_density']
                metric_b = features_b['quote_density']
            elif axis == 'fight_vs_fix':
                metric_a = features_a['sentence_count']
                metric_b = features_b['sentence_count']
            elif axis == 'certain_vs_caution':
                metric_a = features_a['certainty_count'] - features_a['hedge_count']
                metric_b = features_b['certainty_count'] - features_b['hedge_count']
            else:
                metric_a = features_a['quote_density']
                metric_b = features_b['quote_density'] + diff * 0.0001
            diff_metric = metric_b - metric_a
            if abs(diff_metric) < 0.05:
                return 'tie', 0.5
            winner = 'B' if diff_metric > 0 else 'A'
            confidence = min(0.9, 0.5 + abs(diff_metric))
            return winner, confidence


__all__ = ['MockOpenAI', 'MockResponse']
