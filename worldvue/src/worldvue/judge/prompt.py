from __future__ import annotations

STYLE_AXES = [
    'one_sidedness',
    'hype',
    'sourcing',
    'fight_vs_fix',
    'certain_vs_caution',
]

JUDGE_PROMPT = """Compare articles A and B on these style dimensions.
Return JSON only:
{{
  "one_sidedness": {{"winner": "A|B|tie", "confidence": 0.0-1.0}},
  "hype": {{"winner": "A|B|tie", "confidence": 0.0-1.0}},
  "sourcing": {{"winner": "A|B|tie", "confidence": 0.0-1.0}},
  "fight_vs_fix": {{"winner": "A|B|tie", "confidence": 0.0-1.0}},
  "certain_vs_caution": {{"winner": "A|B|tie", "confidence": 0.0-1.0}}
}}

A: {text_a}
B: {text_b}"""


def build_prompt(text_a: str, text_b: str) -> str:
    return JUDGE_PROMPT.format(text_a=text_a, text_b=text_b)


__all__ = ['JUDGE_PROMPT', 'STYLE_AXES', 'build_prompt']
