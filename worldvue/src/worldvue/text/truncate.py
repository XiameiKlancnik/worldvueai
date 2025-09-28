from __future__ import annotations

from typing import Tuple

def smart_truncate(text: str, *, max_tokens: int = 500) -> str:
    if not text:
        return ''
    tokens = text.split()
    if len(tokens) <= max_tokens:
        return text
    truncated = ' '.join(tokens[:max_tokens])
    return truncated


def truncate_pair(text_a: str, text_b: str, *, max_tokens: int = 500) -> Tuple[str, str]:
    return smart_truncate(text_a, max_tokens=max_tokens), smart_truncate(text_b, max_tokens=max_tokens)


__all__ = ['smart_truncate', 'truncate_pair']
