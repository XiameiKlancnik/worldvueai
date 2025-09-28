from __future__ import annotations

import html
import re
from typing import Iterable

WHITESPACE_RE = re.compile(r'\s+')
HTML_TAG_RE = re.compile(r'<[^>]+>')


def strip_html(value: str) -> str:
    return HTML_TAG_RE.sub(' ', value)


def normalize_whitespace(value: str) -> str:
    return WHITESPACE_RE.sub(' ', value).strip()


def basic_clean(text: str) -> str:
    text = html.unescape(text or '')
    text = strip_html(text)
    text = normalize_whitespace(text)
    return text


def concatenate_text(parts: Iterable[str]) -> str:
    joined = ' '.join(filter(None, parts))
    return normalize_whitespace(joined)


__all__ = ['basic_clean', 'normalize_whitespace', 'strip_html', 'concatenate_text']
