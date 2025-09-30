"""Parallel LLM-based topic labeling for articles (primary category + confidence)."""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Dict, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Semaphore

import click
import pandas as pd

from .categories import CATEGORIES, DESCRIPTIONS


@dataclass
class TopicLabel:
    article_id: str
    primary: str
    confidence: float
    secondaries: List[str]
    flags: Dict[str, any]
    raw_response: Optional[str] = None


class ParallelTopicJudge:
    def __init__(self, max_workers: int = 8, requests_per_minute: int = 300, max_retries: int = 3):
        self.max_workers = max_workers
        self.max_retries = max_retries
        self.requests_per_minute = requests_per_minute
        self.min_interval = 60.0 / max(1, requests_per_minute)
        self._last_ts = 0.0
        self._rate_lock = Semaphore(1)
        self._setup_client()
        self.stats = {
            'total_requests': 0,
            'successful': 0,
            'failed': 0,
            'retries': 0,
            'api_errors': 0,
            'parse_errors': 0,
        }

    def _setup_client(self):
        import os
        import openai
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            raise RuntimeError('OPENAI_API_KEY not set in environment')
        self.client = openai.OpenAI(api_key=api_key)

    def _wait(self):
        with self._rate_lock:
            now = time.time()
            dt = now - self._last_ts
            if dt < self.min_interval:
                time.sleep(self.min_interval - dt)
            self._last_ts = time.time()

    def _call_llm(self, system_prompt: str, user_prompt: str) -> Optional[str]:
        model = 'gpt-4o-mini'
        for attempt in range(self.max_retries):
            try:
                self._wait()
                self.stats['total_requests'] += 1
                resp = self.client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    temperature=0.1,
                    max_tokens=300,
                    response_format={"type": "json_object"}
                )
                content = resp.choices[0].message.content
                self.stats['successful'] += 1
                return content
            except Exception as e:
                self.stats['api_errors'] += 1
                self.stats['retries'] += 1
                if attempt == self.max_retries - 1:
                    self.stats['failed'] += 1
                    click.echo(f"API error: {str(e)[:120]}", err=True)
                    return None
                time.sleep(2 ** attempt)
        return None

    def _prompts(self, categories: List[str], title: str, text: str) -> tuple[str, str]:
        cat_lines = [f"- {c}: {DESCRIPTIONS.get(c,'')}" for c in categories]
        system = (
            "You are labeling news articles into topical categories for a politics-focused site.\n"
            "Label as POLITICAL if there is any public-interest policy, regulatory, legal, economic, or governance angle,\n"
            "including markets/companies/climate/health education etc. Label as NON-POLITICAL for entertainment, sports,\n"
            "lifestyle, or weather coverage. Pick a PRIMARY category from the list and optional SECONDARIES.\n"
            "If article is in a foreign language, mentally translate first. Respond with strict JSON."
        )
        user = (
            f"Categories (choose primary + optional secondaries):\n{chr(10).join(cat_lines)}\n\n"
            f"Article Title:\n{title[:300]}\n\n"
            f"Article Body (truncated):\n{text[:2000]}\n\n"
            "Respond as:\n{\n  \"primary\": \"<one_of_categories>\",\n  \"confidence\": 0.0-1.0,\n  \"secondaries\": [\"<optional_other_categories>\"],\n  \"rationale\": \"brief reason\"\n}"
        )
        return system, user

    def _parse(self, article_id: str, raw: str) -> TopicLabel:
        try:
            data = json.loads(raw or '{}')
            primary = data.get('primary', '')
            if primary not in CATEGORIES:
                # Try loose mapping by lowercase contains
                lower = str(primary).lower()
                for c in CATEGORIES:
                    if c in lower:
                        primary = c
                        break
            secondaries = [c for c in data.get('secondaries', []) if c in CATEGORIES]
            conf = float(data.get('confidence', 0.0))
            return TopicLabel(article_id=article_id, primary=primary, confidence=conf, secondaries=secondaries, flags={}, raw_response=raw)
        except Exception as e:
            self.stats['parse_errors'] += 1
            return TopicLabel(article_id=article_id, primary='', confidence=0.0, secondaries=[], flags={'parse_error': True, 'error': str(e)[:120]}, raw_response=raw)

    def judge_articles(self, articles_df: pd.DataFrame, output_dir: Optional[Path] = None, categories: Optional[List[str]] = None) -> List[TopicLabel]:
        cats = categories or CATEGORIES
        results: List[TopicLabel] = []
        titles = articles_df.get('title') if 'title' in articles_df.columns else pd.Series(['']*len(articles_df))
        texts = articles_df.get('text') if 'text' in articles_df.columns else pd.Series(['']*len(articles_df))
        ids = (articles_df['article_id'] if 'article_id' in articles_df.columns else articles_df['id']).astype(str)

        with ThreadPoolExecutor(max_workers=self.max_workers) as ex:
            fut2id = {}
            for aid, title, text in zip(ids, titles, texts):
                sys_p, usr_p = self._prompts(cats, str(title or ''), str(text or ''))
                fut = ex.submit(self._call_llm, sys_p, usr_p)
                fut2id[fut] = aid
            processed = 0
            for fut in as_completed(fut2id):
                aid = fut2id[fut]
                raw = fut.result()
                if raw:
                    results.append(self._parse(aid, raw))
                else:
                    results.append(TopicLabel(article_id=aid, primary='', confidence=0.0, secondaries=[], flags={'api_error': True}))
                processed += 1
                if processed % 50 == 0:
                    click.echo(f"Labeled {processed}/{len(fut2id)} articles")

        if output_dir:
            output_dir.mkdir(parents=True, exist_ok=True)
            path = output_dir / 'topic_labels.jsonl'
            with path.open('w', encoding='utf-8') as f:
                for r in results:
                    f.write(json.dumps(asdict(r), ensure_ascii=False) + '\n')
            click.echo(f"Saved labels -> {path}")
        return results
