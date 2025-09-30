"""Utility to export full-text articles from Supabase to Parquet (all languages)."""
from __future__ import annotations

import os
import sys
import json
from pathlib import Path
from typing import List, Dict, Any

import pandas as pd


def ensure_httpx_proxy_support() -> None:
    try:
        import httpx
        import inspect
    except Exception:
        return
    client = httpx.Client
    if getattr(client, '_legacy_proxy_patch', False):
        return
    sig = inspect.signature(client.__init__)
    if 'proxy' in sig.parameters:
        client._legacy_proxy_patch = True
        return
    original_init = client.__init__

    def patched_init(self, *args, proxy=None, **kwargs):
        if proxy is not None and 'proxies' not in kwargs:
            kwargs['proxies'] = proxy
        return original_init(self, *args, **kwargs)

    client.__init__ = patched_init

    if hasattr(httpx, 'AsyncClient'):
        async_client = httpx.AsyncClient
        async_sig = inspect.signature(async_client.__init__)
        if 'proxy' not in async_sig.parameters:
            original_async = async_client.__init__

            def async_patched(self, *args, proxy=None, **kwargs):
                if proxy is not None and 'proxies' not in kwargs:
                    kwargs['proxies'] = proxy
                return original_async(self, *args, **kwargs)

            async_client.__init__ = async_patched

    client._legacy_proxy_patch = True


ensure_httpx_proxy_support()


def export_fulltext(out_parquet: str,
                    *,
                    min_words: int = 200,
                    table: str = 'articles',
                    batch_size: int = 1000,
                    add_raw_dump: bool = False) -> None:
    """Export full-text articles from Supabase to a Parquet file (all languages)."""
    from supabase import create_client

    supabase_url = os.getenv('SUPABASE_URL')
    supabase_key = os.getenv('SUPABASE_SERVICE_ROLE_KEY') or os.getenv('SUPABASE_ANON_KEY')
    if not supabase_url or not supabase_key:
        print('ERROR: SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY (or ANON) must be set', file=sys.stderr)
        sys.exit(1)

    client = create_client(supabase_url, supabase_key)

    print(f'Fetching {table} with has_full_text=True, batch_size={batch_size}')
    rows: List[Dict[str, Any]] = []
    offset = 0
    while True:
        query = (client.table(table)
                 .select('*')
                 .eq('has_full_text', True)
                 .range(offset, offset + batch_size - 1))
        result = query.execute()
        batch = result.data or []
        if not batch:
            break
        rows.extend(batch)
        offset += batch_size
        print(f'  fetched {len(rows)} rows so far...')
        if len(batch) < batch_size:
            break

    if not rows:
        print('No rows returned from Supabase.')
        return

    if add_raw_dump:
        Path('artifacts').mkdir(exist_ok=True)
        dump_path = Path('artifacts/supabase_raw_dump.jsonl')
        with dump_path.open('w', encoding='utf-8') as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + '\n')
        print(f'Wrote raw dump -> {dump_path}')

    df = pd.DataFrame(rows)
    print(f'Total fetched: {len(df)}')

    summary = df.get('summary').fillna('').astype(str) if 'summary' in df.columns else ''
    content = df.get('content').fillna('').astype(str) if 'content' in df.columns else ''
    text = (summary + '\n\n' + content).str.strip() if isinstance(summary, pd.Series) else content

    if 'word_count' in df.columns:
        word_count = df['word_count'].fillna(0).astype(int)
    else:
        word_count = text.str.split().str.len().fillna(0).astype(int)

    keep_mask = word_count >= min_words
    filtered = df[keep_mask].copy()
    filtered['text'] = text[keep_mask]
    filtered['word_count'] = word_count[keep_mask]

    if 'id' not in filtered.columns:
        filtered['id'] = filtered.get('article_id', filtered.index).astype(str)
    filtered['article_id'] = filtered['id'].astype(str)

    if 'language' not in filtered.columns:
        filtered['language'] = 'und'

    cols = [
        'id', 'title', 'text', 'source_name', 'source_country', 'language',
        'published_at', 'url', 'keywords', 'word_count', 'has_full_text',
        'article_id'
    ]
    for col in cols:
        if col not in filtered.columns:
            filtered[col] = None

    out_path = Path(out_parquet)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    filtered[cols].to_parquet(out_path, index=False)
    print(f'Saved {len(filtered)} rows to {out_path} (min_words={min_words})')


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Export Supabase full-text articles to Parquet (all languages).')
    parser.add_argument('--out', default='all_articles.parquet', help='Output parquet path')
    parser.add_argument('--min-words', type=int, default=200, help='Minimum word count')
    parser.add_argument('--table', default='articles', help='Supabase table name')
    parser.add_argument('--batch-size', type=int, default=1000, help='Pagination size')
    parser.add_argument('--raw-dump', action='store_true', help='Write raw JSONL to artifacts/')
    args = parser.parse_args()
    export_fulltext(args.out, min_words=args.min_words, table=args.table, batch_size=args.batch_size, add_raw_dump=args.raw_dump)
