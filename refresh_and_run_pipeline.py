#!/usr/bin/env python3
"""End-to-end helper: CSV -> parquet -> embeddings -> optimized pipeline."""

import argparse
import subprocess
import sys
from pathlib import Path

import pandas as pd
from worldvue.embeddings.store import load_store


def run_command(cmd, cwd=None):
    """Run a subprocess command, streaming output and raising on failure."""
    print(f"\n>>> Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=cwd)
    if result.returncode != 0:
        raise SystemExit(f"Command failed with exit code {result.returncode}: {' '.join(cmd)}")


def convert_csv_to_parquet(csv_path: Path, parquet_path: Path):
    """Convert CSV file to parquet format."""
    print(f"Converting CSV -> parquet: {csv_path} -> {parquet_path}")
    if not csv_path.exists():
        raise SystemExit(f"CSV file not found: {csv_path}")
    df = pd.read_csv(csv_path)
    print(f"  Loaded {len(df):,} rows from CSV")
    parquet_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(parquet_path, index=False)
    print(f"  Wrote parquet with {len(df):,} rows")


def filter_quality(parquet_path: Path, min_words: int, require_full_text: bool, drop_empty_text: bool) -> tuple[int, int]:
    """Enforce has_full_text, non-empty text, and minimum word count before embedding."""
    df = pd.read_parquet(parquet_path)
    initial = len(df)
    if initial == 0:
        print('Parquet contains no rows; skipping quality filter')
        return 0, 0

    if 'text' in df.columns:
        text_series = df['text'].fillna('').astype(str)
    else:
        text_series = pd.Series([''] * initial)

    keep_mask = pd.Series([True] * initial)

    if drop_empty_text:
        keep_mask &= text_series.str.strip().str.len() > 0

    if require_full_text and 'has_full_text' in df.columns:
        keep_mask &= df['has_full_text'].fillna(False).astype(bool)

    if 'word_count' in df.columns:
        word_counts = df['word_count'].fillna(0).astype(int)
    else:
        word_counts = text_series.str.split().str.len().fillna(0).astype(int)
    keep_mask &= word_counts >= min_words

    filtered = df[keep_mask].copy()
    dropped = initial - len(filtered)
    if dropped:
        print(f"Quality filter removed {dropped:,} / {initial:,} rows (min_words={min_words}, require_full_text={require_full_text}, drop_empty_text={drop_empty_text})")
    else:
        print('Quality filter removed 0 rows')

    filtered.to_parquet(parquet_path, index=False)
    return initial, dropped


def filter_articles_by_terms(parquet_path: Path, terms: list[str]) -> tuple[int, int]:
    """Remove articles whose metadata contains any of the provided terms."""
    if not terms:
        print('No exclude keywords provided; skipping article filter')
        return 0, 0

    df = pd.read_parquet(parquet_path)
    initial = len(df)
    if initial == 0:
        print('Parquet contains no rows; skipping article filter')
        return 0, 0

    terms_lower = [term.lower() for term in terms]
    candidate_cols = [col for col in ('keywords', 'frames', 'category', 'section', 'topic', 'title', 'summary', 'text') if col in df.columns]

    if not candidate_cols:
        print('No textual columns available for filtering; skipping article filter')
        return initial, 0

    combined = df[candidate_cols].fillna('').astype(str)
    search_series = combined.agg(' '.join, axis=1).str.lower()
    drop_mask = search_series.apply(lambda text: any(term in text for term in terms_lower))

    filtered_df = df[~drop_mask].copy()
    dropped = initial - len(filtered_df)
    if dropped:
        print(f"Filtered out {dropped:,} / {initial:,} articles based on exclude terms: {terms}")
    else:
        print('Article filter removed 0 rows')

    filtered_df.to_parquet(parquet_path, index=False)
    return initial, dropped


def write_articles_with_embeddings(parquet_path: Path, cache_path: Path, output_path: Path) -> None:
    """Persist article parquet with updated embeddings from cache."""
    df = pd.read_parquet(parquet_path)
    store = load_store(cache_path)
    if not store:
        print('Warning: embedding store appears empty; writing original parquet')
    id_column = None
    if 'article_id' in df.columns:
        id_column = 'article_id'
    elif 'id' in df.columns:
        df['article_id'] = df['id'].astype(str)
        id_column = 'article_id'
    if id_column is None:
        raise SystemExit('Parquet must contain an id or article_id column to attach embeddings')
    original_embeddings = df['embedding'] if 'embedding' in df.columns else None
    id_series = df[id_column].astype(str)
    df['embedding'] = id_series.map(store)
    if 'country' not in df.columns and 'source_country' in df.columns:
        df['country'] = df['source_country']
    if original_embeddings is not None:
        df['embedding'] = df['embedding'].where(df['embedding'].notna(), original_embeddings)
    missing = df['embedding'].isna().sum()
    if missing:
        print(f'Warning: {missing} rows missing embeddings in cache; leaving NaN entries')
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)
    print(f'Updated parquet with embeddings -> {output_path}')

def rebuild_embeddings(parquet_path: Path, cache_path: Path, refresh: bool = True, batch_size: int = 64, num_workers: int = 0):
    """Invoke the CLI embed command to rebuild article embeddings."""
    if not parquet_path.exists():
        raise SystemExit(f"Parquet file not found for embeddings: {parquet_path}")
    cmd = [
        sys.executable,
        '-m', 'worldvue.cli.main',
        'embed',
        '--input', str(parquet_path),
        '--cache', str(cache_path),
        '--batch-size', str(batch_size),
        '--num-workers', str(num_workers)
    ]
    if refresh:
        cmd.append('--refresh')
    run_command(cmd)


def run_optimized_pipeline(script_path: Path):
    """Run the optimized pipeline helper script."""
    if not script_path.exists():
        raise SystemExit(f"Pipeline script not found: {script_path}")
    run_command([sys.executable, str(script_path)])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Refresh WorldVue artifacts from Supabase CSV.")
    parser.set_defaults(drop_empty_text=True)
    parser.add_argument('--csv', default='all_articles.csv', help='Input CSV exported from Supabase')
    parser.add_argument('--parquet', default='all_articles.parquet', help='Intermediate parquet output')
    parser.add_argument('--output-parquet', default='articles_with_embeddings.parquet', help='Parquet file with embeddings for downstream steps')
    parser.add_argument('--cache', default='artifacts/embeddings_cache', help='Embedding cache directory')
    parser.add_argument('--pipeline-script', default='run_optimized_pipeline.py', help='Pipeline runner script')
    parser.add_argument('--batch-size', type=int, default=64, help='Embedding batch size')
    parser.add_argument('--embedding-workers', type=int, default=0, help='Number of dataloader workers for embeddings')
    parser.add_argument('--min-words', type=int, default=150, help='Minimum word count to keep')
    parser.add_argument('--require-full-text', dest='require_full_text', action='store_true', default=True, help='Require has_full_text=True if available')
    parser.add_argument('--allow-partial-text', dest='require_full_text', action='store_false', help='Allow rows even if has_full_text is False/NA')
    parser.add_argument('--keep-empty-text', action='store_false', dest='drop_empty_text', help='Allow rows with empty text (default: drop)')
    parser.add_argument('--exclude-keywords', default='celebrity,entertainment,sports,lifestyle', help='Comma-separated substrings that cause an article to be dropped')
    parser.add_argument('--no-filter', action='store_true', help='Disable article filtering step')
    parser.add_argument('--skip-convert', action='store_true', help='Skip CSV->parquet conversion')
    parser.add_argument('--skip-embed', action='store_true', help='Skip embedding rebuild')
    parser.add_argument('--skip-pipeline', action='store_true', help='Skip running optimized pipeline')
    return parser.parse_args()


def main():
    args = parse_args()

    csv_path = Path(args.csv)
    parquet_path = Path(args.parquet)
    cache_path = Path(args.cache)
    pipeline_script = Path(args.pipeline_script)
    output_parquet = Path(args.output_parquet)

    if not args.skip_convert:
        convert_csv_to_parquet(csv_path, parquet_path)
    else:
        print('Skipping CSV->parquet conversion as requested')

    filter_quality(parquet_path, args.min_words, args.require_full_text, args.drop_empty_text)

    if args.no_filter:
        print('Article filtering disabled (--no-filter provided)')
    else:
        base_terms = [term.strip() for term in args.exclude_keywords.split(',') if term.strip()]
        commercial_terms = [
            'sponsored', 'sponsored content', 'advertorial', 'press release', 'prnewswire', 'newswire',
            'paid partnership', 'affiliate', 'coupon', 'promo code', 'discount', 'sale', 'buy now',
            'brand partnership', 'deal', 'offer'
        ]
        exclude_terms = list(dict.fromkeys(base_terms + commercial_terms))
        if exclude_terms:
            initial, dropped = filter_articles_by_terms(parquet_path, exclude_terms)
            if initial:
                print(f"Articles remaining after filter: {initial - dropped:,}")
        else:
            print('Exclude keywords list empty; skipping article filter')

    if not args.skip_embed:
        rebuild_embeddings(parquet_path, cache_path, refresh=True, batch_size=args.batch_size, num_workers=args.embedding_workers)
    else:
        print('Skipping embedding rebuild as requested')

    write_articles_with_embeddings(parquet_path, cache_path, output_parquet)

    if not args.skip_pipeline:
        run_optimized_pipeline(pipeline_script)
    else:
        print('Skipping optimized pipeline run as requested')

    print('\nAll requested steps completed successfully.')


if __name__ == '__main__':
    main()



