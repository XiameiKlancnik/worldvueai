#!/usr/bin/env python3
"""Pipeline: fetch latest Supabase articles and produce filtered embeddings."""

import argparse
import subprocess
import sys
from pathlib import Path

from export_supabase_fulltext import export_fulltext
from dotenv import load_dotenv


def run_command(cmd):
    print(f"\n>>> Running: {' '.join(str(c) for c in cmd)}")
    result = subprocess.run(cmd)
    if result.returncode != 0:
        raise SystemExit(f"Command failed with exit code {result.returncode}")


def main():
    parser = argparse.ArgumentParser(description="Fetch latest Supabase data and build embeddings")
    parser.add_argument('--out-dir', default='artifacts/supabase_run', help='Working directory for exports')
    parser.add_argument('--min-words', type=int, default=200, help='Minimum word count for quality filter')
    parser.add_argument('--batch-size', type=int, default=1000, help='Supabase fetch batch size')
    parser.add_argument('--reuse-cache', action='store_true', help='Reuse existing embeddings cache (skip embedding step)')
    parser.add_argument('--embedding-batch', type=int, default=64, help='Embedding batch size')
    parser.add_argument('--embedding-workers', type=int, default=0, help='Number of dataloader workers for embeddings')
    parser.add_argument('--exclude-keywords', default='celebrity,entertainment,sports,lifestyle', help='Comma-separated substrings to drop (only used when --enable-subject-filter)')
    parser.add_argument('--enable-subject-filter', action='store_true', help='Apply keyword filtering for non-political content')
    parser.add_argument('--skip-pipeline', action='store_true', help='Skip running optimized pipeline after embeddings')
    args = parser.parse_args()

    load_dotenv()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    raw_parquet = out_dir / 'all_articles.parquet'
    print(f"\n=== Step 1: Exporting Supabase full-text articles -> {raw_parquet}")
    export_fulltext(str(raw_parquet), min_words=0, batch_size=args.batch_size, add_raw_dump=False)

    embeddings_parquet = out_dir / 'articles_with_embeddings.parquet'
    cache_path = Path('artifacts/embeddings_cache')

    helper_args = [
        sys.executable,
        'refresh_and_run_pipeline.py',
        '--parquet', str(raw_parquet),
        '--output-parquet', str(embeddings_parquet),
        '--cache', str(cache_path),
        '--min-words', str(args.min_words),
        '--batch-size', str(args.embedding_batch),
        '--embedding-workers', str(args.embedding_workers),
        '--skip-convert',
    ]
    if args.enable_subject_filter:
        helper_args.extend(['--exclude-keywords', args.exclude_keywords])
    else:
        helper_args.append('--no-filter')
    if args.reuse_cache:
        helper_args.append('--skip-embed')
    helper_args.append('--skip-pipeline')

    print("\n=== Step 2: Filtering and embedding")
    run_command(helper_args)

    if not args.skip_pipeline:
        print("\n=== Step 3: Running optimized pipeline")
        run_command([sys.executable, 'run_optimized_pipeline.py'])
    else:
        print("Skipping optimized pipeline as requested")

    print("\nPipeline complete. Outputs:")
    print(f"  Raw articles: {raw_parquet}")
    print(f"  Articles with embeddings: {embeddings_parquet}")


if __name__ == '__main__':
    main()
