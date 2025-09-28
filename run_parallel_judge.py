#!/usr/bin/env python3
"""
Run parallel judge with retries and rate limiting for massive speedup!

FEATURES:
- 10x faster with parallel workers
- Automatic retries on API errors
- Rate limiting to avoid hitting OpenAI limits
- Progress tracking and intermediate saves
- Balanced A/B randomization

Usage:
    python run_parallel_judge.py [workers]

Default: 10 workers (adjust based on your OpenAI rate limits)
"""

import os
import sys
from pathlib import Path
import pandas as pd
import click

# Add worldvue to path
sys.path.insert(0, str(Path(__file__).parent / "worldvue" / "src"))

from worldvue.budget import load_budget_config
from worldvue.judge.parallel_style_judge import ParallelStyleJudge


def main():
    """Run parallel judging on the latest pairs."""

    # Configuration
    MAX_WORKERS = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    REQUESTS_PER_MINUTE = 500  # GPT-4o-mini tier limits

    print("="*60)
    print(" 🚀 PARALLEL JUDGE - TURBO MODE")
    print("="*60)
    print(f" Workers: {MAX_WORKERS}")
    print(f" Rate limit: {REQUESTS_PER_MINUTE} req/min")
    print("="*60)

    # Find the latest run directory
    run_dirs = sorted([d for d in Path('.').glob('artifacts_run_*') if d.is_dir()])
    if not run_dirs:
        print("❌ No artifact run directories found!")
        print("Run the pipeline first: python run_optimized_pipeline.py")
        return

    latest_dir = run_dirs[-1]
    print(f"\n📁 Using latest run: {latest_dir}")

    # Paths
    pairs_path = latest_dir / "pairs.parquet"
    articles_path = Path("articles_with_embeddings.parquet")
    budget_path = Path("worldvue/configs/budget.yaml")
    output_path = latest_dir / "judge_results_parallel.jsonl"

    # Check files exist
    if not pairs_path.exists():
        print(f"❌ Pairs file not found: {pairs_path}")
        return
    if not articles_path.exists():
        print(f"❌ Articles file not found: {articles_path}")
        return
    if not budget_path.exists():
        print(f"❌ Budget config not found: {budget_path}")
        return

    # Check OpenAI API key
    if not os.getenv('OPENAI_API_KEY'):
        print("❌ OPENAI_API_KEY not found in environment!")
        print("Set it with: set OPENAI_API_KEY=your-key-here")
        return

    # Load data
    print("\n📊 Loading data...")
    config = load_budget_config(str(budget_path))
    pairs_df = pd.read_parquet(pairs_path)
    articles_df = pd.read_parquet(articles_path)

    print(f"  Pairs: {len(pairs_df)}")
    print(f"  Articles: {len(articles_df)}")
    print(f"  Total judgments: {len(pairs_df) * 5} (5 axes per pair)")

    # Estimate time
    estimated_time = (len(pairs_df) * 5) / REQUESTS_PER_MINUTE * 60
    parallel_time = estimated_time / MAX_WORKERS
    print(f"\n⏱️ Time estimate:")
    print(f"  Sequential: ~{estimated_time:.0f}s")
    print(f"  Parallel ({MAX_WORKERS} workers): ~{parallel_time:.0f}s")
    print(f"  Speedup: {estimated_time/parallel_time:.1f}x")

    # Confirm
    response = input(f"\n🚀 Start parallel judging? (y/n): ")
    if response.lower() != 'y':
        print("Cancelled.")
        return

    # Create parallel judge
    print("\n🔧 Initializing parallel judge...")
    judge = ParallelStyleJudge(
        config,
        max_workers=MAX_WORKERS,
        requests_per_minute=REQUESTS_PER_MINUTE,
        max_retries=3
    )

    # Process all pairs in parallel
    print("\n🏃 Starting parallel processing...")
    results = judge.judge_pairs_batch(pairs_df, articles_df)

    # Save results
    judge.results = results
    judge.save_results(output_path)

    print(f"\n🎉 COMPLETE!")
    print(f"📁 Results saved to: {output_path}")
    print(f"\n📊 Final statistics:")
    print(f"  Total judgments: {len(results)}")

    # Check balance
    a_wins = sum(1 for r in results if r.winner == 'A')
    b_wins = sum(1 for r in results if r.winner == 'B')
    total = a_wins + b_wins
    if total > 0:
        print(f"  A wins: {a_wins} ({a_wins/total*100:.1f}%)")
        print(f"  B wins: {b_wins} ({b_wins/total*100:.1f}%)")

    print(f"\n✅ You can now continue with:")
    print(f"  python -m worldvue.cli.main pairs labels --in {output_path} --out {latest_dir}/pairs_labeled_parallel.parquet --pairs {pairs_path}")


if __name__ == '__main__':
    main()