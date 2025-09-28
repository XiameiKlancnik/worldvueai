#!/usr/bin/env python3
"""
Pipeline-integrated parallel judge for run_optimized_pipeline.py

This script is called automatically by the pipeline to run parallel judging.
"""

import os
import sys
from pathlib import Path
import pandas as pd

# Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv()

# Add worldvue to path
sys.path.insert(0, str(Path(__file__).parent / "worldvue" / "src"))

from worldvue.budget import load_budget_config
from worldvue.judge.parallel_style_judge import ParallelStyleJudge


def main():
    """Run parallel judging for the pipeline."""

    if len(sys.argv) != 2:
        print("Usage: run_parallel_judge_for_pipeline.py <run_dir>")
        sys.exit(1)

    run_dir = Path(sys.argv[1])

    print("="*60)
    print(" PARALLEL JUDGE - PIPELINE MODE")
    print("="*60)
    print(f" Run directory: {run_dir}")
    print(" Workers: 10 (auto-configured)")
    print(" Rate limit: 500 req/min")
    print("="*60)

    # Paths
    pairs_path = run_dir / "pairs.parquet"
    articles_path = Path("articles_with_embeddings.parquet")
    budget_path = Path("worldvue/configs/budget.yaml")
    output_path = run_dir / "judge_results_parallel.jsonl"

    # Check files exist
    if not pairs_path.exists():
        print(f"ERROR: Pairs file not found: {pairs_path}")
        sys.exit(1)
    if not articles_path.exists():
        print(f"ERROR: Articles file not found: {articles_path}")
        sys.exit(1)
    if not budget_path.exists():
        print(f"ERROR: Budget config not found: {budget_path}")
        sys.exit(1)

    # Check OpenAI API key
    if not os.getenv('OPENAI_API_KEY'):
        print("ERROR: OPENAI_API_KEY not found in environment!")
        print("Set it with: set OPENAI_API_KEY=your-key-here")
        sys.exit(1)

    # Load data
    print("\nLoading data...")
    config = load_budget_config(str(budget_path))
    pairs_df = pd.read_parquet(pairs_path)
    articles_df = pd.read_parquet(articles_path)

    print(f"  Pairs: {len(pairs_df)}")
    print(f"  Articles: {len(articles_df)}")
    print(f"  Total judgments: {len(pairs_df) * 5} (5 axes per pair)")

    # Estimate time
    estimated_time = (len(pairs_df) * 5) / 500 * 60  # 500 req/min
    parallel_time = estimated_time / 10  # 10 workers
    print(f"\nTime estimate: ~{parallel_time:.0f}s (vs {estimated_time:.0f}s sequential)")
    print(f"  Speedup: {estimated_time/parallel_time:.1f}x")

    # Create parallel judge
    print("\nInitializing parallel judge...")
    judge = ParallelStyleJudge(
        config,
        max_workers=10,
        requests_per_minute=500,
        max_retries=3
    )

    # Process all pairs in parallel
    print("\nStarting parallel processing...")
    results = judge.judge_pairs_batch(pairs_df, articles_df, output_dir=run_dir)

    # Save results
    judge.results = results
    judge.save_results(output_path)

    print(f"\nParallel judging complete!")
    print(f"Results saved to: {output_path}")

    # Check balance
    a_wins = sum(1 for r in results if r.winner == 'A')
    b_wins = sum(1 for r in results if r.winner == 'B')
    total = a_wins + b_wins
    if total > 0:
        print(f"Balance: A={a_wins} ({a_wins/total*100:.1f}%), B={b_wins} ({b_wins/total*100:.1f}%)")

    # Summary for pipeline
    print(f"\nSummary for pipeline:")
    print(f"  Total judgments: {len(results)}")
    print(f"  Parse errors: {judge.stats['parse_errors']}")
    print(f"  API failures: {judge.stats['failed_requests']}")
    print(f"  Success rate: {judge.stats['successful_requests']/judge.stats['total_requests']*100:.1f}%")


if __name__ == '__main__':
    main()