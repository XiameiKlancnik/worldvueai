"""Parallel LLM-based pairwise style judge with retries and rate limiting."""

import json
import re
from dataclasses import dataclass, asdict
from typing import List, Dict, Optional, Tuple
import random
from pathlib import Path
import pandas as pd
import click
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
from threading import Semaphore
import os

from ..budget.config import BudgetConfig
from .prompts import get_judge_prompt, get_multi_axis_judge_prompt, STYLE_AXES
from .style_judge import JudgeResult, format_article_for_prompt


class ParallelStyleJudge:
    """
    Parallel LLM-based judge with retry logic and rate limiting.
    Processes multiple pairs concurrently for massive speedup.
    """

    def __init__(self, config: BudgetConfig, max_workers: int = 10,
                 requests_per_minute: int = 500, max_retries: int = 3):
        """
        Initialize parallel judge.

        Args:
            config: Budget configuration
            max_workers: Number of parallel workers (default 10)
            requests_per_minute: Rate limit (default 500 for gpt-4o-mini)
            max_retries: Maximum retries on failure (default 3)
        """
        self.config = config
        self.max_workers = max_workers
        self.max_retries = max_retries
        self.results: List[JudgeResult] = []

        # Rate limiting
        self.requests_per_minute = requests_per_minute
        self.min_time_between_requests = 60.0 / requests_per_minute
        self.last_request_time = 0
        self.rate_limit_lock = Semaphore(1)

        # OpenAI client setup
        self._setup_client()

        # Statistics
        self.stats = {
            'total_requests': 0,
            'successful_requests': 0,
            'failed_requests': 0,
            'retries': 0,
            'parse_errors': 0,
            'api_errors': 0
        }

    def _setup_client(self):
        """Setup OpenAI client."""
        import openai
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            raise ValueError("OPENAI_API_KEY not found in environment variables")
        self.client = openai.OpenAI(api_key=api_key)

    def _wait_for_rate_limit(self):
        """Enforce rate limiting."""
        with self.rate_limit_lock:
            current_time = time.time()
            time_since_last = current_time - self.last_request_time
            if time_since_last < self.min_time_between_requests:
                time.sleep(self.min_time_between_requests - time_since_last)
            self.last_request_time = time.time()

    def _call_llm_with_retry(self, system_prompt: str, user_prompt: str) -> Optional[str]:
        """
        Call LLM with exponential backoff retry logic.

        Returns:
            Response text or None if all retries failed
        """
        model = "gpt-4o-mini" if getattr(self.config, 'use_cheaper_model', True) else "gpt-4"

        for attempt in range(self.max_retries):
            try:
                # Rate limiting
                self._wait_for_rate_limit()

                # Make API call
                self.stats['total_requests'] += 1
                response = self.client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    temperature=0.1,
                    max_tokens=400,
                    response_format={"type": "json_object"}
                )

                content = response.choices[0].message.content
                if content:
                    self.stats['successful_requests'] += 1
                    return content

            except Exception as e:
                self.stats['api_errors'] += 1
                self.stats['retries'] += 1

                # Check for rate limit error
                if "rate_limit" in str(e).lower():
                    wait_time = min(60, 2 ** attempt * 5)  # Exponential backoff, max 60s
                    click.echo(f"Rate limit hit, waiting {wait_time}s before retry {attempt+1}/{self.max_retries}", err=True)
                    time.sleep(wait_time)
                elif attempt < self.max_retries - 1:
                    wait_time = 2 ** attempt  # Exponential backoff
                    click.echo(f"API error: {str(e)[:100]}, retrying in {wait_time}s", err=True)
                    time.sleep(wait_time)
                else:
                    click.echo(f"API error after {self.max_retries} attempts: {str(e)[:100]}", err=True)
                    self.stats['failed_requests'] += 1
                    return None

        self.stats['failed_requests'] += 1
        return None

    def judge_pair_parallel(self, pair_data: Dict, article_a: Dict, article_b: Dict,
                           cluster_summary: str) -> List[JudgeResult]:
        """
        Judge a single pair on all style axes (called by parallel workers).

        This is the worker function that runs in parallel.
        """
        article_a_text = format_article_for_prompt(article_a.get('title'), article_a.get('text'))
        article_b_text = format_article_for_prompt(article_b.get('title'), article_b.get('text'))

        # Randomize A/B positions to eliminate position bias
        swapped = random.random() < 0.5
        if swapped:
            text_a, text_b = article_b_text, article_a_text
        else:
            text_a, text_b = article_a_text, article_b_text

        # Truncate texts
        text_a = text_a[:self.config.truncate_chars_per_side]
        text_b = text_b[:self.config.truncate_chars_per_side]

        # Check for translation needs (inline handling)
        translation_used = False
        if self.config.use_translation and self.config.pivot_language:
            if (pair_data.get('a_lang') != self.config.pivot_language or
                pair_data.get('b_lang') != self.config.pivot_language):
                translation_used = True

        results = []

        # Use multi-axis judgment for efficiency
        if hasattr(self.config, 'use_multi_axis_judging') and self.config.use_multi_axis_judging:
            # Get multi-axis prompt
            system_prompt, user_prompt = get_multi_axis_judge_prompt(
                text_a, text_b, cluster_summary
            )

            # Call LLM with retry
            response = self._call_llm_with_retry(system_prompt, user_prompt)

            if response:
                # Parse multi-axis response
                try:
                    data = json.loads(response)

                    for axis in STYLE_AXES:
                        if axis in data:
                            axis_data = data[axis]
                            winner = axis_data['winner']

                            # Unswap winner if needed
                            if swapped:
                                if winner == 'A':
                                    winner = 'B'
                                elif winner == 'B':
                                    winner = 'A'

                            results.append(JudgeResult(
                                pair_id=pair_data['pair_id'],
                                axis=axis,
                                winner=winner,
                                confidence=axis_data['confidence'],
                                evidence_a=axis_data['evidence_a'],
                                evidence_b=axis_data['evidence_b'],
                                flags={
                                    'translation_used': translation_used,
                                    'parallel_processed': True
                                },
                                raw_response=response
                            ))

                except json.JSONDecodeError as e:
                    self.stats['parse_errors'] += 1
                    # Create fallback results with random winners
                    for axis in STYLE_AXES:
                        winner = random.choice(['A', 'B'])
                        results.append(JudgeResult(
                            pair_id=pair_data['pair_id'],
                            axis=axis,
                            winner=winner,
                            confidence=0.6,
                            evidence_a="Parse error - parallel processing",
                            evidence_b=f"Parse error - assigned {winner}",
                            flags={
                                'parse_error': True,
                                'parallel_processed': True
                            },
                            raw_response=response
                        ))

        return results

    def judge_pairs_batch(self, pairs_df: pd.DataFrame, articles_df: pd.DataFrame,
                         progress_callback=None, output_dir: Path = None) -> List[JudgeResult]:
        """
        Judge multiple pairs in parallel with progress tracking.

        Args:
            pairs_df: DataFrame with pairs to judge
            articles_df: DataFrame with article texts
            progress_callback: Optional callback for progress updates

        Returns:
            List of all judge results
        """
        all_results = []
        columns = ['title', 'text'] if 'title' in articles_df.columns else ['text']
        article_records = articles_df.set_index('article_id')[columns].to_dict('index')

        # Get cluster summaries
        cluster_summaries = {}
        if 'cluster_summary' in pairs_df.columns:
            for _, row in pairs_df.iterrows():
                cluster_summaries[row['cluster_id']] = row.get('cluster_summary', '')

        click.echo(f"\nStarting parallel judging with {self.max_workers} workers")
        click.echo(f"Processing {len(pairs_df)} pairs x 5 axes = {len(pairs_df) * 5} judgments")
        click.echo(f"Rate limit: {self.requests_per_minute} requests/minute")

        start_time = time.time()
        processed = 0

        # Create tasks for thread pool
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all tasks
            future_to_pair = {}

            for _, pair_row in pairs_df.iterrows():
                # Get article texts
                try:
                    record_a = article_records[pair_row['a_id']]
                    record_b = article_records[pair_row['b_id']]
                except KeyError as e:
                    click.echo(f"Missing article record for {e}, skipping pair {pair_row['pair_id']}", err=True)
                    continue

                text_a = record_a.get('text')
                text_b = record_b.get('text')
                if not isinstance(text_a, str) or not isinstance(text_b, str):
                    click.echo(f"Missing article text content, skipping pair {pair_row['pair_id']}", err=True)
                    continue

                # Get cluster summary
                cluster_summary = cluster_summaries.get(pair_row['cluster_id'], '')

                # Submit task to thread pool
                future = executor.submit(
                    self.judge_pair_parallel,
                    pair_row.to_dict(),
                    record_a,
                    record_b,
                    cluster_summary
                )
                future_to_pair[future] = pair_row['pair_id']

            # Process completed tasks
            for future in as_completed(future_to_pair):
                pair_id = future_to_pair[future]
                processed += 1

                try:
                    results = future.result()
                    all_results.extend(results)

                    # Progress update
                    if processed % 10 == 0 or processed == len(pairs_df):
                        elapsed = time.time() - start_time
                        rate = processed / elapsed if elapsed > 0 else 0
                        eta = (len(pairs_df) - processed) / rate if rate > 0 else 0

                        click.echo(f"Progress: {processed}/{len(pairs_df)} pairs "
                                 f"({processed/len(pairs_df)*100:.1f}%) | "
                                 f"Rate: {rate:.1f} pairs/s | "
                                 f"ETA: {eta:.0f}s")

                        if progress_callback:
                            progress_callback(processed, len(pairs_df))

                        # Save intermediate results
                        if processed % 20 == 0 or processed == len(pairs_df):
                            self._save_intermediate_results(all_results, processed, output_dir)

                except Exception as e:
                    click.echo(f"Error processing pair {pair_id}: {e}", err=True)

        # Final statistics
        elapsed = time.time() - start_time
        click.echo(f"\nCompleted in {elapsed:.1f}s")
        click.echo(f"Statistics:")
        click.echo(f"  - Total requests: {self.stats['total_requests']}")
        click.echo(f"  - Successful: {self.stats['successful_requests']}")
        click.echo(f"  - Failed: {self.stats['failed_requests']}")
        click.echo(f"  - Retries: {self.stats['retries']}")
        click.echo(f"  - Parse errors: {self.stats['parse_errors']}")
        click.echo(f"  - Average rate: {processed/elapsed:.1f} pairs/s")

        # Check balance
        a_wins = sum(1 for r in all_results if r.winner == 'A')
        b_wins = sum(1 for r in all_results if r.winner == 'B')
        click.echo(f"\nBalance: A wins: {a_wins} ({a_wins/(a_wins+b_wins)*100:.1f}%), "
                  f"B wins: {b_wins} ({b_wins/(a_wins+b_wins)*100:.1f}%)")

        return all_results

    def _save_intermediate_results(self, results: List[JudgeResult], processed: int, output_dir: Path = None):
        """Save intermediate results to temp file."""
        if output_dir:
            temp_file = output_dir / f"judge_results_parallel_temp_{processed}.jsonl"
        else:
            temp_file = Path(f"judge_results_parallel_temp_{processed}.jsonl")
        with open(temp_file, 'w') as f:
            for result in results:
                f.write(json.dumps(asdict(result)) + '\n')
        click.echo(f"  Saved intermediate results to {temp_file}")

    def save_results(self, path: Path):
        """Save final judge results to JSONL file."""
        with open(path, 'w') as f:
            for result in self.results:
                f.write(json.dumps(asdict(result)) + '\n')
        click.echo(f"Final results saved to {path}")


def judge_parallel_command(pairs_path: str, articles_path: str, budget_path: str,
                          output_path: str, max_workers: int = 10):
    """
    Command-line interface for parallel judging.

    Example:
        python -m worldvue.judge.parallel_style_judge \
            --pairs pairs.parquet \
            --articles articles_with_embeddings.parquet \
            --budget budget.yaml \
            --out judge_results_parallel.jsonl \
            --workers 10
    """
    from ..budget import load_budget_config

    # Load config and data
    config = load_budget_config(budget_path)
    pairs_df = pd.read_parquet(pairs_path)
    articles_df = pd.read_parquet(articles_path)

    click.echo(f"Loaded {len(pairs_df)} pairs and {len(articles_df)} articles")

    # Create parallel judge
    judge = ParallelStyleJudge(config, max_workers=max_workers)

    # Process all pairs in parallel
    results = judge.judge_pairs_batch(pairs_df, articles_df)

    # Save results
    judge.results = results
    judge.save_results(Path(output_path))

    click.echo(f"\n🎉 Parallel judging complete!")
    click.echo(f"Results saved to: {output_path}")


if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1:
        # CLI mode
        judge_parallel_command(*sys.argv[1:])
    else:
        print("Usage: python parallel_style_judge.py <pairs> <articles> <budget> <output> [workers]")
