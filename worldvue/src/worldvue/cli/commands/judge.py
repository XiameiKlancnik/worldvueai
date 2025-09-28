"""Style judge commands for pairwise article comparison."""

import click
import pandas as pd
from pathlib import Path

from worldvue.budget import load_budget_config, BudgetEnforcer
from worldvue.judge import StyleJudge, MockJudge


@click.group()
def judge():
    """LLM-based style judging commands."""
    pass


@judge.command()
@click.option('--pairs', type=click.Path(exists=True), required=True,
              help='Path to pairs parquet file')
@click.option('--articles', type=click.Path(exists=True), required=True,
              help='Path to articles parquet file with texts')
@click.option('--budget', type=click.Path(exists=True), required=True,
              help='Path to budget config YAML')
@click.option('--out', type=click.Path(), required=True,
              help='Output path for judge results JSONL file')
@click.option('--llm-client', help='LLM client configuration (optional)')
def style(pairs, articles, budget, out, llm_client):
    """Judge article pairs on style axes using LLM."""
    click.echo("DEBUG: Using UPDATED judge code with progress logging!")
    # Load config and data
    config = load_budget_config(budget)
    pairs_df = pd.read_parquet(pairs)
    articles_df = pd.read_parquet(articles)

    click.echo(f"Loaded {len(pairs_df)} pairs and {len(articles_df)} articles")

    # Check cost limit
    enforcer = BudgetEnforcer(config)
    estimated_cost = config.estimated_cost_usd

    if not enforcer.check_cost_limit(estimated_cost):
        click.echo("Aborting due to cost limit")
        return

    # Create judge (mock if dry_run)
    if config.dry_run:
        judge = MockJudge(config, llm_client)
        click.echo("Running in DRY RUN mode - no actual LLM calls")
    else:
        judge = StyleJudge(config, llm_client)
        click.echo(f"Running with estimated cost: ${estimated_cost:.2f}")

    # Process pairs
    all_results = []
    article_texts = articles_df.set_index('article_id')['text'].to_dict()

    # Get cluster summaries if available
    cluster_summaries = {}
    if 'cluster_summary' in pairs_df.columns:
        for _, row in pairs_df.iterrows():
            cluster_summaries[row['cluster_id']] = row.get('cluster_summary', '')

    processed = 0
    skipped = 0
    click.echo(f"Starting to process {len(pairs_df)} pairs...")

    for idx, pair_row in pairs_df.iterrows():
        # Get article texts
        try:
            text_a = article_texts[pair_row['a_id']]
            text_b = article_texts[pair_row['b_id']]
        except KeyError as e:
            skipped += 1
            click.echo(f"Missing article text for {e}, skipping pair {pair_row['pair_id']} (skipped: {skipped})", err=True)
            continue

        # Get cluster summary
        cluster_summary = cluster_summaries.get(pair_row['cluster_id'], '')

        # Judge this pair
        try:
            click.echo(f"Judging pair {processed + 1}: {pair_row['pair_id']} (row {idx})")
            results = judge.judge_pair(
                pair_row.to_dict(),
                text_a,
                text_b,
                cluster_summary
            )
            click.echo(f"Completed pair {processed + 1}: got {len(results)} judgments")
            all_results.extend(results)
            processed += 1

            # Show progress more frequently and save intermediate results
            if processed % 10 == 0 or processed == len(pairs_df):
                click.echo(f"Processed {processed}/{len(pairs_df)} pairs... ({processed/len(pairs_df)*100:.1f}%)")

                # Show stats about parse errors and decisions
                if all_results:
                    parse_errors = sum(1 for r in all_results if r.flags.get('parse_error', False))
                    total_results = len(all_results)
                    a_wins = sum(1 for r in all_results if r.winner == 'A')
                    b_wins = sum(1 for r in all_results if r.winner == 'B')
                    click.echo(f"  Parse errors: {parse_errors}/{total_results}, A wins: {a_wins}, B wins: {b_wins}")

                # Save intermediate results frequently for small datasets, less frequently for large ones
                save_frequency = min(20, max(5, len(pairs_df) // 5))  # Every 5-20 pairs depending on total
                if processed % save_frequency == 0 or processed == len(pairs_df):
                    temp_judge = judge.__class__(judge.config, judge.llm_client)
                    temp_judge.results = all_results
                    temp_out = Path(out).with_suffix(f'.temp_{processed}.jsonl')
                    temp_judge.save_results(temp_out)
                    click.echo(f"  Saved intermediate results to {temp_out}")

        except Exception as e:
            click.echo(f"Error judging pair {pair_row['pair_id']}: {e}", err=True)
            continue

    # Save results
    judge.results = all_results
    judge.save_results(Path(out))

    click.echo(f"\nJudging complete:")
    click.echo(f"  Processed: {processed}/{len(pairs_df)} pairs")
    click.echo(f"  Skipped: {skipped} pairs")
    click.echo(f"  Total judgments: {len(all_results)}")
    click.echo(f"  Results saved to: {out}")

    if not config.dry_run and hasattr(judge, 'actual_cost'):
        click.echo(f"  Actual cost: ${judge.actual_cost:.2f}")


__all__ = ['judge']
