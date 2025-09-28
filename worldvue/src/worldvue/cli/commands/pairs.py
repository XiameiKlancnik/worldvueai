"""Pair sampling and labeling commands."""

import click
import pandas as pd
from pathlib import Path

from worldvue.budget import load_budget_config, BudgetEnforcer
from worldvue.pairs import HybridPairSampler, PairLabeler
from worldvue.clustering.global_cluster import GlobalTopicClusterer


@click.group()
def pairs():
    """Article pair sampling and labeling commands."""
    pass


@pairs.command()
@click.option('--clusters', type=click.Path(exists=True), required=True,
              help='Path to clusters parquet file')
@click.option('--budget', type=click.Path(exists=True), required=True,
              help='Path to budget config YAML')
@click.option('--out', type=click.Path(), required=True,
              help='Output path for pairs parquet file')
@click.option('--min-word-count', type=int, default=200, show_default=True,
              help='Minimum article word count to keep')
@click.option('--require-full-text/--allow-partial-text', default=True, show_default=True,
              help='Require full-text articles (drop summaries)')
def make(clusters, budget, out, min_word_count, require_full_text):
    """Create hybrid pairs (within-country and cross-country)."""
    # Load config and data
    config = load_budget_config(budget)
    clusters_df = pd.read_parquet(clusters)

    click.echo(f"Loaded {len(clusters_df)} clustered articles")

    # Quality filtering
    word_mask = clusters_df['word_count'].fillna(0) >= min_word_count
    if require_full_text:
        if 'has_full_text' in clusters_df.columns:
            full_text_mask = clusters_df['has_full_text'].fillna(False)
            quality_mask = word_mask & full_text_mask
        else:
            click.echo("Warning: has_full_text column missing; applying word-count filter only", err=True)
            quality_mask = word_mask
    else:
        quality_mask = word_mask

    filtered_df = clusters_df[quality_mask].copy()
    dropped = len(clusters_df) - len(filtered_df)
    if dropped:
        click.echo(f"Filtered out {dropped} articles below quality thresholds")
    if filtered_df.empty:
        raise click.ClickException("No articles remain after quality filtering; adjust thresholds.")

    click.echo(f"Articles remaining after filtering: {len(filtered_df)}")

    # Enforce budget constraints
    enforcer = BudgetEnforcer(config)
    clusters_df = enforcer.enforce_cluster_size(filtered_df)

    # Create sampler and sample pairs
    # Since we have a merged dataframe, split it back for the sampler
    articles_df = clusters_df[['article_id', 'title', 'text', 'country', 'source_name', 'language', 'url', 'published_at', 'embedding']].copy()
    # Rename columns to match expected names
    articles_df = articles_df.rename(columns={'source_name': 'outlet', 'published_at': 'published_date'})
    cluster_assignments = clusters_df[['article_id', 'cluster_id']].copy()

    sampler = HybridPairSampler(config)
    pairs_df = sampler.sample_pairs(articles_df, cluster_assignments)

    # Validate pairs respect budget
    if not enforcer.validate_pairs(pairs_df):
        click.echo("Warning: Pairs exceed budget constraints", err=True)

    # Save results
    pairs_df.to_parquet(out, index=False)

    # Print statistics
    stats = sampler.get_statistics()
    click.echo(f"\nPair sampling complete:")
    click.echo(f"  Total pairs: {stats['total_pairs']:,}")
    click.echo(f"  Within-country: {stats['within_country']:,}")
    click.echo(f"  Cross-country: {stats['cross_country']:,}")
    click.echo(f"  Validation pairs: {stats['validation_pairs']:,}")
    click.echo(f"  Unique articles: {stats['unique_articles']:,}")
    click.echo(f"  Average degree: {stats['avg_degree']:.1f}")
    click.echo(f"  Max degree: {stats['max_degree']}")
    click.echo(f"\nSaved to {out}")


@pairs.command()
@click.option('--in', 'judge_results', type=click.Path(exists=True), required=True,
              help='Path to judge results JSONL file')
@click.option('--out', type=click.Path(), required=True,
              help='Output path for labeled pairs parquet file')
@click.option('--pairs', type=click.Path(exists=True), required=False,
              help='Path to original pairs parquet file (to get a_id/b_id metadata)')
@click.option('--min-confidence', type=float, default=0.55,
              help='Minimum confidence threshold')
def labels(judge_results, out, pairs, min_confidence):
    """Process judge results into training labels."""
    labeler = PairLabeler(min_confidence=min_confidence)

    # Process judge results
    labels_df = labeler.process_judge_results(Path(judge_results))

    # Add pair metadata if pairs file provided
    if pairs:
        labels_df = labeler._add_pair_metadata(labels_df, Path(pairs))

    # Filter for training
    training_df = labeler.filter_for_training(labels_df)

    # Save results
    labeler.save_labels(training_df, Path(out))

    # Print statistics
    stats = labeler.get_statistics(training_df)
    click.echo(f"\nLabel processing complete:")
    for key, value in stats.items():
        if isinstance(value, (int, float)):
            click.echo(f"  {key}: {value}")
        elif isinstance(value, dict):
            click.echo(f"  {key}: {value}")

    click.echo(f"\nLabeled pairs saved to {out}")