"""Global clustering commands."""

import click
import pandas as pd
import numpy as np
from pathlib import Path

from worldvue.budget import load_budget_config, BudgetEnforcer
from worldvue.clustering.global_cluster import GlobalTopicClusterer


@click.group()
def clusters():
    """Global topic clustering commands."""
    pass


@clusters.command()
@click.option('--articles', type=click.Path(exists=True), required=True,
              help='Path to articles parquet file with embeddings')
@click.option('--budget', type=click.Path(exists=True), required=True,
              help='Path to budget config YAML')
@click.option('--out', type=click.Path(), required=True,
              help='Output path for clusters parquet file')
@click.option('--save-clusterer', type=click.Path(),
              help='Path to save clusterer model (optional)')
def make(articles, budget, out, save_clusterer):
    """Create global topic clusters for cross-country comparison."""
    # Load config and data
    config = load_budget_config(budget)
    articles_df = pd.read_parquet(articles)

    click.echo(f"Loaded {len(articles_df)} articles")

    # Enforce budget constraints on articles
    enforcer = BudgetEnforcer(config)
    articles_df = enforcer.filter_articles(articles_df)

    # Check for required columns
    required_cols = ['article_id', 'embedding', 'text']
    missing_cols = [col for col in required_cols if col not in articles_df.columns]
    if missing_cols:
        click.echo(f"Error: Missing required columns: {missing_cols}", err=True)
        return

    # Extract embeddings and texts
    embeddings = np.stack(articles_df['embedding'].values)
    texts = articles_df['text'].tolist()
    article_ids = articles_df['article_id'].tolist()

    # Create texts for clustering (title + first part of content)
    cluster_texts = []
    for _, row in articles_df.iterrows():
        title = row.get('title', '')
        content = row['text'][:800]  # First 800 chars
        cluster_text = f"{title}. {content}" if title else content
        cluster_texts.append(cluster_text)

    click.echo(f"Extracted {len(embeddings)} embeddings with dimension {embeddings.shape[1]}")

    # Create clusterer
    clusterer = GlobalTopicClusterer(
        min_cluster_size=config.cluster_min_size,
        max_clusters=100,
        use_hdbscan=True
    )

    # Fit clustering
    click.echo("Fitting global topic clustering...")
    clusterer.fit(embeddings, cluster_texts)

    # Convert to DataFrame
    clusters_df = clusterer.to_dataframe(article_ids)

    # Merge with original article data
    result_df = articles_df.merge(clusters_df, on='article_id', how='left')

    # Remove unclustered articles
    result_df = result_df.dropna(subset=['cluster_id'])

    click.echo(f"Clustered {len(result_df)} articles into {result_df['cluster_id'].nunique()} clusters")

    # Print cluster statistics
    cluster_sizes = result_df['cluster_id'].value_counts()
    click.echo(f"Cluster size statistics:")
    click.echo(f"  Mean: {cluster_sizes.mean():.1f}")
    click.echo(f"  Median: {cluster_sizes.median():.1f}")
    click.echo(f"  Min: {cluster_sizes.min()}")
    click.echo(f"  Max: {cluster_sizes.max()}")

    # Save results
    result_df.to_parquet(out, index=False)
    click.echo(f"Clusters saved to {out}")

    # Save clusterer if requested
    if save_clusterer:
        clusterer.save(Path(save_clusterer))
        click.echo(f"Clusterer model saved to {save_clusterer}")