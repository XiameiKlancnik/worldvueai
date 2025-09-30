"""WorldVue CLI - LLM Judges + Transformers, Hybrid Pairing, Budget Controls."""

import json
from pathlib import Path

import click
from dotenv import load_dotenv

from worldvue.clustering.cluster import cluster_articles
from worldvue.clustering.dedup import remove_near_duplicates
from worldvue.data.loaders import load_articles
from worldvue.embeddings.encoder import EmbeddingEncoder
from worldvue.embeddings.store import DEFAULT_STORE_PATH
from worldvue.text.clean import basic_clean

from .commands.budget import budget
from .commands.clusters import clusters
from .commands.pairs import pairs
from .commands.judge import judge
from .commands.train import train
from .commands.score import score
from .commands.eval import eval
from .commands.run import run
from .commands.topic import topic


@click.group()
def cli():
    """WorldVue - LLM Judges + Transformers, Hybrid Pairing, Budget Controls."""
    load_dotenv()


# Add command groups
cli.add_command(budget)
cli.add_command(clusters)
cli.add_command(pairs)
cli.add_command(judge)
cli.add_command(train)
cli.add_command(score)
cli.add_command(eval)
cli.add_command(run)
cli.add_command(topic)


@cli.command()
@click.option('--input', type=click.Path(exists=True), required=True,
              help='Article dataset (CSV/JSON/Parquet)')
@click.option('--cache', type=click.Path(), default=str(DEFAULT_STORE_PATH),
              help='Embedding cache path')
@click.option('--refresh', is_flag=True, help='Recompute embeddings even if cached')
@click.option('--batch-size', type=int, default=32, help='Encoding batch size')
@click.option('--num-workers', type=int, default=0, help='Number of dataloader workers for encoding')
def embed(input, cache, refresh, batch_size, num_workers):
    """Encode articles with multilingual sentence transformer."""
    click.echo('Loading articles...')
    articles = load_articles(Path(input))
    for article in articles:
        article.text = basic_clean(article.text)
    encoder = EmbeddingEncoder(cache_path=str(cache), num_workers=num_workers)
    results = encoder.encode_articles(articles, batch_size=batch_size, refresh=refresh)
    click.echo(f'Encoded {len(results)} articles (refresh={refresh}). Cache -> {cache}')


@cli.command()
@click.option('--input', type=click.Path(exists=True), required=True,
              help='Article dataset path')
@click.option('--cache', type=click.Path(exists=True), default=str(DEFAULT_STORE_PATH),
              help='Embedding cache path')
@click.option('--min-size', type=int, default=3, help='Minimum cluster size')
@click.option('--deduplicate', is_flag=True, default=True,
              help='Remove near duplicates before clustering')
@click.option('--output', type=click.Path(), default='artifacts/clusters.json',
              help='Output JSON for cluster membership')
def cluster(input, cache, min_size, deduplicate, output):
    """Cluster articles by embedding proximity (legacy command)."""
    click.echo('Loading articles and embeddings...')
    articles = load_articles(Path(input))
    encoder = EmbeddingEncoder(cache_path=str(cache), num_workers=num_workers)
    encoder.encode_articles(articles, refresh=False)
    working = articles
    if deduplicate:
        working, dropped = remove_near_duplicates(articles)
        click.echo(f'Removed {len(dropped)} near-duplicate articles')
    clusters = cluster_articles(working, min_cluster_size=min_size)
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    Path(output).write_text(json.dumps(clusters, indent=2), encoding='utf-8')
    click.echo(f'Generated {len(clusters)} clusters → {output}')


if __name__ == '__main__':
    cli()


__all__ = ['cli']

