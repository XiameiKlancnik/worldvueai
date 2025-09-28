from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, List

import typer
from rich.console import Console

from worldvue.data.loaders import dump_pairs, load_articles, load_cached_pairs
from worldvue.data.types import Pair
from worldvue.embeddings.store import attach_embeddings, load_store
from worldvue.pairs.generator import generate_pairs
from worldvue.weak.labelers import DEFAULT_LABELERS
from worldvue.weak.model import EMLabelModel
from worldvue.weak.validator import weak_vs_llm_agreement

console = Console()
weak_app = typer.Typer(help='Weak supervision tooling')


def _load_clusters(path: Path) -> dict:
    if not path.exists():
        raise typer.BadParameter(f'Cluster assignments not found: {path}')
    return json.loads(path.read_text(encoding='utf-8'))


def _pairs_from_cache(path: Path) -> List[Pair]:
    payloads = load_cached_pairs(path)
    return [Pair(**payload) for payload in payloads]


def _attach_articles(pairs: Iterable[Pair], articles) -> None:
    index = {article.id: article for article in articles}
    for pair in pairs:
        if pair.article_a_id in index and pair.article_b_id in index:
            pair.attach_articles(index[pair.article_a_id], index[pair.article_b_id])


@weak_app.command('label')
def label_all(
    input: Path = typer.Option(..., exists=True, help='Article dataset (CSV/JSON/Parquet).'),
    clusters_path: Path = typer.Option(Path('artifacts/clusters.json'), help='Cluster assignment JSON.'),
    output: Path = typer.Option(Path('artifacts/pairs_weak.jsonl'), help='Destination for weak labels.'),
) -> None:
    """Apply all weak labeling functions across generated pairs."""
    console.log('Loading articles...')
    articles = load_articles(input)
    store = load_store()
    attach_embeddings(articles, store)
    clusters = _load_clusters(clusters_path)
    console.log(f'Generating pairs from {len(clusters)} clusters...')
    pairs = generate_pairs(articles, clusters)
    console.log(f'Running {len(DEFAULT_LABELERS)} labeling functions...')
    model = EMLabelModel(DEFAULT_LABELERS)
    model.fit(pairs)
    dump_pairs(output, [pair.dict(exclude={'article_a', 'article_b'}) for pair in pairs])
    console.log(f'Wrote weak labels for {len(pairs)} pairs → {output}')


@weak_app.command('stats')
def stats(
    pairs_path: Path = typer.Option(Path('artifacts/pairs_weak.jsonl'), exists=True, help='Weak label cache.'),
) -> None:
    """Display coverage and agreement metrics for weak labels."""
    pairs = _pairs_from_cache(pairs_path)
    metrics = weak_vs_llm_agreement(pairs)
    total_pairs = len(pairs)
    console.print(f'[bold]Weak label summary[/bold] ({total_pairs} pairs)')
    for axis, payload in metrics.items():
        console.print(f'- {axis}: agreement={payload["agreement_rate"]:.2%} confidence={payload["confidence"]:.2f}')


__all__ = ['weak_app']
