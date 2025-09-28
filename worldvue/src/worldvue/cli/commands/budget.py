"""Budget planning and management commands."""

import click
import pandas as pd
from pathlib import Path

from worldvue.budget import load_budget_config, BudgetPlanner


@click.group()
def budget():
    """Budget planning and cost estimation commands."""
    pass


@budget.command()
@click.option('--articles', type=click.Path(exists=True), help='Path to articles parquet file')
@click.option('--budget', type=click.Path(exists=True), required=True, help='Path to budget config YAML')
@click.option('--out', type=click.Path(), help='Output path for budget report (optional)')
def plan(articles, budget, out):
    """Plan budget and estimate costs for the pipeline."""
    # Load config
    config = load_budget_config(budget)
    planner = BudgetPlanner(config)

    # Load articles if provided
    articles_df = None
    if articles:
        articles_df = pd.read_parquet(articles)
        click.echo(f"Loaded {len(articles_df)} articles from {articles}")

    # Generate report
    report = planner.generate_report(articles_df)

    # Output report
    if out:
        with open(out, 'w') as f:
            f.write(report)
        click.echo(f"Budget report saved to {out}")
    else:
        click.echo(report)