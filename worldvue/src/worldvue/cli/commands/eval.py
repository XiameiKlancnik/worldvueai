"""Evaluation commands for style scoring system."""

import click
import pandas as pd
from pathlib import Path

from worldvue.budget import load_budget_config
from worldvue.eval.style_eval import StyleEvaluator


@click.group()
def eval():
    """Evaluation commands."""
    pass


@eval.command()
@click.option('--models-dir', type=click.Path(exists=True), required=True,
              help='Directory with trained cross-encoder models')
@click.option('--test-pairs', type=click.Path(exists=True), required=True,
              help='Test pairs with ground truth judgments')
@click.option('--scores', type=click.Path(exists=True), required=True,
              help='Article scores from scoring system')
@click.option('--articles', type=click.Path(exists=True), required=True,
              help='Article metadata with countries')
@click.option('--budget', type=click.Path(exists=True), required=True,
              help='Budget configuration')
@click.option('--out', type=click.Path(), required=True,
              help='Output path for evaluation results')
@click.option('--anchors', type=click.Path(),
              help='Anchor pack for consistency validation (optional)')
def style(models_dir, test_pairs, scores, articles, budget, out, anchors):
    """Comprehensive evaluation of style scoring system."""
    # Load config
    config = load_budget_config(budget)

    # Create evaluator
    evaluator = StyleEvaluator(config)

    click.echo("Running comprehensive style evaluation...")

    # Run evaluation
    results = evaluator.evaluate_system(
        models_dir=Path(models_dir),
        test_pairs_path=Path(test_pairs),
        scores_path=Path(scores),
        articles_path=Path(articles),
        anchor_pack_path=Path(anchors) if anchors else None
    )

    # Save results
    evaluator.save_results(Path(out))

    # Print summary
    summary = results.get('overall_summary', {})
    click.echo(f"\nEvaluation Summary:")
    click.echo(f"  Tests passed: {summary.get('tests_passed', 0)}/{summary.get('tests_total', 0)}")
    click.echo(f"  Success rate: {summary.get('overall_success_rate', 0):.1%}")

    # Print key metrics
    if 'pairwise_accuracy' in results and 'overall' in results['pairwise_accuracy']:
        pairwise = results['pairwise_accuracy']['overall']
        click.echo(f"  Mean pairwise accuracy: {pairwise.get('mean_accuracy', 0):.3f}")
        click.echo(f"  Mean ROC-AUC: {pairwise.get('mean_roc_auc', 0):.3f}")

    # Print critical failures if any
    failures = summary.get('critical_failures', [])
    if failures:
        click.echo(f"\nCritical Failures:")
        for failure in failures:
            click.echo(f"  - {failure}")

    click.echo(f"\nDetailed results saved to {out}")
    click.echo(f"Report saved to {Path(out).with_suffix('.md')}")


@eval.command()
@click.option('--scores', type=click.Path(exists=True), required=True,
              help='Article scores parquet file')
@click.option('--articles', type=click.Path(exists=True), required=True,
              help='Article metadata with countries')
@click.option('--out', type=click.Path(), required=True,
              help='Output path for country analysis report')
def countries(scores, articles, out):
    """Analyze country-level score distributions."""
    # Load data
    scores_df = pd.read_parquet(scores)
    articles_df = pd.read_parquet(articles)

    # Merge with country info
    df = scores_df.merge(articles_df[['article_id', 'country']], on='article_id')

    axes = [col for col in scores_df.columns if col != 'article_id']
    countries = df['country'].unique()

    # Generate report
    report = ["# Country Analysis Report\n\n"]

    report.append(f"**Total Articles**: {len(df)}\n")
    report.append(f"**Countries**: {len(countries)}\n")
    report.append(f"**Style Axes**: {len(axes)}\n\n")

    # Per-country statistics
    report.append("## Country Statistics\n\n")
    report.append("| Country | Articles | " + " | ".join(f"{axis} Mean" for axis in axes) + " |\n")
    report.append("| ------- | -------- | " + " | ".join("-----------" for _ in axes) + " |\n")

    for country in sorted(countries):
        country_df = df[df['country'] == country]
        if len(country_df) < 5:
            continue

        row = f"| {country} | {len(country_df)} |"
        for axis in axes:
            mean_score = country_df[axis].mean()
            row += f" {mean_score:.1f} |"
        report.append(row + "\n")

    # Cross-country variance
    report.append("\n## Cross-Country Variance\n\n")
    for axis in axes:
        country_means = [df[df['country'] == c][axis].mean()
                        for c in countries
                        if len(df[df['country'] == c]) >= 5]

        if country_means:
            variance = pd.Series(country_means).var()
            report.append(f"- **{axis}**: Cross-country variance = {variance:.2f}\n")

    # Target compliance
    report.append("\n## Target Compliance (Mean within 50±3)\n\n")
    for axis in axes:
        compliant_countries = 0
        total_countries = 0

        for country in countries:
            country_df = df[df['country'] == country]
            if len(country_df) >= 5:
                mean_score = country_df[axis].mean()
                if abs(mean_score - 50) <= 3:
                    compliant_countries += 1
                total_countries += 1

        compliance_rate = compliant_countries / total_countries if total_countries > 0 else 0
        report.append(f"- **{axis}**: {compliant_countries}/{total_countries} countries ({compliance_rate:.1%})\n")

    # Save report
    with open(out, 'w') as f:
        f.write(''.join(report))

    click.echo(f"Country analysis saved to {out}")
    click.echo(f"Analyzed {len(countries)} countries across {len(axes)} axes")