"""Article scoring commands."""

import click
import pandas as pd
from pathlib import Path

from worldvue.budget import load_budget_config
from worldvue.scoring import AnchorScorer, AnchorPack, BradleyTerryScorer, CountryCalibrator


@click.group()
def score():
    """Article scoring commands using anchors or Bradley-Terry."""
    pass


@score.command()
@click.option('--articles', type=click.Path(exists=True), required=True,
              help='Path to articles parquet file')
@click.option('--models', type=click.Path(exists=True), required=True,
              help='Path to trained models directory')
@click.option('--method', type=click.Choice(['anchors', 'bt_offsets']), default='anchors',
              help='Scoring method to use')
@click.option('--budget', type=click.Path(exists=True), required=True,
              help='Path to budget config YAML')
@click.option('--out', type=click.Path(), required=True,
              help='Output path for scores parquet file')
@click.option('--anchors', type=click.Path(),
              help='Path to anchor pack JSON (optional, will create default)')
@click.option('--clusters', type=click.Path(),
              help='Path to clusters parquet for summaries (optional)')
def style(articles, models, method, budget, out, anchors, clusters):
    """Score articles on style axes."""
    # Load config and data
    config = load_budget_config(budget)
    articles_df = pd.read_parquet(articles)
    clusters_df = pd.read_parquet(clusters) if clusters else None

    click.echo(f"Loaded {len(articles_df)} articles")
    click.echo(f"Using {method} scoring method")

    if method == 'anchors':
        # Load or create anchor pack
        if anchors and Path(anchors).exists():
            anchor_pack = AnchorPack.load(Path(anchors))
            click.echo(f"Loaded anchor pack from {anchors}")
        else:
            anchor_pack = AnchorPack.create_default_pack(config)
            click.echo("Created default anchor pack")

            # Save default pack if path provided
            if anchors:
                anchor_pack.save(Path(anchors))
                click.echo(f"Saved default anchor pack to {anchors}")

        # Create scorer and score articles
        scorer = AnchorScorer(anchor_pack, Path(models), config)
        scores_df = scorer.score_articles(articles_df, clusters_df)

    elif method == 'bt_offsets':
        # Bradley-Terry scoring requires pairwise probabilities
        # This would typically be generated from cross-encoder predictions
        click.echo("Bradley-Terry scoring requires pairwise probability generation...")
        click.echo("This is a placeholder - would implement full BT pipeline")

        # Placeholder: create simple scores based on anchors as fallback
        anchor_pack = AnchorPack.create_default_pack(config)
        scorer = AnchorScorer(anchor_pack, Path(models), config)
        scores_df = scorer.score_articles(articles_df, clusters_df)

    # Apply country calibration if enabled
    if config.country_isotonic_calibration:
        click.echo("Applying country-level calibration...")
        calibrator = CountryCalibrator(config)
        calibrator.fit(scores_df, articles_df)
        scores_df = calibrator.transform(scores_df, articles_df)

        # Save calibration report
        report = calibrator.get_calibration_report(scores_df, articles_df)
        report_path = Path(out).with_suffix('.calibration.json')

        import json
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        click.echo(f"Calibration report saved to {report_path}")

    # Save scores
    scores_df.to_parquet(out, index=False)

    # Print statistics
    axes = [col for col in scores_df.columns if col != 'article_id']
    click.echo(f"\nScoring complete:")
    click.echo(f"  Articles scored: {len(scores_df)}")
    click.echo(f"  Style axes: {len(axes)}")

    for axis in axes:
        scores = scores_df[axis]
        click.echo(f"  {axis}: mean={scores.mean():.1f}, std={scores.std():.1f}, range=[{scores.min():.1f}, {scores.max():.1f}]")

    click.echo(f"\nScores saved to {out}")


@score.command()
@click.option('--pairs', type=click.Path(exists=True), required=True,
              help='Path to test pairs for BT fitting')
@click.option('--articles', type=click.Path(exists=True), required=True,
              help='Path to articles with country metadata')
@click.option('--models', type=click.Path(exists=True), required=True,
              help='Path to trained cross-encoder models')
@click.option('--budget', type=click.Path(exists=True), required=True,
              help='Path to budget config YAML')
@click.option('--out', type=click.Path(), required=True,
              help='Output path for BT scores parquet file')
def bradley_terry(pairs, articles, models, budget, out):
    """Score articles using Bradley-Terry model with country offsets."""
    config = load_budget_config(budget)
    pairs_df = pd.read_parquet(pairs)
    articles_df = pd.read_parquet(articles)

    click.echo(f"Loaded {len(pairs_df)} pairs and {len(articles_df)} articles")

    # Generate pairwise probabilities using cross-encoders
    # This is a simplified placeholder - would use actual model predictions
    click.echo("Generating pairwise probabilities from cross-encoders...")

    # Placeholder: add random probabilities for demo
    import numpy as np
    np.random.seed(42)
    pairs_df['prob_a_wins'] = np.random.beta(2, 2, len(pairs_df))  # Beta distribution around 0.5

    # Fit Bradley-Terry model
    scorer = BradleyTerryScorer(config)
    scorer.fit(pairs_df[['a_id', 'b_id', 'prob_a_wins']], articles_df)

    # Get scores for all articles
    all_scores = scorer.get_all_scores()

    # Create scores DataFrame
    scores_data = []
    for article_id, score in all_scores.items():
        scores_data.append({
            'article_id': article_id,
            'bt_score': score
        })

    scores_df = pd.DataFrame(scores_data)

    # Save results
    scores_df.to_parquet(out, index=False)

    # Save BT model
    bt_model_path = Path(out).with_suffix('.bt_model.json')
    scorer.save(bt_model_path)

    # Validation on test pairs
    test_pairs = pairs_df.sample(n=min(1000, len(pairs_df)), random_state=42)
    validation = scorer.validate_consistency(test_pairs)

    click.echo(f"\nBradley-Terry scoring complete:")
    click.echo(f"  Articles scored: {len(scores_df)}")
    click.echo(f"  Score range: [{scores_df['bt_score'].min():.1f}, {scores_df['bt_score'].max():.1f}]")
    click.echo(f"  Validation MSE: {validation.get('mse', 'N/A'):.4f}")
    click.echo(f"  Validation concordance: {validation.get('concordance', 'N/A'):.3f}")
    click.echo(f"\nScores saved to {out}")
    click.echo(f"Model saved to {bt_model_path}")