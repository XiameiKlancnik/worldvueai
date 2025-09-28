"""Cross-encoder training commands."""

import click
import pandas as pd
from pathlib import Path

from worldvue.budget import load_budget_config
from worldvue.training.cross_encoder import CrossEncoderTrainer


@click.group()
def train():
    """Model training commands."""
    pass


@train.command()
@click.option('--pairs-labeled', type=click.Path(exists=True), required=True,
              help='Path to labeled pairs parquet file')
@click.option('--articles', type=click.Path(exists=True), required=True,
              help='Path to articles parquet file with texts')
@click.option('--axes', multiple=True,
              help='Specific axes to train (default: all)')
@click.option('--out-dir', type=click.Path(), required=True,
              help='Output directory for trained models')
@click.option('--model-name', default='xlm-roberta-base',
              help='Base transformer model name')
@click.option('--epochs', type=int, default=3,
              help='Number of training epochs')
@click.option('--batch-size', type=int, default=16,
              help='Training batch size')
@click.option('--learning-rate', type=float, default=2e-5,
              help='Learning rate')
def style(pairs_labeled, articles, axes, out_dir, model_name, epochs, batch_size, learning_rate):
    """Train cross-encoder models for style axes."""
    # Load data
    labels_df = pd.read_parquet(pairs_labeled)
    articles_df = pd.read_parquet(articles)

    click.echo(f"Loaded {len(labels_df)} labeled pairs and {len(articles_df)} articles")

    # Check for required columns
    required_label_cols = ['a_id', 'b_id', 'axis', 'y']
    missing_cols = [col for col in required_label_cols if col not in labels_df.columns]
    if missing_cols:
        click.echo(f"Error: Missing required columns in labels: {missing_cols}", err=True)
        return

    if 'text' not in articles_df.columns:
        click.echo("Error: Missing 'text' column in articles", err=True)
        return

    # Create trainer
    trainer = CrossEncoderTrainer(
        model_name=model_name,
        learning_rate=learning_rate
    )

    # Train models
    output_dir = Path(out_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if axes:
        # Train specific axes
        axes_to_train = list(axes)
    else:
        # Train all axes
        axes_to_train = labels_df['axis'].unique().tolist()

    click.echo(f"Training models for axes: {axes_to_train}")

    results = trainer.train_all_axes(
        labels_df,
        articles_df,
        output_dir,
        axes=axes_to_train
    )

    # Print results summary
    click.echo(f"\nTraining Results:")
    for axis, result in results.items():
        if 'best_val_auc' in result:
            click.echo(f"  {axis}: AUC = {result['best_val_auc']:.3f}")
        else:
            click.echo(f"  {axis}: Training failed")

    click.echo(f"\nModels saved to {out_dir}")


__all__ = ['train']
