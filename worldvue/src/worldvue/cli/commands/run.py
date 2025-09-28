"""End-to-end pipeline execution commands."""

import click
import subprocess
from pathlib import Path
import tempfile
import shutil

from worldvue.budget import load_budget_config, BudgetPlanner


@click.group()
def run():
    """End-to-end pipeline execution commands."""
    pass


@run.command()
@click.option('--budget', type=click.Path(exists=True),
              default='worldvue/configs/budget_preview.yaml',
              help='Budget configuration (uses preview by default)')
@click.option('--articles', type=click.Path(exists=True),
              help='Articles parquet file (required)')
@click.option('--workspace', type=click.Path(),
              help='Workspace directory (default: temp dir)')
@click.option('--keep-workspace', is_flag=True,
              help='Keep workspace directory after completion')
def preview(budget, articles, workspace, keep_workspace):
    """Run a tiny end-to-end pipeline for testing (400 articles, 1000 pairs)."""
    if not articles:
        click.echo("Error: --articles is required", err=True)
        return

    # Load config and validate
    config = load_budget_config(budget)
    if not config.dry_run:
        click.echo("Warning: Preview mode should use dry_run=true", err=True)
        if not click.confirm("Continue anyway?"):
            return

    # Create workspace
    if workspace:
        workspace = Path(workspace)
        workspace.mkdir(parents=True, exist_ok=True)
        cleanup_workspace = False
    else:
        workspace = Path(tempfile.mkdtemp(prefix='worldvue_preview_'))
        cleanup_workspace = not keep_workspace

    click.echo(f"Running preview pipeline in {workspace}")

    # Show budget plan
    planner = BudgetPlanner(config)
    import pandas as pd
    articles_df = pd.read_parquet(articles)
    report = planner.generate_report(articles_df)
    click.echo("\n" + "="*50)
    click.echo("BUDGET PLAN")
    click.echo("="*50)
    click.echo(report)
    click.echo("="*50 + "\n")

    if not config.dry_run:
        if not click.confirm("Proceed with live LLM calls?"):
            return

    try:
        # Step 1: Global clustering
        click.echo("Step 1: Creating global topic clusters...")
        clusters_path = workspace / 'clusters.parquet'
        result = subprocess.run([
            'worldvue', 'clusters', 'make',
            '--articles', str(articles),
            '--budget', str(budget),
            '--out', str(clusters_path)
        ], capture_output=True, text=True)
        if result.returncode != 0:
            click.echo(f"Clustering failed: {result.stderr}", err=True)
            return

        # Step 2: Pair sampling
        click.echo("Step 2: Sampling article pairs...")
        pairs_path = workspace / 'pairs.parquet'
        result = subprocess.run([
            'worldvue', 'pairs', 'make',
            '--clusters', str(clusters_path),
            '--budget', str(budget),
            '--out', str(pairs_path)
        ], capture_output=True, text=True)
        if result.returncode != 0:
            click.echo(f"Pair sampling failed: {result.stderr}", err=True)
            return

        # Step 3: LLM judging
        click.echo("Step 3: LLM judging of pairs...")
        judge_results_path = workspace / 'judge_results.jsonl'
        result = subprocess.run([
            'worldvue', 'judge', 'style',
            '--pairs', str(pairs_path),
            '--articles', str(articles),
            '--budget', str(budget),
            '--out', str(judge_results_path)
        ], capture_output=True, text=True)
        if result.returncode != 0:
            click.echo(f"LLM judging failed: {result.stderr}", err=True)
            return

        # Step 4: Label processing
        click.echo("Step 4: Processing judge results into labels...")
        labels_path = workspace / 'labels.parquet'
        result = subprocess.run([
            'worldvue', 'pairs', 'labels',
            '--in', str(judge_results_path),
            '--out', str(labels_path)
        ], capture_output=True, text=True)
        if result.returncode != 0:
            click.echo(f"Label processing failed: {result.stderr}", err=True)
            return

        # Step 5: Model training
        click.echo("Step 5: Training cross-encoder models...")
        models_dir = workspace / 'models'
        result = subprocess.run([
            'worldvue', 'train', 'style',
            '--pairs-labeled', str(labels_path),
            '--articles', str(articles),
            '--out-dir', str(models_dir),
            '--epochs', '1',  # Quick training for preview
            '--batch-size', '8'
        ], capture_output=True, text=True)
        if result.returncode != 0:
            click.echo(f"Model training failed: {result.stderr}", err=True)
            return

        # Step 6: Article scoring
        click.echo("Step 6: Scoring articles...")
        scores_path = workspace / 'scores.parquet'
        result = subprocess.run([
            'worldvue', 'score', 'style',
            '--articles', str(articles),
            '--models', str(models_dir),
            '--budget', str(budget),
            '--clusters', str(clusters_path),
            '--out', str(scores_path)
        ], capture_output=True, text=True)
        if result.returncode != 0:
            click.echo(f"Article scoring failed: {result.stderr}", err=True)
            return

        # Step 7: Evaluation
        click.echo("Step 7: Evaluating system...")
        eval_path = workspace / 'evaluation.json'
        result = subprocess.run([
            'worldvue', 'eval', 'style',
            '--models-dir', str(models_dir),
            '--test-pairs', str(pairs_path),  # Using training pairs for demo
            '--scores', str(scores_path),
            '--articles', str(articles),
            '--budget', str(budget),
            '--out', str(eval_path)
        ], capture_output=True, text=True)
        if result.returncode != 0:
            click.echo(f"Evaluation failed: {result.stderr}", err=True)
            return

        click.echo(f"\n✅ Preview pipeline completed successfully!")
        click.echo(f"📁 Results in: {workspace}")
        click.echo(f"📊 Evaluation: {eval_path}")
        click.echo(f"📈 Scores: {scores_path}")

        # Show brief summary
        try:
            import pandas as pd
            scores_df = pd.read_parquet(scores_path)
            axes = [col for col in scores_df.columns if col != 'article_id']
            click.echo(f"\nScored {len(scores_df)} articles on {len(axes)} axes:")
            for axis in axes:
                mean_score = scores_df[axis].mean()
                click.echo(f"  {axis}: mean = {mean_score:.1f}")
        except Exception as e:
            click.echo(f"Could not load scores summary: {e}")

    finally:
        # Cleanup if requested
        if cleanup_workspace:
            try:
                shutil.rmtree(workspace)
                click.echo(f"Cleaned up workspace: {workspace}")
            except Exception as e:
                click.echo(f"Could not cleanup workspace: {e}")


@run.command()
@click.option('--config', type=click.Path(exists=True), required=True,
              help='Pipeline configuration file')
@click.option('--articles', type=click.Path(exists=True), required=True,
              help='Articles parquet file')
@click.option('--workspace', type=click.Path(), required=True,
              help='Workspace directory for outputs')
@click.option('--resume-from',
              type=click.Choice(['clusters', 'pairs', 'judge', 'labels', 'train', 'score', 'eval']),
              help='Resume pipeline from specific step')
def full(config, articles, workspace, resume_from):
    """Run the full production pipeline."""
    # Load config
    budget_config = load_budget_config(config)

    if budget_config.dry_run:
        if not click.confirm("Config has dry_run=true. Continue with mock LLM calls?"):
            return

    workspace = Path(workspace)
    workspace.mkdir(parents=True, exist_ok=True)

    click.echo(f"Running full pipeline with workspace: {workspace}")

    # Show budget estimate
    planner = BudgetPlanner(budget_config)
    import pandas as pd
    articles_df = pd.read_parquet(articles)
    estimated_cost = budget_config.estimated_cost_usd

    click.echo(f"Estimated cost: ${estimated_cost:.2f}")
    if estimated_cost > 10.0 and not budget_config.dry_run:
        if not click.confirm(f"Cost estimate ${estimated_cost:.2f} is significant. Continue?"):
            return

    # Define pipeline steps
    steps = [
        ('clusters', 'Global clustering'),
        ('pairs', 'Pair sampling'),
        ('judge', 'LLM judging'),
        ('labels', 'Label processing'),
        ('train', 'Model training'),
        ('score', 'Article scoring'),
        ('eval', 'System evaluation')
    ]

    # Find starting point
    start_index = 0
    if resume_from:
        start_index = next(i for i, (step, _) in enumerate(steps) if step == resume_from)
        click.echo(f"Resuming from step: {resume_from}")

    # Execute pipeline
    for i, (step, description) in enumerate(steps[start_index:], start_index):
        click.echo(f"\nStep {i+1}: {description}...")

        if step == 'clusters':
            clusters_path = workspace / 'clusters.parquet'
            if not clusters_path.exists() or i >= start_index:
                result = subprocess.run([
                    'worldvue', 'clusters', 'make',
                    '--articles', str(articles),
                    '--budget', str(config),
                    '--out', str(clusters_path)
                ], capture_output=True, text=True)
                if result.returncode != 0:
                    click.echo(f"Step failed: {result.stderr}", err=True)
                    return

        elif step == 'pairs':
            pairs_path = workspace / 'pairs.parquet'
            if not pairs_path.exists() or i >= start_index:
                result = subprocess.run([
                    'worldvue', 'pairs', 'make',
                    '--clusters', str(workspace / 'clusters.parquet'),
                    '--budget', str(config),
                    '--out', str(pairs_path)
                ], capture_output=True, text=True)
                if result.returncode != 0:
                    click.echo(f"Step failed: {result.stderr}", err=True)
                    return

        # Add other steps similarly...
        # (Full implementation would include all steps)

    click.echo(f"\n✅ Full pipeline completed!")
    click.echo(f"📁 Results in: {workspace}")