import pandas as pd
from typing import Optional
import click
from .config import BudgetConfig


class BudgetEnforcer:
    """Enforces hard budget constraints throughout the pipeline."""

    def __init__(self, config: BudgetConfig):
        self.config = config

    def filter_articles(self, articles_df: pd.DataFrame) -> pd.DataFrame:
        """Filter articles according to budget constraints."""
        df = articles_df.copy()

        # Apply country whitelist
        if self.config.country_whitelist:
            df = df[df['country'].isin(self.config.country_whitelist)]

        # Apply per-country cap
        def sample_country(group):
            if len(group) > self.config.max_articles_per_country:
                return group.sample(n=self.config.max_articles_per_country,
                                    random_state=self.config.seed)
            return group

        df = df.groupby('country', group_keys=False).apply(sample_country)

        # Apply global cap
        if len(df) > self.config.max_articles_global:
            # Sample proportionally from each country
            df = df.sample(n=self.config.max_articles_global,
                           random_state=self.config.seed)

        click.echo(f"Filtered {len(articles_df)} articles to {len(df)} based on budget")
        return df

    def validate_pairs(self, pairs_df: pd.DataFrame) -> bool:
        """Validate that pairs respect budget constraints."""
        total_pairs = len(pairs_df)

        if total_pairs > self.config.target_pairs_total * 1.1:  # Allow 10% overage
            click.echo(f"WARNING: {total_pairs} pairs exceeds target {self.config.target_pairs_total} by >10%",
                       err=True)
            return False

        # Check max pairs per article
        if 'a_id' in pairs_df.columns:
            pairs_per_article = pd.concat([
                pairs_df['a_id'].value_counts(),
                pairs_df['b_id'].value_counts()
            ]).groupby(level=0).sum()

            max_actual = pairs_per_article.max()
            if max_actual > self.config.max_pairs_per_article * 1.5:
                click.echo(f"WARNING: Some articles have {max_actual} pairs, exceeds max {self.config.max_pairs_per_article}",
                           err=True)
                return False

        return True

    def check_cost_limit(self, estimated_cost: float, limit: Optional[float] = None) -> bool:
        """Check if estimated cost is within acceptable limits."""
        if self.config.dry_run:
            click.echo("DRY RUN mode - no actual costs will be incurred")
            return True

        if limit is None:
            limit = 100.0  # Default $100 limit for safety

        if estimated_cost > limit:
            click.echo(f"\n⚠️  COST WARNING: Estimated cost ${estimated_cost:.2f} exceeds limit ${limit:.2f}")
            if not click.confirm("Do you want to continue?"):
                return False

        return True

    def enforce_cluster_size(self, clusters_df: pd.DataFrame) -> pd.DataFrame:
        """Filter out clusters smaller than minimum size."""
        cluster_sizes = clusters_df['cluster_id'].value_counts()
        valid_clusters = cluster_sizes[cluster_sizes >= self.config.cluster_min_size].index
        filtered = clusters_df[clusters_df['cluster_id'].isin(valid_clusters)]

        click.echo(f"Filtered {len(clusters_df)} to {len(filtered)} articles in clusters >= {self.config.cluster_min_size}")
        return filtered