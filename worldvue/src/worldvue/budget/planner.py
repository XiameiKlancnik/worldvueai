import pandas as pd
from pathlib import Path
from typing import Optional, Dict, Any
from .config import BudgetConfig


def estimate_cost(pairs: int, votes: int, tokens_per_call: int = 1000,
                  price_per_m: float = 0.15) -> tuple[int, int, float]:
    """
    Estimate cost for LLM judging.

    Returns:
        (total_calls, total_tokens, cost_usd)
    """
    calls = pairs * votes
    toks = calls * tokens_per_call
    usd = (toks / 1_000_000.0) * price_per_m
    return calls, toks, usd


class BudgetPlanner:
    """Plans and estimates costs for the entire pipeline."""

    def __init__(self, config: BudgetConfig):
        self.config = config

    def estimate_articles(self, articles_df: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
        """Estimate article sampling based on budget constraints."""
        result = {
            'max_articles_global': self.config.max_articles_global,
            'max_articles_per_country': self.config.max_articles_per_country,
        }

        if articles_df is not None:
            # Count articles per country
            country_counts = articles_df.groupby('country').size()

            # Apply country whitelist if specified
            if self.config.country_whitelist:
                country_counts = country_counts[country_counts.index.isin(self.config.country_whitelist)]

            # Apply per-country cap
            country_counts = country_counts.clip(upper=self.config.max_articles_per_country)

            # Apply global cap
            total_articles = min(country_counts.sum(), self.config.max_articles_global)

            result.update({
                'countries': len(country_counts),
                'total_articles_available': len(articles_df),
                'total_articles_used': int(total_articles),
                'articles_per_country': country_counts.to_dict()
            })

        return result

    def estimate_pairs(self) -> Dict[str, Any]:
        """Estimate pair sampling based on budget constraints."""
        within_pairs = int(self.config.target_pairs_total * self.config.within_country_ratio)
        cross_pairs = self.config.target_pairs_total - within_pairs

        return {
            'target_pairs_total': self.config.target_pairs_total,
            'within_country_pairs': within_pairs,
            'cross_country_pairs': cross_pairs,
            'max_pairs_per_article': self.config.max_pairs_per_article,
            'pairs_per_cluster_min': self.config.pairs_per_cluster_min,
            'pairs_per_cluster_max': self.config.pairs_per_cluster_max,
        }

    def estimate_judging_cost(self) -> Dict[str, Any]:
        """Estimate LLM judging costs."""
        calls, tokens, usd = estimate_cost(
            self.config.target_pairs_total,
            self.config.votes_per_pair,
            self.config.tokens_per_call_estimate,
            self.config.price_per_mtoken_usd
        )

        return {
            'pairs': self.config.target_pairs_total,
            'votes_per_pair': self.config.votes_per_pair,
            'total_calls': calls,
            'estimated_tokens': tokens,
            'estimated_cost_usd': usd,
            'price_per_mtoken_usd': self.config.price_per_mtoken_usd,
            'dry_run': self.config.dry_run
        }

    def generate_report(self, articles_df: Optional[pd.DataFrame] = None) -> str:
        """Generate a comprehensive budget report."""
        article_est = self.estimate_articles(articles_df)
        pair_est = self.estimate_pairs()
        cost_est = self.estimate_judging_cost()

        report = []
        report.append("# Budget Planning Report\n")
        report.append(f"**Seed**: {self.config.seed}\n")

        # Articles section
        report.append("## Article Sampling\n")
        report.append(f"- **Global cap**: {article_est['max_articles_global']:,}")
        report.append(f"- **Per-country cap**: {article_est['max_articles_per_country']}")
        if 'total_articles_used' in article_est:
            report.append(f"- **Total available**: {article_est['total_articles_available']:,}")
            report.append(f"- **Total to use**: {article_est['total_articles_used']:,}")
            report.append(f"- **Countries**: {article_est['countries']}")
        report.append(f"- **Min cluster size**: {self.config.cluster_min_size}\n")

        # Pairs section
        report.append("## Pair Sampling\n")
        report.append(f"- **Target total pairs**: {pair_est['target_pairs_total']:,}")
        report.append(f"- **Within-country pairs**: {pair_est['within_country_pairs']:,} ({self.config.within_country_ratio:.0%})")
        report.append(f"- **Cross-country pairs**: {pair_est['cross_country_pairs']:,} ({self.config.cross_country_ratio:.0%})")
        report.append(f"- **Max pairs per article**: {pair_est['max_pairs_per_article']}")
        report.append(f"- **Pairs per cluster**: {pair_est['pairs_per_cluster_min']}-{pair_est['pairs_per_cluster_max']}\n")

        # LLM Judging section
        report.append("## LLM Judging Costs\n")
        report.append(f"- **Pairs to judge**: {cost_est['pairs']:,}")
        report.append(f"- **Votes per pair**: {cost_est['votes_per_pair']}")
        report.append(f"- **Total LLM calls**: {cost_est['total_calls']:,}")
        report.append(f"- **Estimated tokens**: {cost_est['estimated_tokens']:,}")
        report.append(f"- **Price per M tokens**: ${cost_est['price_per_mtoken_usd']:.2f}")
        report.append(f"- **Estimated cost**: ${cost_est['estimated_cost_usd']:.2f}")

        if cost_est['dry_run']:
            report.append(f"- **MODE**: DRY RUN (no actual LLM calls)\n")
        else:
            report.append(f"- **MODE**: LIVE (will make actual LLM calls)\n")

        # Configuration section
        report.append("## Other Configuration\n")
        report.append(f"- **Pivot language**: {self.config.pivot_language}")
        report.append(f"- **Use translation**: {self.config.use_translation}")
        report.append(f"- **Entity masking**: {self.config.entity_masking}")
        report.append(f"- **Truncate chars**: {self.config.truncate_chars_per_side}")
        report.append(f"- **Scoring method**: {self.config.scoring_method}")
        report.append(f"- **Anchors per axis**: {self.config.anchors_per_axis}")
        report.append(f"- **Country calibration**: {self.config.country_isotonic_calibration}\n")

        return "\n".join(report)