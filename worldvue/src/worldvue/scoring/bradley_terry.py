"""Bradley-Terry model for article scoring with country offsets."""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional
from scipy.optimize import minimize
from scipy.sparse import csr_matrix
import click
from pathlib import Path
import json

from ..budget.config import BudgetConfig


class BradleyTerryScorer:
    """
    Bradley-Terry model with country offsets for robust cross-country scoring.
    """

    def __init__(self, config: BudgetConfig):
        self.config = config
        self.article_scores = {}
        self.country_offsets = {}
        self.convergence_info = {}

    def fit(self, pairwise_probs: pd.DataFrame, articles_df: pd.DataFrame) -> 'BradleyTerryScorer':
        """
        Fit Bradley-Terry model with country offsets.

        Args:
            pairwise_probs: DataFrame with columns [a_id, b_id, prob_a_wins]
            articles_df: DataFrame with article metadata including country

        Returns:
            Self for chaining
        """
        # Merge to get country information
        pairs_with_countries = pairwise_probs.merge(
            articles_df[['article_id', 'country']].rename(columns={'article_id': 'a_id', 'country': 'a_country'}),
            on='a_id'
        ).merge(
            articles_df[['article_id', 'country']].rename(columns={'article_id': 'b_id', 'country': 'b_country'}),
            on='b_id'
        )

        # Get unique articles and countries
        all_articles = list(set(pairs_with_countries['a_id']) | set(pairs_with_countries['b_id']))
        all_countries = list(set(pairs_with_countries['a_country']) | set(pairs_with_countries['b_country']))

        # Create mappings
        article_to_idx = {aid: i for i, aid in enumerate(all_articles)}
        country_to_idx = {c: i for i, c in enumerate(all_countries)}

        n_articles = len(all_articles)
        n_countries = len(all_countries)

        click.echo(f"Fitting BT model: {n_articles} articles, {n_countries} countries, {len(pairs_with_countries)} pairs")

        # Prepare data for optimization
        pairs_data = []
        for _, row in pairs_with_countries.iterrows():
            a_idx = article_to_idx[row['a_id']]
            b_idx = article_to_idx[row['b_id']]
            a_country_idx = country_to_idx[row['a_country']]
            b_country_idx = country_to_idx[row['b_country']]
            prob = row['prob_a_wins']

            pairs_data.append((a_idx, b_idx, a_country_idx, b_country_idx, prob))

        # Optimization
        result = self._optimize_bt_with_offsets(pairs_data, n_articles, n_countries)

        # Extract results
        article_scores = result[:n_articles]
        country_offsets = result[n_articles:n_articles + n_countries]

        # Rescale to 0-100
        article_scores = self._rescale_scores(article_scores)

        # Store results
        self.article_scores = {aid: score for aid, score in zip(all_articles, article_scores)}
        self.country_offsets = {country: offset for country, offset in zip(all_countries, country_offsets)}

        click.echo(f"BT fitting complete. Score range: {min(article_scores):.1f} - {max(article_scores):.1f}")

        return self

    def _optimize_bt_with_offsets(self, pairs_data: list, n_articles: int,
                                 n_countries: int) -> np.ndarray:
        """
        Optimize Bradley-Terry model with country offsets.

        Solves:
        min_{s,β} Σ (σ((s_A - s_B) + (β_c(A) - β_c(B))) - P(A>B))² + λ(||s||² + ||β||²)

        Subject to: Σ β_c = 0
        """
        n_params = n_articles + n_countries
        lambda_reg = 0.01

        def objective(params):
            article_scores = params[:n_articles]
            country_offsets = params[n_articles:]

            # Constraint: sum of country offsets = 0
            offset_constraint = np.sum(country_offsets) ** 2

            loss = 0
            for a_idx, b_idx, a_country_idx, b_country_idx, prob in pairs_data:
                # BT prediction with country offsets
                score_diff = (article_scores[a_idx] - article_scores[b_idx] +
                             country_offsets[a_country_idx] - country_offsets[b_country_idx])

                pred_prob = 1 / (1 + np.exp(-score_diff))
                loss += (pred_prob - prob) ** 2

            # Regularization
            reg = lambda_reg * (np.sum(article_scores ** 2) + np.sum(country_offsets ** 2))

            return loss + reg + 1000 * offset_constraint  # Heavy penalty for constraint violation

        # Initialize parameters
        x0 = np.random.normal(0, 0.1, n_params)

        # Optimize
        result = minimize(objective, x0, method='BFGS', options={'maxiter': 1000})

        if not result.success:
            click.echo(f"Warning: BT optimization did not converge: {result.message}", err=True)

        self.convergence_info = {
            'success': result.success,
            'iterations': result.nit,
            'final_loss': result.fun
        }

        return result.x

    def _rescale_scores(self, scores: np.ndarray) -> np.ndarray:
        """Rescale scores to 0-100 range."""
        scores = scores - np.mean(scores)  # Center
        scores = scores / np.std(scores) * 15 + 50  # Scale to roughly 20-80 range
        return np.clip(scores, 0, 100)

    def predict(self, article_ids: list) -> Dict[str, float]:
        """Get scores for specific articles."""
        return {aid: self.article_scores.get(aid, 50.0) for aid in article_ids}

    def get_all_scores(self) -> Dict[str, float]:
        """Get all article scores."""
        return self.article_scores.copy()

    def get_country_analysis(self) -> pd.DataFrame:
        """Analyze country-level biases."""
        data = []
        for country, offset in self.country_offsets.items():
            # Get articles from this country
            country_articles = [aid for aid, score in self.article_scores.items()
                              if aid in self.article_scores]  # This would need country lookup

            data.append({
                'country': country,
                'offset': offset,
                'n_articles': len(country_articles)
            })

        return pd.DataFrame(data)

    def save(self, path: Path):
        """Save BT model results."""
        data = {
            'article_scores': self.article_scores,
            'country_offsets': self.country_offsets,
            'convergence_info': self.convergence_info
        }

        with open(path, 'w') as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load(cls, path: Path, config: BudgetConfig) -> 'BradleyTerryScorer':
        """Load BT model results."""
        scorer = cls(config)

        with open(path, 'r') as f:
            data = json.load(f)

        scorer.article_scores = data['article_scores']
        scorer.country_offsets = data['country_offsets']
        scorer.convergence_info = data['convergence_info']

        return scorer

    def validate_consistency(self, test_pairs: pd.DataFrame) -> Dict:
        """
        Validate model consistency on held-out pairs.

        Args:
            test_pairs: Test pairs with true probabilities

        Returns:
            Validation metrics
        """
        predictions = []
        actuals = []

        for _, row in test_pairs.iterrows():
            a_score = self.article_scores.get(row['a_id'])
            b_score = self.article_scores.get(row['b_id'])

            if a_score is not None and b_score is not None:
                # Predict probability A wins
                score_diff = a_score - b_score
                pred_prob = 1 / (1 + np.exp(-score_diff / 10))  # Scale for reasonable probabilities

                predictions.append(pred_prob)
                actuals.append(row['prob_a_wins'])

        if not predictions:
            return {'error': 'No valid predictions'}

        # Calculate metrics
        mse = np.mean([(p - a) ** 2 for p, a in zip(predictions, actuals)])
        mae = np.mean([abs(p - a) for p, a in zip(predictions, actuals)])

        # Concordance (how often ranking is preserved)
        concordant = 0
        total = 0
        for i, (p, a) in enumerate(zip(predictions, actuals)):
            for j, (p2, a2) in enumerate(zip(predictions[i+1:], actuals[i+1:], start=i+1)):
                if (p > p2) == (a > a2):
                    concordant += 1
                total += 1

        concordance = concordant / total if total > 0 else 0

        return {
            'mse': mse,
            'mae': mae,
            'concordance': concordance,
            'n_predictions': len(predictions)
        }