"""Country-level calibration for global score comparability."""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional
from sklearn.isotonic import IsotonicRegression
from pathlib import Path
import json
import click

from ..budget.config import BudgetConfig


class CountryCalibrator:
    """
    Applies country-level calibration to ensure comparable score distributions.
    """

    def __init__(self, config: BudgetConfig):
        self.config = config
        self.calibrators = {}  # country -> {axis -> IsotonicRegression}
        self.reference_quantiles = {}  # axis -> quantiles
        self.country_stats = {}

    def fit(self, scores_df: pd.DataFrame, articles_df: pd.DataFrame,
           target_mean: float = 50.0, target_std: float = 15.0) -> 'CountryCalibrator':
        """
        Fit country-level calibration functions.

        Args:
            scores_df: DataFrame with article scores per axis
            articles_df: DataFrame with article metadata
            target_mean: Target mean score across countries
            target_std: Target standard deviation

        Returns:
            Self for chaining
        """
        if not self.config.country_isotonic_calibration:
            click.echo("Country calibration disabled in config")
            return self

        # Merge scores with country information
        df = scores_df.merge(articles_df[['article_id', 'country']], on='article_id')

        # Get style axes (excluding article_id)
        axes = [col for col in scores_df.columns if col != 'article_id']

        # Compute global reference distribution for each axis
        for axis in axes:
            all_scores = df[axis].values
            self.reference_quantiles[axis] = np.percentile(all_scores, np.arange(0, 101, 5))

        # Fit calibration for each country and axis
        countries = df['country'].unique()
        self.calibrators = {country: {} for country in countries}

        for country in countries:
            country_df = df[df['country'] == country]

            if len(country_df) < 10:  # Skip countries with too few articles
                click.echo(f"Skipping calibration for {country}: only {len(country_df)} articles")
                continue

            click.echo(f"Fitting calibration for {country} ({len(country_df)} articles)")

            for axis in axes:
                scores = country_df[axis].values

                # Compute country quantiles
                country_quantiles = np.percentile(scores, np.arange(0, 101, 5))

                # Fit isotonic regression to map country quantiles to reference quantiles
                iso_reg = IsotonicRegression(out_of_bounds='clip')
                iso_reg.fit(country_quantiles, self.reference_quantiles[axis])

                self.calibrators[country][axis] = iso_reg

                # Store statistics
                if country not in self.country_stats:
                    self.country_stats[country] = {}

                self.country_stats[country][axis] = {
                    'original_mean': float(np.mean(scores)),
                    'original_std': float(np.std(scores)),
                    'n_articles': len(scores)
                }

        click.echo(f"Fitted calibration for {len(countries)} countries across {len(axes)} axes")
        return self

    def transform(self, scores_df: pd.DataFrame, articles_df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply country-level calibration to scores.

        Args:
            scores_df: DataFrame with raw article scores
            articles_df: DataFrame with article metadata

        Returns:
            DataFrame with calibrated scores
        """
        if not self.config.country_isotonic_calibration or not self.calibrators:
            return scores_df.copy()

        # Merge with country information
        df = scores_df.merge(articles_df[['article_id', 'country']], on='article_id')
        axes = [col for col in scores_df.columns if col != 'article_id']

        calibrated_df = df.copy()

        for _, row in df.iterrows():
            country = row['country']

            if country not in self.calibrators:
                continue  # No calibration available for this country

            for axis in axes:
                if axis in self.calibrators[country]:
                    original_score = row[axis]
                    calibrator = self.calibrators[country][axis]

                    # Apply calibration
                    calibrated_score = calibrator.predict([original_score])[0]
                    calibrated_df.loc[calibrated_df['article_id'] == row['article_id'], axis] = calibrated_score

        # Return only score columns
        return calibrated_df[['article_id'] + axes]

    def get_calibration_report(self, scores_df: pd.DataFrame,
                              articles_df: pd.DataFrame) -> Dict:
        """Generate calibration quality report."""
        if not self.calibrators:
            return {'error': 'No calibration fitted'}

        # Apply calibration
        calibrated_df = self.transform(scores_df, articles_df)

        # Merge with countries
        df_orig = scores_df.merge(articles_df[['article_id', 'country']], on='article_id')
        df_cal = calibrated_df.merge(articles_df[['article_id', 'country']], on='article_id')

        axes = [col for col in scores_df.columns if col != 'article_id']
        countries = df_orig['country'].unique()

        report = {
            'axes': axes,
            'countries': list(countries),
            'country_stats': {}
        }

        for country in countries:
            country_orig = df_orig[df_orig['country'] == country]
            country_cal = df_cal[df_cal['country'] == country]

            if len(country_orig) == 0:
                continue

            country_report = {
                'n_articles': len(country_orig),
                'axes': {}
            }

            for axis in axes:
                orig_scores = country_orig[axis].values
                cal_scores = country_cal[axis].values

                country_report['axes'][axis] = {
                    'original_mean': float(np.mean(orig_scores)),
                    'original_std': float(np.std(orig_scores)),
                    'calibrated_mean': float(np.mean(cal_scores)),
                    'calibrated_std': float(np.std(cal_scores)),
                    'drift_reduction': self._calculate_drift_reduction(orig_scores, cal_scores)
                }

            report['country_stats'][country] = country_report

        # Global statistics
        global_stats = {}
        for axis in axes:
            all_orig = df_orig[axis].values
            all_cal = df_cal[axis].values

            global_stats[axis] = {
                'original_mean': float(np.mean(all_orig)),
                'original_std': float(np.std(all_orig)),
                'calibrated_mean': float(np.mean(all_cal)),
                'calibrated_std': float(np.std(all_cal)),
                'cross_country_variance_reduction': self._calculate_cross_country_variance_reduction(
                    df_orig, df_cal, axis
                )
            }

        report['global_stats'] = global_stats

        return report

    def _calculate_drift_reduction(self, original: np.ndarray, calibrated: np.ndarray) -> float:
        """Calculate how much calibration reduces drift from target (50.0)."""
        target = 50.0
        orig_drift = abs(np.mean(original) - target)
        cal_drift = abs(np.mean(calibrated) - target)

        if orig_drift == 0:
            return 1.0

        return max(0, (orig_drift - cal_drift) / orig_drift)

    def _calculate_cross_country_variance_reduction(self, df_orig: pd.DataFrame,
                                                   df_cal: pd.DataFrame, axis: str) -> float:
        """Calculate reduction in cross-country mean variance."""
        # Country means before calibration
        orig_country_means = df_orig.groupby('country')[axis].mean()

        # Country means after calibration
        cal_country_means = df_cal.groupby('country')[axis].mean()

        orig_var = np.var(orig_country_means)
        cal_var = np.var(cal_country_means)

        if orig_var == 0:
            return 1.0

        return max(0, (orig_var - cal_var) / orig_var)

    def save(self, path: Path):
        """Save calibration functions and statistics."""
        # Convert IsotonicRegression to serializable format
        serializable_calibrators = {}
        for country, axis_calibrators in self.calibrators.items():
            serializable_calibrators[country] = {}
            for axis, calibrator in axis_calibrators.items():
                serializable_calibrators[country][axis] = {
                    'x_': calibrator.X_.tolist(),
                    'y_': calibrator.y_.tolist(),
                    'f_': calibrator.f_.tolist()
                }

        data = {
            'calibrators': serializable_calibrators,
            'reference_quantiles': {k: v.tolist() for k, v in self.reference_quantiles.items()},
            'country_stats': self.country_stats,
            'config': {
                'country_isotonic_calibration': self.config.country_isotonic_calibration
            }
        }

        with open(path, 'w') as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load(cls, path: Path, config: BudgetConfig) -> 'CountryCalibrator':
        """Load calibration functions from file."""
        calibrator = cls(config)

        with open(path, 'r') as f:
            data = json.load(f)

        # Reconstruct IsotonicRegression objects
        calibrator.calibrators = {}
        for country, axis_calibrators in data['calibrators'].items():
            calibrator.calibrators[country] = {}
            for axis, cal_data in axis_calibrators.items():
                iso_reg = IsotonicRegression(out_of_bounds='clip')
                iso_reg.X_ = np.array(cal_data['x_'])
                iso_reg.y_ = np.array(cal_data['y_'])
                iso_reg.f_ = np.array(cal_data['f_'])
                calibrator.calibrators[country][axis] = iso_reg

        calibrator.reference_quantiles = {
            k: np.array(v) for k, v in data['reference_quantiles'].items()
        }
        calibrator.country_stats = data['country_stats']

        return calibrator