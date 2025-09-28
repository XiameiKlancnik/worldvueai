"""Label aggregation for pairwise judgments (no ensemble methods)."""

import pandas as pd
from pathlib import Path
from typing import List, Dict, Optional
import json
import click


class PairLabeler:
    """
    Processes and aggregates LLM judge results into training labels.
    No weak supervision or ensemble methods - just direct aggregation.
    """

    def __init__(self, min_confidence: float = 0.55):
        self.min_confidence = min_confidence

    def process_judge_results(self, judge_path: Path) -> pd.DataFrame:
        """
        Process judge results from JSONL file into labeled pairs.

        Args:
            judge_path: Path to judge results JSONL file

        Returns:
            DataFrame with labeled pairs
        """
        records = []

        with open(judge_path, 'r') as f:
            for line in f:
                try:
                    result = json.loads(line)

                    # Skip if parse error or topic mismatch
                    flags = result.get('flags', {})
                    if flags.get('parse_error') or flags.get('topic_mismatch'):
                        continue

                    # Skip if low confidence
                    if result['confidence'] < self.min_confidence:
                        continue

                    # Convert to training label
                    # For binary classification, map Tie to 0.5 or skip
                    if result['winner'] == 'Tie':
                        y = 0.5  # Soft label for ties
                    elif result['winner'] == 'A':
                        y = 1.0
                    else:  # B wins
                        y = 0.0

                    records.append({
                        'pair_id': result['pair_id'],
                        'axis': result['axis'],
                        'winner': result['winner'],
                        'y': y,
                        'confidence': result['confidence'],
                        'evidence_a': result['evidence_a'],
                        'evidence_b': result['evidence_b'],
                        'flags': json.dumps(flags)
                    })

                except json.JSONDecodeError as e:
                    click.echo(f"Skipping malformed line: {e}", err=True)
                    continue

        df = pd.DataFrame(records)

        # Add pair metadata if available
        df = self._add_pair_metadata(df)

        click.echo(f"Processed {len(df)} labeled pairs from {len(records)} judge results")

        return df

    def _add_pair_metadata(self, labels_df: pd.DataFrame,
                           pairs_path: Optional[Path] = None) -> pd.DataFrame:
        """Add metadata from original pairs if available."""
        if pairs_path and pairs_path.exists():
            pairs_df = pd.read_parquet(pairs_path)

            duplicate_mask = pairs_df['pair_id'].duplicated(keep=False)
            if duplicate_mask.any():
                duplicate_ids = pairs_df.loc[duplicate_mask, 'pair_id'].unique()
                sample_ids = ', '.join(map(str, duplicate_ids[:5]))
                raise click.ClickException(
                    f"Duplicate pair_id values detected in {pairs_path}: {sample_ids}"
                    + ("..." if len(duplicate_ids) > 5 else '')
                )

            # Merge on pair_id
            labels_df = labels_df.merge(
                pairs_df[['pair_id', 'a_id', 'b_id', 'a_country', 'b_country',
                         'cluster_id', 'is_cross_country']],
                on='pair_id',
                how='left'
            )

        return labels_df

    def filter_for_training(self, labels_df: pd.DataFrame,
                           min_examples_per_axis: int = 100) -> pd.DataFrame:
        """
        Filter labeled pairs for training.

        Args:
            labels_df: DataFrame with all labeled pairs
            min_examples_per_axis: Minimum examples needed per axis

        Returns:
            Filtered DataFrame ready for training
        """
        filtered = []

        for axis in labels_df['axis'].unique():
            axis_df = labels_df[labels_df['axis'] == axis]

            if len(axis_df) < min_examples_per_axis:
                click.echo(f"Warning: Only {len(axis_df)} examples for {axis} "
                          f"(min: {min_examples_per_axis})", err=True)

            # Balance classes if needed
            balanced = self._balance_classes(axis_df)
            filtered.append(balanced)

        result = pd.concat(filtered, ignore_index=True)
        click.echo(f"Filtered to {len(result)} training pairs across {result['axis'].nunique()} axes")

        return result

    def _balance_classes(self, df: pd.DataFrame, max_imbalance: float = 1.5) -> pd.DataFrame:
        """
        Balance classes for training. Excludes ties and balances A vs B wins.

        Args:
            df: DataFrame for a single axis
            max_imbalance: Maximum ratio between A and B wins

        Returns:
            Balanced DataFrame with only A/B winners
        """
        # Filter out ties completely
        df_no_ties = df[df['winner'].isin(['A', 'B'])].copy()

        if len(df_no_ties) == 0:
            click.echo(f"Warning: No A/B decisions found for axis, only ties", err=True)
            return df_no_ties

        # Count wins for each side
        a_wins = len(df_no_ties[df_no_ties['winner'] == 'A'])
        b_wins = len(df_no_ties[df_no_ties['winner'] == 'B'])

        if a_wins == 0 or b_wins == 0:
            click.echo(f"Warning: Only one class present - A: {a_wins}, B: {b_wins}", err=True)
            return df_no_ties

        # Check if already balanced
        ratio = max(a_wins, b_wins) / min(a_wins, b_wins)
        if ratio <= max_imbalance:
            return df_no_ties

        # Balance by downsampling majority class
        target_count = min(a_wins, b_wins)
        target_count = max(target_count, 30)  # Minimum 30 examples per class

        balanced = []
        for winner in ['A', 'B']:
            winner_df = df_no_ties[df_no_ties['winner'] == winner]
            if len(winner_df) > target_count:
                winner_df = winner_df.sample(n=target_count, random_state=42)
            balanced.append(winner_df)

        result = pd.concat(balanced, ignore_index=True)
        click.echo(f"Balanced from A:{a_wins}, B:{b_wins} to A:{len(result[result['winner']=='A'])}, B:{len(result[result['winner']=='B'])}")

        return result

    def create_splits(self, labels_df: pd.DataFrame,
                     val_ratio: float = 0.15,
                     test_ratio: float = 0.15) -> Dict[str, pd.DataFrame]:
        """
        Create train/val/test splits.

        Args:
            labels_df: All labeled pairs
            val_ratio: Fraction for validation
            test_ratio: Fraction for test

        Returns:
            Dictionary with 'train', 'val', 'test' DataFrames
        """
        # Shuffle
        df = labels_df.sample(frac=1.0, random_state=42).reset_index(drop=True)

        n = len(df)
        n_test = int(n * test_ratio)
        n_val = int(n * val_ratio)
        n_train = n - n_test - n_val

        splits = {
            'test': df.iloc[:n_test],
            'val': df.iloc[n_test:n_test + n_val],
            'train': df.iloc[n_test + n_val:]
        }

        for name, split_df in splits.items():
            click.echo(f"{name}: {len(split_df)} pairs")

        return splits

    def save_labels(self, labels_df: pd.DataFrame, output_path: Path):
        """Save processed labels to parquet."""
        labels_df.to_parquet(output_path, index=False)
        click.echo(f"Saved {len(labels_df)} labeled pairs to {output_path}")

    def get_statistics(self, labels_df: pd.DataFrame) -> Dict:
        """Get statistics about the labeled data."""
        stats = {
            'total_pairs': len(labels_df),
            'axes': list(labels_df['axis'].unique()),
            'avg_confidence': labels_df['confidence'].mean(),
            'winner_distribution': labels_df['winner'].value_counts().to_dict(),
            'cross_country_pairs': labels_df['is_cross_country'].sum() if 'is_cross_country' in labels_df else 0
        }

        # Per-axis statistics
        for axis in labels_df['axis'].unique():
            axis_df = labels_df[labels_df['axis'] == axis]
            stats[f'{axis}_count'] = len(axis_df)
            stats[f'{axis}_avg_conf'] = axis_df['confidence'].mean()

        return stats