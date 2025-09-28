"""Anchor-based scoring for global article comparability."""

import math
from dataclasses import dataclass
from typing import Dict, List, Optional
import numpy as np
import pandas as pd
from pathlib import Path
import json
import torch
from transformers import AutoTokenizer

from ..training.cross_encoder import CrossEncoder
from ..budget.config import BudgetConfig


@dataclass
class AnchorText:
    """Reference text with fixed target score."""
    anchor_id: str
    text: str
    target_score: float
    language: str = "en"
    axis: str = ""


class AnchorPack:
    """Collection of anchor texts for scoring calibration."""

    def __init__(self, anchors: List[AnchorText]):
        self.anchors = anchors
        self.by_axis = self._group_by_axis()

    def _group_by_axis(self) -> Dict[str, List[AnchorText]]:
        """Group anchors by style axis."""
        groups = {}
        for anchor in self.anchors:
            axis = anchor.axis
            if axis not in groups:
                groups[axis] = []
            groups[axis].append(anchor)

        # Sort by target score
        for axis in groups:
            groups[axis].sort(key=lambda a: a.target_score)

        return groups

    def get_anchors(self, axis: str) -> List[AnchorText]:
        """Get anchors for a specific axis."""
        return self.by_axis.get(axis, [])

    @classmethod
    def create_default_pack(cls, config: BudgetConfig) -> 'AnchorPack':
        """Create default anchor pack for testing."""
        anchors = []
        axes = ['one_sidedness', 'hype', 'sourcing', 'fight_vs_fix', 'certain_vs_caution']

        # Create anchors at regular intervals
        n_anchors = config.anchors_per_axis
        target_scores = np.linspace(10, 95, n_anchors)

        for axis in axes:
            for i, score in enumerate(target_scores):
                anchor = AnchorText(
                    anchor_id=f"{axis}_anchor_{i}",
                    text=cls._generate_sample_text(axis, score),
                    target_score=score,
                    language="en",
                    axis=axis
                )
                anchors.append(anchor)

        return cls(anchors)

    @staticmethod
    def _generate_sample_text(axis: str, score: float) -> str:
        """Generate sample anchor text (for testing)."""
        # In production, these would be carefully curated reference texts
        templates = {
            'one_sidedness': {
                10: "This issue presents multiple viewpoints and acknowledges the complexity...",
                95: "There is only one correct position on this matter and opposing views are wrong..."
            },
            'hype': {
                10: "The situation developed gradually according to official reports...",
                95: "SHOCKING: This EXPLOSIVE development DEVASTATES everything we thought we knew..."
            },
            'sourcing': {
                10: "Claims are made without citations or verification...",
                95: "According to peer-reviewed research from Harvard University (Nature, 2023)..."
            },
            'fight_vs_fix': {
                10: "Stakeholders are working together to find collaborative solutions...",
                95: "This represents a fundamental battle between opposing forces..."
            },
            'certain_vs_caution': {
                10: "The evidence suggests this may be a contributing factor, though more research is needed...",
                95: "Scientists have definitively proven this causes the effect with no doubt remaining..."
            }
        }

        # Linear interpolation between low and high examples
        low_text = templates[axis][10]
        high_text = templates[axis][95]

        # Simple mixing based on score
        if score < 50:
            return low_text
        else:
            return high_text

    def save(self, path: Path):
        """Save anchor pack to JSON."""
        data = {
            'anchors': [
                {
                    'anchor_id': a.anchor_id,
                    'text': a.text,
                    'target_score': a.target_score,
                    'language': a.language,
                    'axis': a.axis
                }
                for a in self.anchors
            ]
        }
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load(cls, path: Path) -> 'AnchorPack':
        """Load anchor pack from JSON."""
        with open(path, 'r') as f:
            data = json.load(f)

        anchors = [
            AnchorText(**anchor_data)
            for anchor_data in data['anchors']
        ]
        return cls(anchors)


def logit(p: float, eps: float = 1e-5) -> float:
    """Convert probability to logit with clipping."""
    p = max(eps, min(1 - eps, p))
    return math.log(p / (1 - p))


class AnchorScorer:
    """Score articles using anchor-based calibration."""

    def __init__(self, anchor_pack: AnchorPack, models_dir: Path,
                config: BudgetConfig):
        self.anchor_pack = anchor_pack
        self.models_dir = models_dir
        self.config = config
        self.models = {}
        self.tokenizers = {}

        # Load models for each axis
        self._load_models()

    def _load_models(self):
        """Load trained cross-encoder models."""
        for axis in self.anchor_pack.by_axis.keys():
            model_path = self.models_dir / f'crossenc_{axis}'
            if model_path.exists():
                try:
                    model, tokenizer = CrossEncoder.load_model(model_path)
                    self.models[axis] = model
                    self.tokenizers[axis] = tokenizer
                except Exception as e:
                    print(f"Failed to load model for {axis}: {e}")

    def score_article(self, article_text: str, cluster_summary: str = "") -> Dict[str, float]:
        """
        Score a single article on all axes using anchor comparisons.

        Args:
            article_text: Text of the article to score
            cluster_summary: Topic cluster summary

        Returns:
            Dictionary mapping axis names to scores (0-100)
        """
        scores = {}

        for axis in self.anchor_pack.by_axis.keys():
            if axis not in self.models:
                continue

            anchor_scores = []
            anchors = self.anchor_pack.get_anchors(axis)

            # Compare article to each anchor
            for anchor in anchors:
                prob = self._compare_to_anchor(
                    article_text, anchor.text, cluster_summary, axis
                )

                # Convert to score using logit transformation
                anchor_score = anchor.target_score + 10 * logit(prob)
                anchor_scores.append(anchor_score)

            # Average anchor scores
            if anchor_scores:
                final_score = np.mean(anchor_scores)
                scores[axis] = np.clip(final_score, 0, 100)
            else:
                scores[axis] = 50.0  # Default middle score

        return scores

    def _compare_to_anchor(self, article_text: str, anchor_text: str,
                          cluster_summary: str, axis: str) -> float:
        """
        Compare article to anchor using cross-encoder.

        Returns probability that article scores higher than anchor.
        """
        model = self.models[axis]
        tokenizer = self.tokenizers[axis]

        # Format input like training data
        if cluster_summary:
            input_text = f"{cluster_summary} [SEP] {article_text} [SEP] {anchor_text}"
        else:
            input_text = f"{article_text} [SEP] {anchor_text}"

        # Tokenize
        encoding = tokenizer(
            input_text,
            truncation=True,
            padding=True,
            max_length=512,
            return_tensors='pt'
        )

        # Get prediction
        with torch.no_grad():
            prob = model(encoding['input_ids'], encoding['attention_mask'])
            return prob.item()

    def score_articles(self, articles_df: pd.DataFrame,
                      clusters_df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        Score multiple articles.

        Args:
            articles_df: DataFrame with article texts
            clusters_df: Optional cluster summaries

        Returns:
            DataFrame with scores for each article and axis
        """
        # Merge with clusters if provided
        if clusters_df is not None:
            df = articles_df.merge(
                clusters_df[['article_id', 'cluster_summary']],
                on='article_id',
                how='left'
            )
        else:
            df = articles_df.copy()
            df['cluster_summary'] = ""

        records = []

        for _, row in df.iterrows():
            article_id = row['article_id']
            text = row['text']
            summary = row.get('cluster_summary', "")

            # Score article
            scores = self.score_article(text, summary)

            # Create record
            record = {'article_id': article_id}
            record.update(scores)
            records.append(record)

        return pd.DataFrame(records)

    def validate_anchor_order(self, axis: str, language: str = "en") -> Dict:
        """
        Validate that anchor ladder is correctly ordered for a language.

        Returns validation statistics.
        """
        anchors = [a for a in self.anchor_pack.get_anchors(axis)
                  if a.language == language]

        if len(anchors) < 2:
            return {'valid': False, 'reason': 'Not enough anchors'}

        # Compare adjacent anchors
        violations = 0
        total_comparisons = 0

        for i in range(len(anchors) - 1):
            lower_anchor = anchors[i]
            higher_anchor = anchors[i + 1]

            # Compare using model
            prob = self._compare_to_anchor(
                higher_anchor.text, lower_anchor.text, "", axis
            )

            # Higher anchor should beat lower anchor
            if prob < 0.5:
                violations += 1

            total_comparisons += 1

        accuracy = (total_comparisons - violations) / total_comparisons

        return {
            'valid': accuracy >= 0.95,
            'accuracy': accuracy,
            'violations': violations,
            'total_comparisons': total_comparisons,
            'axis': axis,
            'language': language
        }