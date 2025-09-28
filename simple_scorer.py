#!/usr/bin/env python
"""
Simple scorer that uses judge results directly without training models.
This is much faster and still gives useful scores.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json

def score_from_judgments(judge_results_path, articles_path, output_path):
    """
    Score articles based on judge results without training models.
    Uses a simple voting approach.
    """

    # Load judge results
    judge_data = []
    with open(judge_results_path, 'r') as f:
        for line in f:
            judge_data.append(json.loads(line))

    judge_df = pd.DataFrame(judge_data)

    # Load articles to get IDs
    articles_df = pd.read_parquet(articles_path)

    # Calculate scores for each article based on wins/losses
    article_scores = {}

    for _, row in judge_df.iterrows():
        a_id = row.get('a_id')
        b_id = row.get('b_id')
        axis = row['axis']
        winner = row['winner']

        if a_id and b_id:
            # Initialize if needed
            if a_id not in article_scores:
                article_scores[a_id] = {ax: {'wins': 0, 'losses': 0, 'ties': 0} for ax in judge_df.axis.unique()}
            if b_id not in article_scores:
                article_scores[b_id] = {ax: {'wins': 0, 'losses': 0, 'ties': 0} for ax in judge_df.axis.unique()}

            # Update scores
            if winner == 'A':
                article_scores[a_id][axis]['wins'] += 1
                article_scores[b_id][axis]['losses'] += 1
            elif winner == 'B':
                article_scores[b_id][axis]['wins'] += 1
                article_scores[a_id][axis]['losses'] += 1
            else:  # Tie
                article_scores[a_id][axis]['ties'] += 1
                article_scores[b_id][axis]['ties'] += 1

    # Convert to scores (0-100 scale)
    results = []
    for article_id, axes_data in article_scores.items():
        row = {'article_id': article_id}

        for axis, counts in axes_data.items():
            total = counts['wins'] + counts['losses'] + counts['ties']
            if total > 0:
                # Simple scoring: wins=100, ties=50, losses=0, then average
                score = (counts['wins'] * 100 + counts['ties'] * 50) / total
            else:
                score = 50  # Default neutral score

            row[f'{axis}_score'] = score

        results.append(row)

    # Create DataFrame
    scores_df = pd.DataFrame(results)

    # Add article info (use 'id' instead of 'article_id')
    scores_df = scores_df.merge(
        articles_df[['id', 'title', 'source_name', 'country']].rename(columns={'id': 'article_id'}),
        on='article_id',
        how='left'
    )

    # Calculate overall score (average of all axes)
    score_cols = [c for c in scores_df.columns if c.endswith('_score')]
    scores_df['overall_score'] = scores_df[score_cols].mean(axis=1)

    # Save results
    scores_df.to_parquet(output_path, index=False)

    print(f"Scored {len(scores_df)} articles")
    print(f"Score columns: {score_cols}")
    print(f"\nTop 5 highest overall scores:")
    print(scores_df.nlargest(5, 'overall_score')[['title', 'source_name', 'overall_score']])
    print(f"\nBottom 5 lowest overall scores:")
    print(scores_df.nsmallest(5, 'overall_score')[['title', 'source_name', 'overall_score']])

    # Summary statistics
    print(f"\nScore statistics:")
    for col in score_cols:
        print(f"  {col}: mean={scores_df[col].mean():.1f}, std={scores_df[col].std():.1f}")

    return scores_df

if __name__ == "__main__":
    import sys

    # Default paths
    judge_path = "artifacts/judge_results.jsonl"
    articles_path = "articles_with_embeddings.parquet"
    output_path = "artifacts/simple_scores.parquet"

    # Override with command line args if provided
    if len(sys.argv) > 1:
        judge_path = sys.argv[1]
    if len(sys.argv) > 2:
        articles_path = sys.argv[2]
    if len(sys.argv) > 3:
        output_path = sys.argv[3]

    scores = score_from_judgments(judge_path, articles_path, output_path)
    print(f"\nScores saved to: {output_path}")