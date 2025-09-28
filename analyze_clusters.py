#!/usr/bin/env python3
"""Quick cluster analysis script."""

import pandas as pd

def main():
    # Load clusters
    df = pd.read_parquet('artifacts_run_20250927_193653_optimized_no_ties/clusters.parquet')

    print("=== CLUSTER CONTENT ANALYSIS - STEEKPROEVEN ===\n")

    # Get first 8 clusters for sampling
    clusters = df['cluster_id'].unique()[:8]

    for cid in clusters:
        cluster_data = df[df['cluster_id'] == cid]

        print(f"CLUSTER {cid}:")
        print(f"Keywords: {cluster_data.iloc[0]['cluster_keywords']}")
        print(f"Summary: {cluster_data.iloc[0]['cluster_summary']}")
        print(f"Articles: {len(cluster_data)}")
        print("Sample titles:")

        for title in cluster_data['title'].head(3):
            print(f"  - {title[:80]}...")

        print(f"Countries: {cluster_data['country'].value_counts().head(3).to_dict()}")
        print(f"Sources: {cluster_data['source_name'].value_counts().head(3).to_dict()}")
        print()

if __name__ == '__main__':
    main()