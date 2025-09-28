#!/usr/bin/env python3
"""Cluster content analysis for steekproeven."""

import pandas as pd
import json

def main():
    # Load clusters
    df = pd.read_parquet('artifacts_run_20250927_193653_optimized_no_ties/clusters.parquet')

    print("=== CLUSTER STEEKPROEVEN (SAMPLES) ===\n")

    # Sample 8 different clusters
    clusters = df['cluster_id'].unique()[:8]

    for i, cid in enumerate(clusters):
        cluster_data = df[df['cluster_id'] == cid]

        try:
            print(f"CLUSTER {i+1}: {cid}")
            print(f"  Size: {len(cluster_data)} articles")

            # Safe string handling
            keywords = str(cluster_data.iloc[0]['cluster_keywords'])
            summary = str(cluster_data.iloc[0]['cluster_summary'])

            print(f"  Keywords: {keywords[:100]}...")
            print(f"  Summary: {summary[:150]}...")

            # Country distribution
            countries = cluster_data['country'].value_counts().head(3)
            print(f"  Top countries: {dict(countries)}")

            # Source distribution
            sources = cluster_data['source_name'].value_counts().head(2)
            print(f"  Top sources: {dict(sources)}")

            # Sample titles (first 2)
            print("  Sample titles:")
            for title in cluster_data['title'].head(2):
                clean_title = str(title).encode('ascii', 'ignore').decode('ascii')
                print(f"    - {clean_title[:80]}...")

            print()

        except Exception as e:
            print(f"  Error analyzing cluster {cid}: {e}")
            print()

if __name__ == '__main__':
    main()