"""
Enhanced training pipeline for WorldVue with LLM judging
"""

import os
import sys
import json
import joblib
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add worldvue src to Python path
sys.path.insert(0, str(Path(__file__).parent / "worldvue" / "src"))

from worldvue.data.loaders import load_articles
from worldvue.embeddings.encoder import EmbeddingEncoder
from worldvue.embeddings.store import attach_embeddings, load_store, DEFAULT_STORE_PATH
from worldvue.text.clean import basic_clean
from worldvue.clustering.cluster import cluster_articles
from worldvue.clustering.dedup import remove_near_duplicates
from worldvue.pairs.generator import generate_pairs
from worldvue.weak.labelers import DEFAULT_LABELERS
from worldvue.weak.model import EMLabelModel
from worldvue.training.ranker import train_style_model, evaluate_models
from worldvue.judge.client import BudgetAwareJudge

def dump_pairs(file_path, pairs_data):
    """Simple function to dump pairs to JSONL"""
    with open(file_path, 'w', encoding='utf-8') as f:
        for pair in pairs_data:
            f.write(json.dumps(pair) + '\n')

def run_full_pipeline_with_llm(
    input_csv: str = "all_articles.csv",
    cache_path: str = "embeddings_cache.pkl",
    min_cluster_size: int = 3,
    artifacts_dir: str = "artifacts",
    llm_budget: float = 10.0,
    max_llm_pairs: int = 100
):
    """Run the complete WorldVue training pipeline with LLM judging"""

    # Create artifacts directory
    artifacts_path = Path(artifacts_dir)
    artifacts_path.mkdir(exist_ok=True)

    print("="*60)
    print(" WorldVue Training Pipeline with LLM Judging")
    print("="*60)

    # Step 1: Load articles
    print("Step 1: Loading articles...")
    articles = load_articles(Path(input_csv))
    print(f"Loaded {len(articles)} articles")

    # Clean text
    for article in articles:
        article.text = basic_clean(article.text)

    # Step 2: Generate embeddings
    print("Step 2: Generating embeddings...")
    encoder = EmbeddingEncoder(cache_path=cache_path)
    results = encoder.encode_articles(articles, batch_size=32, refresh=False)
    print(f"Encoded {len(results)} articles")

    # Step 3: Remove duplicates and cluster
    print("Step 3: Clustering articles...")
    working, dropped = remove_near_duplicates(articles)
    print(f"Removed {len(dropped)} near-duplicate articles")

    clusters = cluster_articles(working, min_cluster_size=min_cluster_size)
    clusters_file = artifacts_path / "clusters.json"
    with open(clusters_file, 'w') as f:
        json.dump(clusters, f, indent=2)
    print(f"Generated {len(clusters)} clusters")

    # Step 4: Generate pairs and weak labels
    print("Step 4: Generating pairs and weak labels...")
    pairs = generate_pairs(working, clusters)
    print(f"Generated {len(pairs)} pairs")

    # Apply weak labeling
    model = EMLabelModel(DEFAULT_LABELERS)
    model.fit(pairs)
    print(f"Applied {len(DEFAULT_LABELERS)} weak labeling functions")

    # Save pairs with weak labels
    pairs_file = artifacts_path / "pairs_weak.jsonl"
    dump_pairs(pairs_file, [pair.model_dump(exclude={'article_a', 'article_b'}) for pair in pairs])
    print(f"Saved weak labels to {pairs_file}")

    # Step 5: LLM Judging (NEW!)
    print("Step 5: Adding LLM judgments...")

    # Attach articles to pairs for LLM judging
    article_index = {article.id: article for article in working}

    # Select diverse pairs for LLM judging
    judge_pairs = []
    for pair in pairs[:max_llm_pairs]:  # Limit to prevent overspending
        if pair.article_a_id in article_index and pair.article_b_id in article_index:
            pair.attach_articles(article_index[pair.article_a_id], article_index[pair.article_b_id])
            judge_pairs.append(pair)

    if judge_pairs:
        judge = BudgetAwareJudge(budget_usd=llm_budget)
        judged_count = 0

        for i, pair in enumerate(judge_pairs):
            try:
                result = judge.judge_pair(pair)
                judged_count += 1

                # Progress update
                if (i + 1) % 10 == 0:
                    print(f"  Judged {judged_count}/{len(judge_pairs)} pairs (spent ${judge.spent:.2f})")

            except Exception as e:
                print(f"  Stopped LLM judging: {e}")
                break

        print(f"LLM judged {judged_count} pairs (${judge.spent:.2f} spent)")

        # Save pairs with LLM judgments
        judged_file = artifacts_path / "pairs_judged.jsonl"
        dump_pairs(judged_file, [pair.model_dump(exclude={'article_a', 'article_b'}) for pair in pairs])
        print(f"Saved LLM-enhanced labels to {judged_file}")

    # Step 6: Train final models
    print("Step 6: Training final models...")

    # Ensure all pairs have articles attached
    for pair in pairs:
        if not pair.article_a and pair.article_a_id in article_index:
            pair.attach_articles(article_index[pair.article_a_id], article_index[pair.article_b_id])

    # Train models with LLM-enhanced data
    models = train_style_model(pairs, prefer_lightgbm=False)

    if models:
        model_file = artifacts_path / "style_model_with_llm.joblib"
        joblib.dump(models, model_file)
        print(f"Saved trained models to {model_file}")

        # Evaluate
        metrics = evaluate_models(models, pairs)
        if metrics:
            print("\nTraining accuracy (with LLM enhancement):")
            for axis, value in metrics.items():
                print(f"  {axis}: {value:.2%}")

        print(f"\nTrained {len(models)} axis models:")
        for axis in models.keys():
            print(f"  - {axis}")

        # Show LLM vs weak label comparison
        llm_pairs = [p for p in pairs if p.llm_labels]
        if llm_pairs:
            print(f"\nEnhanced with {len(llm_pairs)} LLM judgments out of {len(pairs)} total pairs")
    else:
        print("Warning: No models trained (insufficient data)")

    print("="*60)
    print("Enhanced training pipeline complete!")
    print("="*60)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='WorldVue Training Pipeline with LLM')
    parser.add_argument('--input', default='all_articles.csv', help='Input CSV file')
    parser.add_argument('--cache', default='embeddings_cache.pkl', help='Embeddings cache path')
    parser.add_argument('--min-cluster-size', type=int, default=3, help='Minimum cluster size')
    parser.add_argument('--artifacts', default='artifacts', help='Artifacts directory')
    parser.add_argument('--llm-budget', type=float, default=10.0, help='LLM budget in USD')
    parser.add_argument('--max-llm-pairs', type=int, default=100, help='Max pairs to judge with LLM')

    args = parser.parse_args()

    run_full_pipeline_with_llm(
        input_csv=args.input,
        cache_path=args.cache,
        min_cluster_size=args.min_cluster_size,
        artifacts_dir=args.artifacts,
        llm_budget=args.llm_budget,
        max_llm_pairs=args.max_llm_pairs
    )