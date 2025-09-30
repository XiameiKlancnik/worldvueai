#!/usr/bin/env python3
"""Pipeline: LLM topic labeling + classifier training."""

import argparse
from pathlib import Path

import pandas as pd

from worldvue.topic_filter.parallel_topic_judge import ParallelTopicJudge
from worldvue.topic_filter.llm_dataset import load_llm_labels, build_multiclass_dataset
from dotenv import load_dotenv


def main():
    parser = argparse.ArgumentParser(description="LLM topic labeling and classifier training pipeline")
    parser.add_argument('--articles', default='articles_with_embeddings.parquet', help='Parquet with embeddings')
    parser.add_argument('--out-dir', default='artifacts/topic_pipeline', help='Output directory')
    parser.add_argument('--max-rows', type=int, default=5000, help='Number of articles to send to LLM (head)')
    parser.add_argument('--workers', type=int, default=8, help='Parallel workers for LLM')
    parser.add_argument('--rpm', type=int, default=300, help='Requests per minute for LLM')
    parser.add_argument('--min-conf', type=float, default=0.9, help='Minimum LLM confidence to keep label')
    parser.add_argument('--max-per-class', type=int, default=3000, help='Balance cap per class in dataset')
    parser.add_argument('--threshold', type=float, default=0.9, help='Probability threshold for political gate (if later used)')
    args = parser.parse_args()

    load_dotenv()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    articles_path = Path(args.articles)
    df = pd.read_parquet(articles_path)
    if args.max_rows:
        df = df.head(args.max_rows)
    print(f"Using {len(df)} articles for LLM labeling")

    # Step 1: LLM labeling
    labels_path = out_dir / 'topic_labels.jsonl'
    judge = ParallelTopicJudge(max_workers=args.workers, requests_per_minute=args.rpm)
    judge.judge_articles(df, output_dir=out_dir)
    print(f"LLM stats: {judge.stats}")

    # Step 2: Build balanced dataset
    labs = load_llm_labels(labels_path, min_conf=args.min_conf)
    if labs.empty:
        raise SystemExit('No labels met the confidence threshold; aborting')
    dataset = build_multiclass_dataset(labs, pd.read_parquet(articles_path), max_per_class=args.max_per_class)
    dataset_path = out_dir / 'topic_dataset.parquet'
    dataset.to_parquet(dataset_path, index=False)
    print(f"Balanced dataset written -> {dataset_path} ({len(dataset)} rows)")

    # Step 3: Train multiclass classifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score
    import joblib
    import numpy as np

    X = np.vstack(dataset['embedding'].to_numpy())
    y_cat = dataset['label'].astype('category')
    y = y_cat.cat.codes
    classes = list(y_cat.cat.categories)

    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    clf = LogisticRegression(max_iter=2000, multi_class='auto')
    clf.fit(Xtr, ytr)
    acc = accuracy_score(yte, clf.predict(Xte))

    model_payload = {
        'model': clf,
        'classes': classes,
        'threshold': args.threshold,
        'min_conf': args.min_conf
    }
    model_path = out_dir / 'topic_model.joblib'
    joblib.dump(model_payload, model_path)
    print(f"Model saved -> {model_path}  Val accuracy={acc:.3f}  Classes={classes}")


if __name__ == '__main__':
    main()
