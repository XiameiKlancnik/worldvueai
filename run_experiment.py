#!/usr/bin/env python
"""
Run WorldVue experiment with automatic folder creation and tracking
"""

import os
import sys
import json
import shutil
import subprocess
from datetime import datetime
from pathlib import Path
import yaml

def create_experiment_folder(description=""):
    """Create a new timestamped experiment folder"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_name = f"exp_{timestamp}"
    if description:
        # Sanitize description for folder name
        safe_desc = "".join(c for c in description if c.isalnum() or c in "-_").lower()[:30]
        exp_name = f"exp_{timestamp}_{safe_desc}"

    exp_path = Path("experiments") / exp_name
    exp_path.mkdir(parents=True, exist_ok=True)

    # Create artifacts subdirectory
    artifacts_path = exp_path / "artifacts"
    artifacts_path.mkdir(exist_ok=True)

    return exp_path

def load_budget_config():
    """Load and return current budget config"""
    with open("worldvue/configs/budget.yaml", "r") as f:
        return yaml.safe_load(f)

def save_metadata(exp_path, config, description=""):
    """Save experiment metadata"""
    metadata = {
        "timestamp": datetime.now().isoformat(),
        "description": description,
        "settings": {
            "target_pairs": config.get("target_pairs_total"),
            "votes_per_pair": config.get("votes_per_pair"),
            "truncate_chars": config.get("truncate_chars_per_side"),
            "model": "gpt-4o-mini" if config.get("use_cheaper_model") else "gpt-4o",
            "multi_axis": config.get("use_multi_axis_judging"),
            "dry_run": config.get("dry_run"),
            "price_per_mtoken": config.get("price_per_mtoken_usd")
        },
        "estimated_cost": calculate_cost_estimate(config)
    }

    with open(exp_path / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    return metadata

def calculate_cost_estimate(config):
    """Calculate estimated cost based on config"""
    pairs = config.get("target_pairs_total", 100)
    votes = config.get("votes_per_pair", 1)
    chars = config.get("truncate_chars_per_side", 500) * 2

    # Estimate tokens (1.3 tokens per char + overhead)
    tokens_per_call = int(chars * 1.3) + 1000

    # Total calls depends on multi-axis setting
    if config.get("use_multi_axis_judging"):
        total_calls = pairs * votes
    else:
        total_calls = pairs * votes * 5

    total_tokens = total_calls * tokens_per_call
    price_per_m = config.get("price_per_mtoken_usd", 0.15)

    return (total_tokens / 1_000_000) * price_per_m

def run_pipeline(exp_path, steps=None):
    """Run the WorldVue pipeline with specified steps"""
    artifacts = str(exp_path / "artifacts")

    if steps is None:
        steps = ["pairs", "judge", "labels", "train", "score"]

    commands = {
        "pairs": [
            "worldvue", "pairs", "make",
            "--clusters", "artifacts/clusters.parquet",
            "--budget", "worldvue/configs/budget.yaml",
            "--out", f"{artifacts}/pairs.parquet"
        ],
        "judge": [
            "worldvue", "judge", "style",
            "--pairs", f"{artifacts}/pairs.parquet",
            "--articles", "articles_with_embeddings.parquet",
            "--budget", "worldvue/configs/budget.yaml",
            "--out", f"{artifacts}/judge_results.jsonl"
        ],
        "labels": [
            "worldvue", "pairs", "labels",
            "--in", f"{artifacts}/judge_results.jsonl",
            "--out", f"{artifacts}/pairs_labeled.parquet"
        ],
        "train": [
            "worldvue", "train", "style",
            "--pairs-labeled", f"{artifacts}/pairs_labeled.parquet",
            "--articles", "articles_with_embeddings.parquet",
            "--out-dir", f"{artifacts}/models/",
            "--epochs", "3"
        ],
        "score": [
            "worldvue", "score", "style",
            "--articles", "articles_with_embeddings.parquet",
            "--models", f"{artifacts}/models/",
            "--budget", "worldvue/configs/budget.yaml",
            "--clusters", "artifacts/clusters.parquet",
            "--out", f"{artifacts}/style_scores.parquet"
        ]
    }

    results = {}
    for step in steps:
        if step in commands:
            print(f"\n{'='*60}")
            print(f"Running step: {step}")
            print(f"{'='*60}")

            cmd = commands[step]
            print(f"Command: {' '.join(cmd)}")

            result = subprocess.run(cmd, capture_output=True, text=True)
            results[step] = {
                "returncode": result.returncode,
                "stdout": result.stdout[-5000:],  # Last 5000 chars
                "stderr": result.stderr[-5000:]
            }

            if result.returncode != 0:
                print(f"ERROR in step {step}:")
                print(result.stderr)
                break
            else:
                print(f"✓ {step} completed successfully")

    # Save results log
    with open(exp_path / "pipeline_results.json", "w") as f:
        json.dump(results, f, indent=2)

    return results

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Run WorldVue experiment")
    parser.add_argument("--description", "-d", help="Experiment description", default="")
    parser.add_argument("--steps", "-s", nargs="+",
                       choices=["pairs", "judge", "labels", "train", "score"],
                       help="Steps to run (default: all)")
    parser.add_argument("--dry-run", action="store_true",
                       help="Set dry_run=true in config temporarily")

    args = parser.parse_args()

    # Load config
    config = load_budget_config()

    # Override dry_run if specified
    original_dry_run = config.get("dry_run")
    if args.dry_run:
        config["dry_run"] = True
        # Temporarily update the config file
        with open("worldvue/configs/budget.yaml", "w") as f:
            yaml.dump(config, f, default_flow_style=False)

    # Create experiment folder
    exp_path = create_experiment_folder(args.description)
    print(f"Created experiment folder: {exp_path}")

    # Save metadata
    metadata = save_metadata(exp_path, config, args.description)
    print(f"Estimated cost: ${metadata['estimated_cost']:.2f}")

    if config.get("dry_run"):
        print("Running in DRY RUN mode (no real LLM calls)")

    # Run pipeline
    results = run_pipeline(exp_path, args.steps)

    # Restore original dry_run setting
    if args.dry_run and not original_dry_run:
        config["dry_run"] = original_dry_run
        with open("worldvue/configs/budget.yaml", "w") as f:
            yaml.dump(config, f, default_flow_style=False)

    print(f"\n{'='*60}")
    print(f"Experiment complete!")
    print(f"Results saved to: {exp_path}")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()