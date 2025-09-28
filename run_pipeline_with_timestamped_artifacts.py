#!/usr/bin/env python3
"""
Run WorldVue pipeline with timestamped artifact folders to prevent overwriting.
"""

import os
import sys
import subprocess
from datetime import datetime
from pathlib import Path

def create_timestamped_run():
    """Create a new timestamped run directory and return the path."""

    # Create timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"run_{timestamp}_optimized_no_ties"

    # Create directory structure
    run_dir = Path(f"artifacts_{run_name}")
    run_dir.mkdir(exist_ok=True)

    print(f"Created new run directory: {run_dir}")
    return run_dir

def run_pipeline_step(command_parts, description, run_dir):
    """Run a pipeline step with the correct output directory."""

    print(f"\n=== {description} ===")
    print(f"Command: {' '.join(command_parts)}")

    try:
        # Run the command
        result = subprocess.run(
            command_parts,
            check=True,
            capture_output=True,
            text=True,
            cwd=os.getcwd()
        )

        print(f"OK {description} completed successfully")
        if result.stdout:
            print("Output:", result.stdout[-200:])  # Last 200 chars

        return True

    except subprocess.CalledProcessError as e:
        print(f"ERROR {description} failed")
        print("Error:", e.stderr)
        return False
    except Exception as e:
        print(f"ERROR {description} failed with exception: {e}")
        return False

def main():
    """Run the complete optimized pipeline with timestamped artifacts."""

    print("=== WorldVue Optimized Pipeline Runner ===")
    print("Using GPT-4o-mini, no-tie prompts, and balanced sampling")

    # Create timestamped run directory
    run_dir = create_timestamped_run()

    # Pipeline steps with timestamped outputs
    steps = [
        {
            "command": [
                "worldvue", "clusters", "make",
                "--articles", "articles_with_embeddings.parquet",
                "--budget", "worldvue/configs/budget.yaml",
                "--out", str(run_dir / "clusters.parquet")
            ],
            "description": "1. Global Clustering"
        },
        {
            "command": [
                "worldvue", "pairs", "make",
                "--clusters", str(run_dir / "clusters.parquet"),
                "--budget", "worldvue/configs/budget.yaml",
                "--out", str(run_dir / "pairs.parquet")
            ],
            "description": "2. Hybrid Pair Sampling"
        },
        {
            "command": [
                "worldvue", "judge", "style",
                "--pairs", str(run_dir / "pairs.parquet"),
                "--articles", "articles_with_embeddings.parquet",
                "--budget", "worldvue/configs/budget.yaml",
                "--out", str(run_dir / "judge_results_no_ties.jsonl")
            ],
            "description": "3. LLM Judging (GPT-4o-mini, No Ties)"
        },
        {
            "command": [
                "worldvue", "pairs", "labels",
                "--in", str(run_dir / "judge_results_no_ties.jsonl"),
                "--out", str(run_dir / "pairs_labeled_balanced.parquet")
            ],
            "description": "4. Label Processing (Balanced A/B)"
        },
        {
            "command": [
                "worldvue", "train", "style",
                "--pairs-labeled", str(run_dir / "pairs_labeled_balanced.parquet"),
                "--articles", "articles_with_embeddings.parquet",
                "--out-dir", str(run_dir / "models/"),
                "--epochs", "3"
            ],
            "description": "5. Cross-Encoder Training"
        },
        {
            "command": [
                "worldvue", "score", "style",
                "--articles", "articles_with_embeddings.parquet",
                "--models", str(run_dir / "models/"),
                "--budget", "worldvue/configs/budget.yaml",
                "--clusters", str(run_dir / "clusters.parquet"),
                "--out", str(run_dir / "style_scores.parquet")
            ],
            "description": "6. Article Scoring"
        },
        {
            "command": [
                "worldvue", "eval", "style",
                "--models-dir", str(run_dir / "models/"),
                "--test-pairs", str(run_dir / "pairs.parquet"),
                "--scores", str(run_dir / "style_scores.parquet"),
                "--articles", "articles_with_embeddings.parquet",
                "--budget", "worldvue/configs/budget.yaml",
                "--out", str(run_dir / "evaluation.json")
            ],
            "description": "7. Evaluation"
        }
    ]

    # Run each step
    success_count = 0
    for step in steps:
        if run_pipeline_step(step["command"], step["description"], run_dir):
            success_count += 1
        else:
            print(f"\nERROR Pipeline failed at step: {step['description']}")
            break

    # Summary
    print(f"\n=== PIPELINE SUMMARY ===")
    print(f"Completed: {success_count}/{len(steps)} steps")
    print(f"Artifacts saved to: {run_dir}")
    print(f"Original artifacts preserved in: artifacts/ and artifacts_backup/")

    if success_count == len(steps):
        print("SUCCESS Pipeline completed successfully!")

        # Show key results
        judge_results = run_dir / "judge_results_no_ties.jsonl"
        pairs_labeled = run_dir / "pairs_labeled_balanced.parquet"

        if judge_results.exists():
            print(f"\nJudge Results: {judge_results}")
        if pairs_labeled.exists():
            print(f"Balanced Labels: {pairs_labeled}")

        # Quick analysis
        print(f"\nQUICK ANALYSIS:")
        analysis_script = f'''
import pandas as pd
import json

# Analyze judge results
with open("{judge_results}", "r") as f:
    judge_data = [json.loads(line) for line in f]

tie_count = sum(1 for j in judge_data if j.get("winner") == "Tie")
total_judgments = len(judge_data)

print(f"Total judgments: {{total_judgments}}")
print(f"Tie rate: {{tie_count/total_judgments*100:.1f}}% (should be 0%)")

# Analyze balanced labels
if Path("{pairs_labeled}").exists():
    labels_df = pd.read_parquet("{pairs_labeled}")

    print(f"\\nTraining pairs: {{len(labels_df)}}")
    for axis in labels_df["axis"].unique():
        axis_data = labels_df[labels_df["axis"] == axis]
        a_wins = len(axis_data[axis_data["winner"] == "A"])
        b_wins = len(axis_data[axis_data["winner"] == "B"])
        ties = len(axis_data[axis_data["winner"] == "Tie"])
        print(f"  {{axis}}: A={{a_wins}}, B={{b_wins}}, Ties={{ties}}")
'''

        try:
            exec(analysis_script)
        except Exception as e:
            print(f"Analysis failed: {e}")
    else:
        print("ERROR Pipeline incomplete - check errors above")

if __name__ == "__main__":
    main()