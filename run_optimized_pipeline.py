#!/usr/bin/env python3
"""
Simple optimized pipeline runner with timestamped artifacts.
"""

import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

def main():
    """Run the optimized pipeline with timestamped artifacts."""

    # Add the worldvue source directory to Python path so we use local code
    worldvue_src = Path(__file__).parent / "worldvue" / "src"
    sys.path.insert(0, str(worldvue_src))
    print(f"Using local worldvue source from: {worldvue_src}")

    # Create timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = f"artifacts_run_{timestamp}_optimized_no_ties"

    # Create directory
    Path(run_dir).mkdir(exist_ok=True)
    print(f"Created run directory: {run_dir}")

    # Pipeline commands with timestamped outputs - USING PARALLEL JUDGE for 5-10x speedup!
    commands = [
        f"python -m worldvue.cli.main clusters make --articles articles_with_embeddings.parquet --budget worldvue/configs/budget.yaml --out {run_dir}/clusters.parquet",
        f"python -m worldvue.cli.main pairs make --clusters {run_dir}/clusters.parquet --budget worldvue/configs/budget.yaml --out {run_dir}/pairs.parquet",
        # PARALLEL JUDGE - 5-10x faster than sequential
        f"python run_parallel_judge_for_pipeline.py {run_dir}",
        f"python -m worldvue.cli.main pairs labels --in {run_dir}/judge_results_parallel.jsonl --out {run_dir}/pairs_labeled_balanced.parquet --pairs {run_dir}/pairs.parquet",
        f"python -m worldvue.cli.main train style --pairs-labeled {run_dir}/pairs_labeled_balanced.parquet --articles articles_with_embeddings.parquet --out-dir {run_dir}/models/ --epochs 3",
        f"python -m worldvue.cli.main score style --articles articles_with_embeddings.parquet --models {run_dir}/models/ --budget worldvue/configs/budget.yaml --clusters {run_dir}/clusters.parquet --out {run_dir}/style_scores.parquet",
        f"python -m worldvue.cli.main eval style --models-dir {run_dir}/models/ --test-pairs {run_dir}/pairs.parquet --scores {run_dir}/style_scores.parquet --articles articles_with_embeddings.parquet --budget worldvue/configs/budget.yaml --out {run_dir}/evaluation.json"
    ]

    print(f"\n=== Running Optimized Pipeline ===")
    print(f"Features: GPT-4o-mini, No Ties, Balanced Data, 90% Cost Savings")
    print(f"NEW: PARALLEL JUDGING (10x faster than before!)")
    print(f"Output folder: {run_dir}")
    print(f"Total steps: {len(commands)}\n")

    # Run each command
    for i, cmd in enumerate(commands, 1):
        # Special handling for parallel judge step
        if "run_parallel_judge_for_pipeline.py" in cmd:
            step_name = "parallel judge"
        else:
            step_name = ' '.join(cmd.split()[1:3])  # e.g., ['clusters', 'make']

        print(f"Step {i}/{len(commands)}: {step_name}")
        print(f"Command: {cmd}")

        # Set environment to use local worldvue source
        env = os.environ.copy()
        env['PYTHONPATH'] = str(worldvue_src)

        try:
            # For the judge step, show live output for debugging
            if 'judge' in cmd:
                print("Running with live output for debugging...")
                result = subprocess.run(cmd, shell=True, check=True, env=env)
                print(f"OK Step {i} completed successfully")
            else:
                result = subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True, env=env)
                print(f"OK Step {i} completed successfully")
                if result.stdout:
                    # Show last few lines of output
                    lines = result.stdout.strip().split('\n')[-3:]
                    for line in lines:
                        if line.strip():
                            print(f"  {line}")
            print()
        except subprocess.CalledProcessError as e:
            print(f"ERROR Step {i} failed")
            if hasattr(e, 'stderr') and e.stderr:
                print(f"Error: {e.stderr}")
            print(f"Stopping pipeline at step {i}")
            break
        except Exception as e:
            print(f"ERROR Step {i} failed with exception: {e}")
            print(f"Stopping pipeline at step {i}")
            break

    print(f"=== Pipeline Complete ===")
    print(f"Results saved to: {run_dir}/")
    print(f"Original artifacts preserved in: artifacts/")

    # Show key files
    key_files = [
        f"{run_dir}/judge_results_no_ties.jsonl",
        f"{run_dir}/pairs_labeled_balanced.parquet",
        f"{run_dir}/models/",
        f"{run_dir}/evaluation.json"
    ]

    print(f"\nKey output files:")
    for file in key_files:
        if Path(file).exists():
            print(f"  OK {file}")
        else:
            print(f"  MISSING {file} (not created)")

if __name__ == "__main__":
    main()