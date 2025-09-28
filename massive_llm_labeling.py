"""
Massive LLM Labeling for High-Quality Training Data
Since LLM labeling costs only $0.0004 per judgment, we can afford to label 10K+ pairs!
"""

import os
import sys
import json
import time
from pathlib import Path
from typing import List, Dict, Any
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
from dotenv import load_dotenv

# Add worldvue src to Python path
sys.path.insert(0, str(Path(__file__).parent / "worldvue" / "src"))

from worldvue.data.loaders import load_articles
from worldvue.data.types import Pair
from worldvue.judge.client import BudgetAwareJudge

load_dotenv()

class MassiveLabelingPipeline:
    """Pipeline for labeling thousands of pairs with LLM"""

    def __init__(
        self,
        articles_csv: str = "all_articles.csv",
        pairs_file: str = "artifacts/pairs_weak.jsonl",
        output_file: str = "artifacts/pairs_massive_llm.jsonl",
        budget_usd: float = 50.0,
        max_pairs: int = 10000,
        batch_size: int = 5,  # Pairs per API call
        workers: int = 5  # Parallel workers
    ):
        self.articles_csv = articles_csv
        self.pairs_file = pairs_file
        self.output_file = output_file
        self.budget_usd = budget_usd
        self.max_pairs = max_pairs
        self.batch_size = batch_size
        self.workers = workers

        # Thread safety
        self.lock = Lock()
        self.total_spent = 0.0
        self.pairs_judged = 0
        self.failures = 0

        # Load data
        self.articles = self.load_articles()
        self.pairs = self.load_pairs()

        print(f"Loaded {len(self.articles)} articles and {len(self.pairs)} pairs")

    def load_articles(self) -> Dict[str, Any]:
        """Load articles and create ID mapping"""
        articles_list = load_articles(Path(self.articles_csv))
        return {article.id: article for article in articles_list}

    def load_pairs(self) -> List[Pair]:
        """Load pairs from JSONL file"""
        pairs = []
        with open(self.pairs_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    pair_data = json.loads(line)
                    pair = Pair(**pair_data)

                    # Attach articles if available
                    if (pair.article_a_id in self.articles and
                        pair.article_b_id in self.articles):
                        pair.attach_articles(
                            self.articles[pair.article_a_id],
                            self.articles[pair.article_b_id]
                        )
                        pairs.append(pair)

        return pairs[:self.max_pairs]

    def select_diverse_pairs(self, n: int) -> List[Pair]:
        """Select diverse pairs for labeling"""
        # Strategy: Select pairs that are:
        # 1. From different sources
        # 2. Have disagreement among weak labelers
        # 3. Cover different topics/time periods

        selected = []
        source_counts = {}

        # Sort by weak labeler disagreement (uncertainty)
        pairs_with_uncertainty = []
        for pair in self.pairs:
            if pair.weak_labels:
                # Calculate disagreement across axes
                scores = list(pair.weak_labels.values())
                if scores:
                    uncertainty = 1.0 - (max(abs(s) for s in scores))  # Low max score = high uncertainty
                    pairs_with_uncertainty.append((pair, uncertainty))

        # Sort by uncertainty (descending)
        pairs_with_uncertainty.sort(key=lambda x: x[1], reverse=True)

        # Select diverse pairs
        for pair, uncertainty in pairs_with_uncertainty:
            if len(selected) >= n:
                break

            # Get source diversity
            source_a = self.articles[pair.article_a_id].source_name
            source_b = self.articles[pair.article_b_id].source_name

            # Prefer pairs from different sources
            source_key = f"{source_a}#{source_b}"
            if source_counts.get(source_key, 0) < 20:  # Max 20 pairs per source combo
                selected.append(pair)
                source_counts[source_key] = source_counts.get(source_key, 0) + 1

        print(f"Selected {len(selected)} diverse pairs for labeling")
        return selected

    def create_batch_prompt(self, pairs: List[Pair]) -> str:
        """Create a batch prompt for multiple pairs"""
        prompt = """You are an expert media analyst. Compare these article pairs on journalistic style dimensions.

For each pair, analyze these 5 dimensions:
1. hype: Which article is more sensational/dramatic vs factual/measured?
2. sourcing: Which article has better source attribution and citations?
3. fight_vs_fix: Which article focuses more on solutions vs conflicts/blame?
4. certain_vs_caution: Which article is more assertive vs cautious in claims?
5. one_sidedness: Which article presents more balanced perspectives?

For each dimension, respond with:
- "A" if article A is better on this dimension
- "B" if article B is better on this dimension
- "tie" if they're roughly equal
- confidence: 0.1 to 1.0 (how certain you are)

Return JSON array with one object per pair:

"""

        for i, pair in enumerate(pairs):
            article_a = self.articles[pair.article_a_id]
            article_b = self.articles[pair.article_b_id]

            # Truncate articles to fit in context
            text_a = article_a.text[:800]
            text_b = article_b.text[:800]

            prompt += f"""
Pair {i+1}:
Article A: {text_a}
Article B: {text_b}

"""

        prompt += """
JSON format:
[
  {
    "pair_index": 1,
    "hype": {"winner": "A", "confidence": 0.8},
    "sourcing": {"winner": "B", "confidence": 0.9},
    "fight_vs_fix": {"winner": "tie", "confidence": 0.5},
    "certain_vs_caution": {"winner": "A", "confidence": 0.7},
    "one_sidedness": {"winner": "B", "confidence": 0.6}
  },
  ...
]
"""
        return prompt

    def judge_batch(self, pairs_batch: List[Pair], judge: BudgetAwareJudge) -> List[Dict]:
        """Judge a batch of pairs"""
        try:
            # Create batch prompt
            prompt = self.create_batch_prompt(pairs_batch)

            # Make API call
            response = judge.client.chat.completions.create(
                model=judge.model,
                temperature=judge.temperature,
                max_tokens=judge.max_tokens * len(pairs_batch),  # Scale tokens for batch
                response_format={'type': 'json_object'},
                messages=[
                    {'role': 'system', 'content': 'You are an expert media analyst. Return valid JSON only.'},
                    {'role': 'user', 'content': prompt}
                ]
            )

            # Calculate cost
            tokens_in = getattr(response.usage, 'prompt_tokens', 0)
            tokens_out = getattr(response.usage, 'completion_tokens', 0)
            cost = tokens_in * judge.prompt_rate + tokens_out * judge.completion_rate

            # Parse response
            content = response.choices[0].message.content
            batch_results = json.loads(content)

            # Attach results to pairs
            labeled_pairs = []
            for i, result in enumerate(batch_results):
                if i < len(pairs_batch):
                    pair = pairs_batch[i]
                    pair.llm_labels = {
                        axis: data for axis, data in result.items()
                        if axis in ['hype', 'sourcing', 'fight_vs_fix', 'certain_vs_caution', 'one_sidedness']
                    }
                    pair.cost_usd += cost / len(pairs_batch)
                    labeled_pairs.append(pair)

            return labeled_pairs, cost

        except Exception as e:
            print(f"Error judging batch: {e}")
            return [], 0.0

    def worker_thread(self, pairs_queue: List[List[Pair]]) -> List[Pair]:
        """Worker thread for parallel processing"""
        judge = BudgetAwareJudge(budget_usd=self.budget_usd / self.workers)
        labeled_pairs = []

        for batch in pairs_queue:
            with self.lock:
                if self.total_spent >= self.budget_usd:
                    break

            batch_pairs, cost = self.judge_batch(batch, judge)

            with self.lock:
                self.total_spent += cost
                self.pairs_judged += len(batch_pairs)
                labeled_pairs.extend(batch_pairs)

                if len(batch_pairs) > 0:
                    print(f"Judged batch of {len(batch_pairs)} pairs (${self.total_spent:.3f} spent, {self.pairs_judged} total)")
                else:
                    self.failures += 1

        return labeled_pairs

    def run_massive_labeling(self):
        """Run the massive labeling pipeline"""
        print("="*60)
        print(" MASSIVE LLM LABELING PIPELINE")
        print("="*60)
        print(f"Budget: ${self.budget_usd}")
        print(f"Max pairs: {self.max_pairs}")
        print(f"Batch size: {self.batch_size}")
        print(f"Workers: {self.workers}")

        # Select diverse pairs
        selected_pairs = self.select_diverse_pairs(self.max_pairs)

        # Create batches
        batches = []
        for i in range(0, len(selected_pairs), self.batch_size):
            batches.append(selected_pairs[i:i + self.batch_size])

        print(f"Created {len(batches)} batches")

        # Distribute batches across workers
        batches_per_worker = len(batches) // self.workers
        worker_queues = []
        for i in range(self.workers):
            start_idx = i * batches_per_worker
            end_idx = start_idx + batches_per_worker if i < self.workers - 1 else len(batches)
            worker_queues.append(batches[start_idx:end_idx])

        # Run parallel labeling
        print("Starting parallel LLM labeling...")
        start_time = time.time()

        all_labeled_pairs = []
        with ThreadPoolExecutor(max_workers=self.workers) as executor:
            futures = [executor.submit(self.worker_thread, queue) for queue in worker_queues]

            for future in as_completed(futures):
                try:
                    labeled_pairs = future.result()
                    all_labeled_pairs.extend(labeled_pairs)
                except Exception as e:
                    print(f"Worker error: {e}")

        elapsed = time.time() - start_time

        # Save results
        self.save_labeled_pairs(all_labeled_pairs)

        # Print statistics
        print("="*60)
        print(" LABELING COMPLETE")
        print("="*60)
        print(f"Time elapsed: {elapsed:.1f} seconds")
        print(f"Pairs labeled: {len(all_labeled_pairs)}")
        print(f"Total cost: ${self.total_spent:.3f}")
        print(f"Cost per pair: ${self.total_spent/max(1, len(all_labeled_pairs)):.4f}")
        print(f"Failures: {self.failures}")
        print(f"Success rate: {len(all_labeled_pairs)/(len(all_labeled_pairs)+self.failures)*100:.1f}%")

        # Show coverage by axis
        axis_coverage = {}
        for pair in all_labeled_pairs:
            if pair.llm_labels:
                for axis in pair.llm_labels:
                    axis_coverage[axis] = axis_coverage.get(axis, 0) + 1

        print("\nLLM label coverage:")
        for axis, count in axis_coverage.items():
            print(f"  {axis}: {count} pairs")

        return all_labeled_pairs

    def save_labeled_pairs(self, labeled_pairs: List[Pair]):
        """Save labeled pairs to JSONL file"""
        # Combine with original pairs (preserve weak labels)
        all_pairs = []

        # Create a mapping of labeled pairs
        labeled_map = {pair.pair_id: pair for pair in labeled_pairs}

        # Merge with original pairs
        for pair in self.pairs:
            if pair.pair_id in labeled_map:
                # Use the labeled version
                all_pairs.append(labeled_map[pair.pair_id])
            else:
                # Keep original with weak labels only
                all_pairs.append(pair)

        # Save to file
        with open(self.output_file, 'w', encoding='utf-8') as f:
            for pair in all_pairs:
                pair_dict = pair.model_dump(exclude={'article_a', 'article_b'})
                f.write(json.dumps(pair_dict) + '\n')

        print(f"Saved {len(all_pairs)} pairs to {self.output_file}")

def main():
    """Main function"""
    import argparse

    parser = argparse.ArgumentParser(description='Massive LLM Labeling Pipeline')
    parser.add_argument('--articles', default='all_articles.csv', help='Articles CSV file')
    parser.add_argument('--pairs', default='artifacts/pairs_weak.jsonl', help='Input pairs file')
    parser.add_argument('--output', default='artifacts/pairs_massive_llm.jsonl', help='Output file')
    parser.add_argument('--budget', type=float, default=50.0, help='Budget in USD')
    parser.add_argument('--max-pairs', type=int, default=10000, help='Max pairs to label')
    parser.add_argument('--batch-size', type=int, default=5, help='Pairs per API call')
    parser.add_argument('--workers', type=int, default=5, help='Parallel workers')

    args = parser.parse_args()

    pipeline = MassiveLabelingPipeline(
        articles_csv=args.articles,
        pairs_file=args.pairs,
        output_file=args.output,
        budget_usd=args.budget,
        max_pairs=args.max_pairs,
        batch_size=args.batch_size,
        workers=args.workers
    )

    labeled_pairs = pipeline.run_massive_labeling()
    print(f"\n✅ Massive labeling complete! {len(labeled_pairs)} pairs labeled for ${pipeline.total_spent:.3f}")

if __name__ == "__main__":
    main()