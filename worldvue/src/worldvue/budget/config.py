from dataclasses import dataclass, field
from typing import List, Optional
import yaml
from pathlib import Path


@dataclass
class BudgetConfig:
    # Article sampling
    seed: int = 42
    max_articles_global: int = 1500
    max_articles_per_country: int = 60
    country_whitelist: List[str] = field(default_factory=list)
    cluster_min_size: int = 6

    # Pair sampling
    target_pairs_total: int = 5000
    cross_country_ratio: float = 0.40
    pairs_per_cluster_min: int = 10
    pairs_per_cluster_max: int = 40
    max_pairs_per_article: int = 12
    balance_by_outlet: bool = True

    # LLM judging
    votes_per_pair: int = 3
    truncate_chars_per_side: int = 1200
    pivot_language: str = "en"
    use_translation: bool = True
    entity_masking: bool = True
    use_multi_axis_judging: bool = False
    use_cheaper_model: bool = False

    # Cost estimator
    price_per_mtoken_usd: float = 0.15
    tokens_per_call_estimate: int = 1000
    dry_run: bool = False

    # Scoring/calibration
    scoring_method: str = "anchors"  # "anchors" | "bt_offsets"
    anchors_per_axis: int = 7
    country_isotonic_calibration: bool = True

    @classmethod
    def from_yaml(cls, path: Path) -> 'BudgetConfig':
        with open(path, 'r') as f:
            data = yaml.safe_load(f)
        return cls(**data)

    def to_yaml(self, path: Path):
        with open(path, 'w') as f:
            yaml.dump(self.__dict__, f, default_flow_style=False)

    @property
    def within_country_ratio(self) -> float:
        return 1.0 - self.cross_country_ratio

    @property
    def total_llm_calls(self) -> int:
        # Account for multi-axis optimization
        if self.use_multi_axis_judging:
            # One call per vote (judges all 5 axes at once)
            return self.target_pairs_total * self.votes_per_pair
        else:
            # One call per axis per vote
            return self.target_pairs_total * self.votes_per_pair * 5

    @property
    def estimated_tokens(self) -> int:
        # Token estimate based on truncation setting
        # ~1.3 tokens per char, 2 articles, plus prompts/response
        chars_total = self.truncate_chars_per_side * 2
        tokens_from_text = int(chars_total * 1.3)
        tokens_overhead = 1000  # Prompts + JSON response
        realistic_tokens_per_call = tokens_from_text + tokens_overhead
        return self.total_llm_calls * realistic_tokens_per_call

    @property
    def estimated_cost_usd(self) -> float:
        return (self.estimated_tokens / 1_000_000) * self.price_per_mtoken_usd


def load_budget_config(path: str | Path) -> BudgetConfig:
    return BudgetConfig.from_yaml(Path(path))