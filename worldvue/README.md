# WorldVue

WorldVue is a production-ready news bias analytics system built for high-volume, multilingual newsrooms. It combines weak supervision, multilingual embeddings, strategic LLM sampling, and calibrated training to deliver actionable bias metrics under strict budget and compute constraints.

## Why WorldVue

- **Budget aware**: Operates with a total OpenAI budget of $10 and produces valuable insights even at $0 spend.
- **Multilingual**: Uses multilingual sentence embeddings and language-agnostic features, no translation required.
- **High throughput**: Handles 13K historical articles plus 100+ new articles per hour on CPU-only hardware.
- **Progressive quality**: Start with weak labels, then incrementally improve accuracy via targeted LLM judgments.
- **Cost tracking**: Every API call is logged in `costs.jsonl` with cumulative spend monitoring.

## Architecture Overview

WorldVue is organized into two layers:

1. **Layer A – Style**: Quantifies how articles are written across five rhetorical axes (`one_sidedness`, `hype`, `sourcing`, `fight_vs_fix`, `certain_vs_caution`).
2. **Layer B – Policy**: Maps the substantive policy positions to ideological axes (`econ_left_right`, `social_lib_cons`).

Layer A is delivered out-of-the-box. Layer B hooks are present via the same infrastructure and can be enabled by adding policy-specific labeling functions and rankers.

## Repository Layout

```
worldvue/
  pyproject.toml
  Makefile
  README.md
  .env.example
  configs/
    weak_labels.yaml
    judge.yaml
    sampling.yaml
  src/worldvue/
    ...
  tests/
    test_weak_labels.py
    test_sampling.py
    test_budget.py
  artifacts/
    .gitkeep
  costs.jsonl
```

## Getting Started

```bash
make install
cp .env.example .env
# populate OPENAI_API_KEY in .env
```

## CLI Highlights

- `worldvue embed --input articles.parquet` – cache multilingual embeddings
- `worldvue cluster --min-size 3` – generate clusters for weak supervision
- `worldvue weak label --all` – apply labeling functions across pairs
- `worldvue validate --sample 100 --budget 0.10` – calibrate with a small LLM budget
- `worldvue judge strategic --n 200 --budget 0.50` – refine with targeted judgments
- `worldvue train style --method hybrid` – fit the production ranker
- `worldvue costs show` – inspect OpenAI spend at any time

## Budget Tiers

| Spend | Accuracy | Confidence | Recommended Use |
|-------|----------|------------|-----------------|
| $0.00 | 0.65     | Low        | Obvious bias detection |
| $0.10 | 0.70     | Calibrated | Trend analysis |
| $0.50 | 0.75     | Good       | Production insights |
| $2.00 | 0.80     | High       | Research quality |

## Validation Metrics

- Weak vs LLM agreement rate (target ≥70%)
- Labeling function coverage and accuracy
- Cross-lingual stability
- Temporal consistency
- Null hypothesis (random pairs → tie)

## Development

- Python 3.11
- Poetry for dependency management
- CPU-only execution path with caching for embeddings and model artifacts

## License

Proprietary – consult WorldVue AI for licensing details.
