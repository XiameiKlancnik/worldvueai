from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
from sklearn.linear_model import LogisticRegression

from worldvue.data.types import Pair, STYLE_AXES
from worldvue.weak.features import extract_pair_features, diff_vector
from worldvue.weak.labelers import score_to_winner, winner_to_score

try:  # pragma: no cover - optional dependency
    import lightgbm as lgb
except Exception:
    lgb = None


def _build_features(pair: Pair) -> np.ndarray:
    pair.ensure_articles()
    _, _, diff = extract_pair_features(pair.article_a.text, pair.article_b.text)
    return diff_vector(diff)


def _collect_training_examples(
    pairs: Iterable[Pair],
    axis: str,
    *,
    use_llm: bool = True,
    llm_weight: float = 3.0,
) -> Tuple[List[np.ndarray], List[int], List[float]]:
    features: List[np.ndarray] = []
    labels: List[int] = []
    weights: List[float] = []
    for pair in pairs:
        vector = _build_features(pair)
        score = 0.0
        weight = pair.weak_confidence.get(axis, 0.5)
        if use_llm and pair.llm_labels and axis in pair.llm_labels:
            winner = pair.llm_labels[axis].get('winner', 'tie')
            score = winner_to_score(str(winner))
            weight = float(pair.llm_labels[axis].get('confidence', 0.5)) * llm_weight
        else:
            score = pair.weak_labels.get(axis, 0.0)
        if score == 0:
            continue
        label = 1 if score > 0 else 0
        features.append(vector)
        labels.append(label)
        weights.append(max(weight, 0.1))
    return features, labels, weights


def _train_logistic(X: List[np.ndarray], y: List[int], sample_weight: List[float]) -> LogisticRegression:
    model = LogisticRegression(max_iter=500, class_weight='balanced')
    model.fit(np.stack(X), np.asarray(y), sample_weight=np.asarray(sample_weight))
    return model


def _train_lightgbm(X: List[np.ndarray], y: List[int], sample_weight: List[float]):
    if lgb is None:
        raise RuntimeError('lightgbm is not installed')
    dataset = lgb.Dataset(np.stack(X), label=np.asarray(y), weight=np.asarray(sample_weight))
    params = {
        'objective': 'binary',
        'learning_rate': 0.05,
        'num_leaves': 15,
        'max_depth': -1,
        'verbosity': -1,
    }
    booster = lgb.train(params, dataset, num_boost_round=60)
    return booster


def train_style_model(
    all_pairs: Iterable[Pair],
    *,
    calibration_pairs: Optional[Iterable[Pair]] = None,
    prefer_lightgbm: bool = False,
) -> Dict[str, object]:
    pairs = list(all_pairs)
    calibration = list(calibration_pairs or [])
    models: Dict[str, object] = {}
    for axis in STYLE_AXES:
        combined = pairs + calibration
        X, y, weights = _collect_training_examples(combined, axis)
        if len(X) < 5 or len(set(y)) < 2:
            continue
        if prefer_lightgbm:
            model = _train_lightgbm(X, y, weights)
        else:
            model = _train_logistic(X, y, weights)
        models[axis] = model
    return models


def predict_axis(model, vector: np.ndarray) -> float:
    if hasattr(model, 'predict_proba'):
        proba = model.predict_proba(vector.reshape(1, -1))[0][1]
        return float(2 * proba - 1)
    if hasattr(model, 'predict'):
        prediction = model.predict(vector.reshape(1, -1))[0]
        return float(prediction)
    raise ValueError('Unsupported model interface')


def evaluate_models(models: Dict[str, object], test_pairs: Iterable[Pair]) -> Dict[str, float]:
    scores: Dict[str, List[float]] = {axis: [] for axis in STYLE_AXES}
    labels: Dict[str, List[float]] = {axis: [] for axis in STYLE_AXES}
    for pair in test_pairs:
        vector = _build_features(pair)
        for axis, model in models.items():
            prediction = predict_axis(model, vector)
            scores[axis].append(prediction)
            labels[axis].append(pair.weak_labels.get(axis, 0.0))
    metrics: Dict[str, float] = {}
    for axis in models.keys():
        if not scores[axis]:
            continue
        predicted = np.sign(scores[axis])
        truth = np.sign(labels[axis])
        accuracy = float(np.mean(predicted == truth))
        metrics[axis] = accuracy
    return metrics


__all__ = ['train_style_model', 'evaluate_models', 'predict_axis']
