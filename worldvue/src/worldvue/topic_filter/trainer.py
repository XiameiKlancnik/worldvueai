from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.model_selection import train_test_split


@dataclass
class TopicFilterTrainer:
    model_path: Path
    threshold: float = 0.9  # probability of omit (positive class)

    def _Xy(self, df: pd.DataFrame, embedding_col: str = 'embedding'):
        X = np.vstack(df[embedding_col].to_numpy())
        y = df['y'].astype(int).to_numpy()
        return X, y

    def train(self, df: pd.DataFrame, *, embedding_col: str = 'embedding', test_size: float = 0.2, seed: int = 42) -> dict:
        X, y = self._Xy(df, embedding_col=embedding_col)
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=test_size, random_state=seed, stratify=y)

        clf = LogisticRegression(max_iter=2000, class_weight='balanced', solver='liblinear')
        clf.fit(Xtr, ytr)

        proba = clf.predict_proba(Xte)[:, 1]
        preds = (proba >= self.threshold).astype(int)
        auc = roc_auc_score(yte, proba)
        acc = accuracy_score(yte, preds)

        self._save_model(clf)
        return {'val_auc': float(auc), 'val_acc': float(acc), 'n_train': int(len(Xtr)), 'n_val': int(len(Xte))}

    def _save_model(self, clf):
        self.model_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {'model': clf, 'threshold': self.threshold}
        joblib.dump(payload, self.model_path)


class TopicFilterInferencer:
    def __init__(self, model_path: Path):
        payload = joblib.load(model_path)
        self.model = payload['model']
        self.threshold = float(payload.get('threshold', 0.9))

    def predict(self, embeddings: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        proba = self.model.predict_proba(embeddings)[:, 1]
        preds = (proba >= self.threshold).astype(int)
        return preds, proba

    def apply_to_parquet(self, parquet_path: Path, output_path: Optional[Path] = None,
                         id_col: str = 'article_id', embedding_col: str = 'embedding') -> Path:
        df = pd.read_parquet(parquet_path)
        X = np.vstack(df[embedding_col].to_numpy())
        preds, proba = self.predict(X)
        df['omit_pred'] = preds
        df['omit_conf'] = proba
        out = output_path or parquet_path
        df.to_parquet(out, index=False)
        return out
