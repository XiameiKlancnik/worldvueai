import click
from pathlib import Path
import pandas as pd

from worldvue.topic_filter.dataset import build_balanced_dataset, POLICY_INCLUDE_DEFAULT, OMIT_EXCLUDE_DEFAULT
from worldvue.topic_filter.trainer import TopicFilterTrainer, TopicFilterInferencer
from worldvue.topic_filter.llm_dataset import load_llm_labels, build_multiclass_dataset


@click.group()
def topic():
    """Topic filter: build dataset, train model, and apply classifier."""
    pass


@topic.command('build-dataset')
@click.option('--articles', type=click.Path(exists=True), required=True, help='Articles parquet with embeddings')
@click.option('--out', type=click.Path(), required=True, help='Output dataset parquet')
@click.option('--include-terms', type=str, default=','.join(POLICY_INCLUDE_DEFAULT), help='Comma-separated policy terms (keep)')
@click.option('--exclude-terms', type=str, default=','.join(OMIT_EXCLUDE_DEFAULT), help='Comma-separated omit terms')
@click.option('--max-per-class', type=int, default=5000, help='Cap per class for balancing')
def build_dataset_cmd(articles, out, include_terms, exclude_terms, max_per_class):
    articles_df = pd.read_parquet(articles)
    ds = build_balanced_dataset(articles_df,
                                include_terms=[t.strip() for t in include_terms.split(',') if t.strip()],
                                exclude_terms=[t.strip() for t in exclude_terms.split(',') if t.strip()],
                                max_per_class=max_per_class)
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    ds.to_parquet(out, index=False)
    click.echo(f'Wrote balanced dataset: {len(ds)} rows -> {out}')


@topic.command('train')
@click.option('--dataset', type=click.Path(exists=True), required=True, help='Balanced dataset parquet with y and embedding')
@click.option('--out', 'model_out', type=click.Path(), required=True, help='Output model path (.joblib)')
@click.option('--threshold', type=float, default=0.9, help='Probability threshold for omit class')
@click.option('--embedding-col', type=str, default='embedding', help='Embedding column name')
def train_cmd(dataset, model_out, threshold, embedding_col):
    df = pd.read_parquet(dataset)
    trainer = TopicFilterTrainer(model_path=Path(model_out), threshold=threshold)
    metrics = trainer.train(df, embedding_col=embedding_col)
    click.echo(f"Model saved -> {model_out}\nVal AUC={metrics['val_auc']:.3f} Val Acc={metrics['val_acc']:.3f} "
               f"(train={metrics['n_train']}, val={metrics['n_val']})")


@topic.command('apply')
@click.option('--articles', type=click.Path(exists=True), required=True, help='Articles parquet with embeddings')
@click.option('--model', type=click.Path(exists=True), required=True, help='Trained model .joblib')
@click.option('--out', type=click.Path(), required=False, help='Output parquet (default: overwrite input)')
@click.option('--threshold', type=float, default=None, help='Override threshold')
@click.option('--filter/--no-filter', default=False, help='Filter to keep only political (omit_pred=0)')
@click.option('--only-high-confidence', is_flag=True, default=False, help='Keep rows with omit_conf<=0.1 or >=0.9 only')
@click.option('--embedding-col', type=str, default='embedding', help='Embedding column name')
@click.option('--id-col', type=str, default='article_id', help='Article id column name')
def apply_cmd(articles, model, out, threshold, filter, only_high_confidence, embedding_col, id_col):
    df = pd.read_parquet(articles)
    infer = TopicFilterInferencer(Path(model))
    if threshold is not None:
        infer.threshold = float(threshold)
    import numpy as np
    X = np.vstack(df[embedding_col].to_numpy())
    preds, proba = infer.predict(X)
    df['omit_pred'] = preds
    df['omit_conf'] = proba

    if only_high_confidence:
        keep_mask = (proba <= 0.1) | (proba >= 0.9)
        df = df[keep_mask].copy()
    if filter:
        df = df[df['omit_pred'] == 0].copy()

    out_path = Path(out) if out else Path(articles)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)
    click.echo(f'Wrote {len(df)} rows -> {out_path}')

@topic.command('judge-llm')
@click.option('--articles', type=click.Path(exists=True), required=True, help='Articles parquet with title/text (+ article_id or id)')
@click.option('--out-dir', type=click.Path(), required=True, help='Directory to write topic_labels.jsonl')
@click.option('--workers', type=int, default=8, help='Parallel workers')
@click.option('--rpm', type=int, default=300, help='Requests per minute rate limit')
@click.option('--max-rows', type=int, default=None, help='Optional cap on number of articles to judge')
@click.option('--categories', type=str, default=None, help='Comma-separated category names to use (default set if omitted)')
def judge_llm_cmd(articles, out_dir, workers, rpm, max_rows, categories):
    """Label articles into topical categories via LLM (parallel)."""
    from worldvue.topic_filter.parallel_topic_judge import ParallelTopicJudge
    from worldvue.topic_filter.categories import CATEGORIES
    df = pd.read_parquet(articles)
    if max_rows:
        df = df.head(max_rows)
    cats = [c.strip() for c in categories.split(',')] if categories else CATEGORIES
    judge = ParallelTopicJudge(max_workers=workers, requests_per_minute=rpm)
    judge.judge_articles(df, output_dir=Path(out_dir), categories=cats)


@topic.command('dataset-from-llm')
@click.option('--labels', type=click.Path(exists=True), required=True, help='topic_labels.jsonl from judge-llm')
@click.option('--articles', type=click.Path(exists=True), required=True, help='Articles parquet with embeddings')
@click.option('--out', type=click.Path(), required=True, help='Output dataset parquet (article_id, embedding, label)')
@click.option('--min-conf', type=float, default=0.85, help='Minimum label confidence to keep')
@click.option('--max-per-class', type=int, default=3000, help='Balance cap per category')
def dataset_from_llm_cmd(labels, articles, out, min_conf, max_per_class):
    """Build a balanced multiclass dataset from LLM topic labels."""
    labs = load_llm_labels(labels, min_conf=min_conf)
    if labs.empty:
        click.echo('No labels met the confidence threshold; aborting.', err=True)
        return
    from worldvue.topic_filter.categories import CATEGORIES
    canonical = {c.upper(): c for c in CATEGORIES}
    normalized = (labs['label'].astype(str)
                  .str.strip()
                  .str.replace('-', '_')
                  .str.replace(' ', '_'))
    labs['label'] = normalized.str.upper().map(canonical)
    labs = labs.dropna(subset=['label'])
    if 'secondaries' in labs.columns:
        def _normalize_list(lst):
            cleaned = []
            for item in lst:
                norm = str(item).strip().replace('-', '_').replace(' ', '_').upper()
                mapped = canonical.get(norm)
                if mapped and mapped not in cleaned:
                    cleaned.append(mapped)
            return cleaned
        labs['secondaries'] = labs['secondaries'].apply(_normalize_list)
    else:
        labs['secondaries'] = [[] for _ in range(len(labs))]
    if labs.empty:
        click.echo('All high-confidence labels mapped outside allowed categories; aborting.', err=True)
        return
    articles_df = pd.read_parquet(articles)
    dataset = build_multiclass_dataset(labs, articles_df, max_per_class=max_per_class)
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    dataset.to_parquet(out, index=False)
    click.echo(f"Wrote balanced multiclass dataset: {len(dataset)} rows -> {out}")



@topic.command('train-mc')
@click.option('--dataset', type=click.Path(exists=True), required=True, help='Multiclass dataset (label + embedding)')
@click.option('--out', 'model_out', type=click.Path(), required=True, help='Output model path (.joblib)')
@click.option('--embedding-col', type=str, default='embedding', help='Embedding column name')
@click.option('--model-type', type=click.Choice(['lr', 'svm', 'xgb']), default='lr', show_default=True, help='Classifier type to train')
@click.option('--cv-folds', type=int, default=1, show_default=True, help='Stratified k-folds for cross-validation (>=2 to enable)')
def train_mc_cmd(dataset, model_out, embedding_col, model_type, cv_folds):
    """Train a multiclass classifier on embeddings (LR / SVM / XGBoost)."""
    import joblib
    import numpy as np
    from sklearn.model_selection import train_test_split, StratifiedKFold
    from sklearn.metrics import accuracy_score

    df = pd.read_parquet(dataset)
    embeddings = df[embedding_col].to_numpy()
    X = np.vstack(embeddings)
    label_cat = df['label'].astype('category')
    y = label_cat.cat.codes
    classes = label_cat.cat.categories.tolist()
    secondaries_series = df['secondaries'] if 'secondaries' in df.columns else [[] for _ in range(len(df))]
    secondary_lists = []
    for val in secondaries_series:
        cleaned = []
        if isinstance(val, (list, tuple)):
            cleaned = list(dict.fromkeys(val))
        elif hasattr(val, 'tolist'):
            try:
                cleaned = list(dict.fromkeys(val.tolist()))
            except TypeError:
                cleaned = list(dict.fromkeys(list(val)))
        elif pd.isna(val):
            cleaned = []
        else:
            cleaned = [val]
        secondary_lists.append(cleaned)

    def build_model():
        if model_type == 'lr':
            from sklearn.linear_model import LogisticRegression
            return LogisticRegression(max_iter=2000, multi_class='auto')
        if model_type == 'svm':
            from sklearn.svm import SVC
            return SVC(kernel='rbf', C=1.0, gamma='scale', probability=True)
        if model_type == 'xgb':
            try:
                import xgboost as xgb
            except ImportError as exc:
                raise click.ClickException('xgboost is not installed. Install it with "pip install xgboost" to use --model-type xgb.') from exc
            return xgb.XGBClassifier(
                objective='multi:softprob',
                num_class=len(classes),
                eval_metric='mlogloss',
                learning_rate=0.1,
                max_depth=6,
                subsample=0.9,
                colsample_bytree=0.9,
                n_estimators=300,
                tree_method='hist'
            )
        raise click.ClickException(f'Unsupported model type: {model_type}')

    if cv_folds > 1:
        skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
        scores = []
        for train_idx, val_idx in skf.split(X, y):
            model_cv = build_model()
            Xtr, Xval = X[train_idx], X[val_idx]
            ytr, yval = y[train_idx], y[val_idx]
            if model_type == 'xgb':
                Xtr = Xtr.astype('float32')
                Xval = Xval.astype('float32')
            model_cv.fit(Xtr, ytr)
            preds = model_cv.predict(Xval)
            scores.append(accuracy_score(yval, preds))
        click.echo(f'CV ({cv_folds} folds) accuracy: mean={np.mean(scores):.3f} std={np.std(scores):.3f}')

    Xtr, Xte, ytr, yte, sec_tr, sec_te = train_test_split(
        X, y, secondary_lists, test_size=0.2, random_state=42, stratify=y)
    if model_type == 'xgb':
        Xtr = Xtr.astype('float32')
        Xte = Xte.astype('float32')
    model = build_model()
    model.fit(Xtr, ytr)
    preds = model.predict(Xte)
    strict_acc = accuracy_score(yte, preds)

    relaxed_scores = []
    for pred_idx, true_idx, sec_list in zip(preds, yte, sec_te):
        predicted_label = classes[pred_idx]
        true_label = classes[true_idx]
        if predicted_label == true_label:
            relaxed_scores.append(1.0)
        elif predicted_label in sec_list:
            relaxed_scores.append(0.5)
        else:
            relaxed_scores.append(0.0)
    relaxed_acc = float(np.mean(relaxed_scores)) if relaxed_scores else 0.0

    payload = {'model': model, 'classes': classes, 'model_type': model_type}
    Path(model_out).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(payload, model_out)
    click.echo(f"Model saved -> {model_out}  Val Acc={strict_acc:.3f}  Relaxed Acc={relaxed_acc:.3f}  Classes={classes}")


@topic.command('apply-mc')
@click.option('--articles', type=click.Path(exists=True), required=True, help='Articles parquet with embeddings')
@click.option('--model', type=click.Path(exists=True), required=True, help='Trained multiclass model .joblib')
@click.option('--out', type=click.Path(), required=False, help='Output parquet (default overwrite input)')
@click.option('--embedding-col', type=str, default='embedding', help='Embedding column name')
def apply_mc_cmd(articles, model, out, embedding_col):
    import joblib, numpy as np
    df = pd.read_parquet(articles)
    payload = joblib.load(model)
    clf = payload['model']
    classes = payload['classes']
    X = np.vstack(df[embedding_col].to_numpy())
    probs = clf.predict_proba(X)
    pred_idx = probs.argmax(axis=1)
    df['topic_pred'] = [classes[i] for i in pred_idx]
    df['topic_conf'] = probs.max(axis=1)
    out_path = Path(out) if out else Path(articles)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)
    click.echo(f'Wrote {len(df)} rows -> {out_path}')

