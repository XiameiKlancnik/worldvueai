#!/usr/bin/env python3
"""Fine-tune a lightweight transformer for topic classification."""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

import torch
from transformers import (AutoTokenizer, AutoModelForSequenceClassification,
                          Trainer, TrainingArguments)


def load_data(dataset_path: Path, articles_path: Path, text_col: str = 'text'):
    df = pd.read_parquet(dataset_path)
    articles = pd.read_parquet(articles_path)
    if 'article_id' not in df.columns:
        raise ValueError('dataset parquet must contain article_id column')
    if 'article_id' not in articles.columns:
        if 'id' in articles.columns:
            articles = articles.assign(article_id=articles['id'].astype(str))
        else:
            raise ValueError('articles parquet must contain article_id or id column')
    if text_col not in articles.columns:
        raise ValueError(f"articles parquet missing '{text_col}' column")
    merged = df.merge(articles[['article_id', text_col]], on='article_id', how='left')
    missing = merged[text_col].isna().sum()
    if missing:
        print(f"Warning: {missing} rows missing text; dropping them")
        merged = merged.dropna(subset=[text_col])
    merged[text_col] = merged[text_col].astype(str).str.strip()
    merged = merged[merged[text_col] != '']
    merged = merged.reset_index(drop=True)
    return merged


def tokenize_dataset(df, tokenizer, text_col, max_len):
    encodings = tokenizer(df[text_col].tolist(), truncation=True, padding='max_length',
                          max_length=max_len, return_tensors='pt')
    labels = torch.tensor(df['label_id'].tolist(), dtype=torch.long)
    dataset = torch.utils.data.TensorDataset(encodings['input_ids'], encodings['attention_mask'], labels)
    return dataset


def main():
    parser = argparse.ArgumentParser(description='Fine-tune transformer for topic classification')
    parser.add_argument('--dataset', required=True, help='Balanced topic dataset parquet (with article_id, label)')
    parser.add_argument('--articles', required=True, help='Articles parquet containing full text')
    parser.add_argument('--output-dir', required=True, help='Directory to save model')
    parser.add_argument('--model-name', default='distilbert-base-multilingual-cased',
                        help='HF model name (default: multilingual DistilBERT)')
    parser.add_argument('--text-col', default='text', help='Text column in articles parquet')
    parser.add_argument('--epochs', type=int, default=3, help='Training epochs')
    parser.add_argument('--batch-size', type=int, default=16, help='Batch size')
    parser.add_argument('--max-length', type=int, default=256, help='Token max length')
    parser.add_argument('--lr', type=float, default=5e-5, help='Learning rate')
    args = parser.parse_args()

    dataset_path = Path(args.dataset)
    articles_path = Path(args.articles)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = load_data(dataset_path, articles_path, text_col=args.text_col)
    label_cat = df['label'].astype('category')
    df = df.assign(label_id=label_cat.cat.codes)
    classes = label_cat.cat.categories.tolist()
    print(f"Loaded dataset with {len(df)} samples across {len(classes)} classes")

    train_df, val_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['label_id'])
    print(f"Train: {len(train_df)}  Val: {len(val_df)}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    train_dataset = tokenize_dataset(train_df, tokenizer, args.text_col, args.max_length)
    val_dataset = tokenize_dataset(val_df, tokenizer, args.text_col, args.max_length)

    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=len(classes)
    )

    training_args = TrainingArguments(
        output_dir=str(output_dir / 'trainer'),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        evaluation_strategy='epoch',
        save_strategy='epoch',
        learning_rate=args.lr,
        load_best_model_at_end=True,
        metric_for_best_model='accuracy',
        greater_is_better=True,
        logging_steps=50,
        save_total_limit=2
    )

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        acc = accuracy_score(labels, preds)
        return {'accuracy': acc}

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )

    trainer.train()
    metrics = trainer.evaluate()
    print(f"Final validation accuracy: {metrics['eval_accuracy']:.3f}")

    trainer.save_model(str(output_dir / 'model'))
    tokenizer.save_pretrained(str(output_dir / 'model'))

    # Save class mapping
    mapping_path = output_dir / 'label_mapping.json'
    import json
    with mapping_path.open('w', encoding='utf-8') as f:
        json.dump({'classes': classes}, f, ensure_ascii=False, indent=2)
    print(f"Model saved to {output_dir / 'model'}  Classes={classes}")


if __name__ == '__main__':
    main()
