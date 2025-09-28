"""Cross-encoder transformer training for style axis prediction."""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel, AdamW, get_linear_schedule_with_warmup
from typing import Dict, List, Tuple, Optional
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
import json
import click
from sklearn.metrics import roc_auc_score, accuracy_score


class PairwiseDataset(Dataset):
    """Dataset for pairwise article comparisons."""

    def __init__(self, pairs_df: pd.DataFrame, articles_df: pd.DataFrame,
                tokenizer, max_length: int = 512, axis: str = None):
        self.pairs = pairs_df
        self.articles = articles_df.set_index('article_id')
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.axis = axis

        # Filter to specific axis if provided
        if axis:
            self.pairs = self.pairs[self.pairs['axis'] == axis]

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        row = self.pairs.iloc[idx]

        # Get article texts
        text_a = self.articles.loc[row['a_id'], 'text']
        text_b = self.articles.loc[row['b_id'], 'text']

        # Add cluster summary if available
        cluster_summary = row.get('cluster_summary', '')

        # Format input: [CLS] summary [SEP] text_a [SEP] text_b
        if cluster_summary:
            input_text = f"{cluster_summary} [SEP] {text_a} [SEP] {text_b}"
        else:
            input_text = f"{text_a} [SEP] {text_b}"

        # Tokenize
        encoding = self.tokenizer(
            input_text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )

        # Get label (1 if A wins, 0 if B wins, 0.5 for tie)
        label = row['y']

        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'label': torch.tensor(label, dtype=torch.float)
        }


class CrossEncoder(nn.Module):
    """Cross-encoder model for pairwise comparison."""

    def __init__(self, model_name: str = 'xlm-roberta-base', dropout: float = 0.1):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.encoder.config.hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        pooled = outputs.last_hidden_state[:, 0, :]  # CLS token
        pooled = self.dropout(pooled)
        logits = self.classifier(pooled)
        probs = self.sigmoid(logits)
        return probs.squeeze()


class CrossEncoderTrainer:
    """Trainer for cross-encoder models."""

    def __init__(self, model_name: str = 'xlm-roberta-base',
                device: str = None, learning_rate: float = 2e-5):
        self.model_name = model_name
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.learning_rate = learning_rate
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def train_axis_model(self, axis: str, train_df: pd.DataFrame, val_df: pd.DataFrame,
                        articles_df: pd.DataFrame, output_dir: Path,
                        epochs: int = 3, batch_size: int = 16) -> Dict:
        """
        Train a cross-encoder for a specific style axis.

        Args:
            axis: Style axis to train for
            train_df: Training pairs
            val_df: Validation pairs
            articles_df: Article texts
            output_dir: Directory to save model
            epochs: Number of training epochs
            batch_size: Batch size

        Returns:
            Training metrics
        """
        click.echo(f"\nTraining cross-encoder for {axis}")

        # Create datasets
        train_dataset = PairwiseDataset(train_df, articles_df, self.tokenizer, axis=axis)
        val_dataset = PairwiseDataset(val_df, articles_df, self.tokenizer, axis=axis)

        # Create dataloaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        # Initialize model
        model = CrossEncoder(self.model_name).to(self.device)

        # Setup optimizer and scheduler
        optimizer = AdamW(model.parameters(), lr=self.learning_rate)
        total_steps = len(train_loader) * epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=int(0.1 * total_steps),
            num_training_steps=total_steps
        )

        # Loss function
        criterion = nn.BCELoss()

        # Training loop
        best_val_auc = 0
        metrics_history = []

        for epoch in range(epochs):
            # Training
            model.train()
            train_loss = 0
            train_preds = []
            train_labels = []

            for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['label'].to(self.device)

                optimizer.zero_grad()
                outputs = model(input_ids, attention_mask)
                loss = criterion(outputs, labels)

                loss.backward()
                optimizer.step()
                scheduler.step()

                train_loss += loss.item()
                train_preds.extend(outputs.cpu().detach().numpy())
                train_labels.extend(labels.cpu().numpy())

            # Validation
            val_metrics = self._evaluate(model, val_loader, criterion)

            # Calculate metrics
            train_auc = roc_auc_score(
                [1 if l > 0.5 else 0 for l in train_labels],
                train_preds
            )

            metrics = {
                'epoch': epoch + 1,
                'train_loss': train_loss / len(train_loader),
                'train_auc': train_auc,
                'val_loss': val_metrics['loss'],
                'val_auc': val_metrics['auc'],
                'val_accuracy': val_metrics['accuracy']
            }
            metrics_history.append(metrics)

            click.echo(f"Epoch {epoch+1}: Train AUC={train_auc:.3f}, "
                      f"Val AUC={val_metrics['auc']:.3f}, "
                      f"Val Acc={val_metrics['accuracy']:.3f}")

            # Save best model
            if val_metrics['auc'] > best_val_auc:
                best_val_auc = val_metrics['auc']
                self._save_model(model, output_dir / f'crossenc_{axis}')

        return {
            'axis': axis,
            'best_val_auc': best_val_auc,
            'metrics_history': metrics_history
        }

    def _evaluate(self, model: CrossEncoder, dataloader: DataLoader,
                 criterion: nn.Module) -> Dict:
        """Evaluate model on a dataset."""
        model.eval()
        total_loss = 0
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for batch in dataloader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['label'].to(self.device)

                outputs = model(input_ids, attention_mask)
                loss = criterion(outputs, labels)

                total_loss += loss.item()
                all_preds.extend(outputs.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        # Convert to binary for metrics
        binary_labels = [1 if l > 0.5 else 0 for l in all_labels]
        binary_preds = [1 if p > 0.5 else 0 for p in all_preds]

        # Handle single-class case for AUC
        try:
            auc = roc_auc_score(binary_labels, all_preds)
        except ValueError as e:
            if "Only one class present" in str(e):
                # Single class - set AUC to 0.5 (random performance)
                auc = 0.5
                print(f"Warning: Only one class present in validation set, setting AUC=0.5")
            else:
                raise e

        return {
            'loss': total_loss / len(dataloader),
            'auc': auc,
            'accuracy': accuracy_score(binary_labels, binary_preds)
        }

    def _save_model(self, model: CrossEncoder, output_dir: Path):
        """Save model and tokenizer."""
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save model
        torch.save(model.state_dict(), output_dir / 'model.pt')

        # Save config
        config = {
            'model_name': self.model_name,
            'device': str(self.device)
        }
        with open(output_dir / 'config.json', 'w') as f:
            json.dump(config, f)

        # Save tokenizer
        self.tokenizer.save_pretrained(output_dir)

        click.echo(f"Saved model to {output_dir}")

    @classmethod
    def load_model(cls, model_dir: Path) -> Tuple[CrossEncoder, AutoTokenizer]:
        """Load a trained model."""
        # Load config
        with open(model_dir / 'config.json', 'r') as f:
            config = json.load(f)

        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_dir)

        # Load model
        model = CrossEncoder(config['model_name'])
        model.load_state_dict(torch.load(model_dir / 'model.pt'))
        model.eval()

        return model, tokenizer

    def train_all_axes(self, labels_df: pd.DataFrame, articles_df: pd.DataFrame,
                      output_dir: Path, axes: Optional[List[str]] = None) -> Dict:
        """Train models for all style axes."""
        if axes is None:
            axes = labels_df['axis'].unique()

        # Create train/val splits
        from ..pairs.labels import PairLabeler
        labeler = PairLabeler()
        splits = labeler.create_splits(labels_df)

        results = {}
        for axis in axes:
            axis_results = self.train_axis_model(
                axis,
                splits['train'],
                splits['val'],
                articles_df,
                output_dir,
                epochs=3,
                batch_size=16
            )
            results[axis] = axis_results

        # Save overall results
        with open(output_dir / 'training_results.json', 'w') as f:
            json.dump(results, f, indent=2)

        click.echo(f"\nTraining complete. Average validation AUC: "
                  f"{np.mean([r['best_val_auc'] for r in results.values()]):.3f}")

        return results