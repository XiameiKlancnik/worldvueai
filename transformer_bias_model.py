"""
Advanced Transformer-based Bias Detection Model
Using sentence transformers with contrastive learning and multi-task heads
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModel
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report
from typing import Dict, List, Tuple, Optional
import json
from pathlib import Path

class BiasDataset(Dataset):
    """Dataset for bias detection training"""

    def __init__(self, pairs_file: str, articles_df: pd.DataFrame, max_length: int = 512):
        self.pairs = self.load_pairs(pairs_file)
        self.articles = articles_df.set_index('id')
        self.max_length = max_length

        # Create article ID to text mapping
        self.article_texts = self.articles['text'].to_dict()

        # Filter pairs where we have both articles
        self.valid_pairs = []
        for pair in self.pairs:
            if (pair['article_a_id'] in self.article_texts and
                pair['article_b_id'] in self.article_texts):
                self.valid_pairs.append(pair)

        print(f"Loaded {len(self.valid_pairs)} valid pairs from {len(self.pairs)} total")

    def load_pairs(self, pairs_file: str) -> List[Dict]:
        """Load pairs from JSONL file"""
        pairs = []
        with open(pairs_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    pairs.append(json.loads(line))
        return pairs

    def __len__(self):
        return len(self.valid_pairs)

    def __getitem__(self, idx):
        pair = self.valid_pairs[idx]

        # Get article texts
        text_a = self.article_texts[pair['article_a_id']][:self.max_length]
        text_b = self.article_texts[pair['article_b_id']][:self.max_length]

        # Extract labels (convert to classification targets)
        labels = {}

        # Use LLM labels if available, otherwise weak labels
        if pair.get('llm_labels'):
            for axis, llm_data in pair['llm_labels'].items():
                winner = llm_data.get('winner', 'tie')
                labels[axis] = self.winner_to_label(winner)
        else:
            # Convert weak label scores to winners
            for axis, score in pair.get('weak_labels', {}).items():
                if abs(score) > 0.1:  # Only use confident weak labels
                    labels[axis] = 1 if score > 0 else 0
                else:
                    labels[axis] = 2  # tie

        return {
            'text_a': text_a,
            'text_b': text_b,
            'labels': labels,
            'pair_id': pair['pair_id']
        }

    def winner_to_label(self, winner: str) -> int:
        """Convert winner string to classification label"""
        mapping = {'A': 0, 'B': 1, 'tie': 2}
        return mapping.get(winner, 2)

class ContrastiveBiasModel(nn.Module):
    """
    Transformer model for bias detection using contrastive learning
    """

    def __init__(
        self,
        model_name: str = 'all-MiniLM-L6-v2',
        bias_axes: List[str] = None,
        hidden_dim: int = 384,
        dropout: float = 0.1
    ):
        super().__init__()

        self.bias_axes = bias_axes or ['hype', 'sourcing', 'fight_vs_fix', 'certain_vs_caution', 'one_sidedness']

        # Load pre-trained sentence transformer
        self.encoder = SentenceTransformer(model_name)
        self.hidden_dim = hidden_dim

        # Freeze encoder initially (will unfreeze for fine-tuning)
        for param in self.encoder.parameters():
            param.requires_grad = False

        # Comparison network
        self.comparison_layers = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim * 2),  # [emb_a, emb_b, abs_diff]
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # Multi-task heads for each bias axis
        self.bias_heads = nn.ModuleDict({
            axis: nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim // 2, 3)  # A wins, B wins, tie
            ) for axis in self.bias_axes
        })

        # Contrastive learning head
        self.contrastive_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 128)  # Embedding space for contrastive loss
        )

    def encode_text(self, texts: List[str]) -> torch.Tensor:
        """Encode texts using sentence transformer"""
        with torch.no_grad():
            embeddings = self.encoder.encode(texts, convert_to_tensor=True)
        return embeddings

    def forward(self, text_a_batch: List[str], text_b_batch: List[str]) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        # Encode both articles
        emb_a = self.encode_text(text_a_batch)
        emb_b = self.encode_text(text_b_batch)

        # Create comparison features
        abs_diff = torch.abs(emb_a - emb_b)
        comparison_input = torch.cat([emb_a, emb_b, abs_diff], dim=1)

        # Pass through comparison network
        comparison_features = self.comparison_layers(comparison_input)

        # Get predictions for each bias axis
        bias_predictions = {}
        for axis in self.bias_axes:
            bias_predictions[axis] = self.bias_heads[axis](comparison_features)

        # Get contrastive embeddings
        contrastive_emb = self.contrastive_head(comparison_features)

        return bias_predictions, contrastive_emb

    def unfreeze_encoder(self):
        """Unfreeze encoder for fine-tuning"""
        for param in self.encoder.parameters():
            param.requires_grad = True

class BiasTrainer:
    """Training pipeline for bias detection model"""

    def __init__(
        self,
        model: ContrastiveBiasModel,
        train_dataset: BiasDataset,
        val_dataset: Optional[BiasDataset] = None,
        batch_size: int = 16,
        learning_rate: float = 1e-4,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        self.model = model.to(device)
        self.device = device
        self.batch_size = batch_size

        self.train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=self.collate_fn
        )

        self.val_loader = None
        if val_dataset:
            self.val_loader = DataLoader(
                val_dataset,
                batch_size=batch_size,
                shuffle=False,
                collate_fn=self.collate_fn
            )

        self.optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=learning_rate
        )

        self.criterion = nn.CrossEntropyLoss(ignore_index=-1)

    def collate_fn(self, batch):
        """Custom collate function for batch processing"""
        text_a_batch = [item['text_a'] for item in batch]
        text_b_batch = [item['text_b'] for item in batch]

        # Create label tensors for each axis
        labels_batch = {}
        for axis in self.model.bias_axes:
            axis_labels = []
            for item in batch:
                if axis in item['labels']:
                    axis_labels.append(item['labels'][axis])
                else:
                    axis_labels.append(-1)  # Ignore this sample for this axis
            labels_batch[axis] = torch.tensor(axis_labels, dtype=torch.long)

        return {
            'text_a': text_a_batch,
            'text_b': text_b_batch,
            'labels': labels_batch
        }

    def train_epoch(self):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        num_batches = 0

        for batch in self.train_loader:
            self.optimizer.zero_grad()

            # Move labels to device
            labels = {axis: labels.to(self.device)
                     for axis, labels in batch['labels'].items()}

            # Forward pass
            bias_predictions, contrastive_emb = self.model(
                batch['text_a'],
                batch['text_b']
            )

            # Calculate loss for each axis
            total_batch_loss = 0
            valid_losses = 0

            for axis in self.model.bias_axes:
                if axis in bias_predictions and axis in labels:
                    # Only calculate loss for samples with valid labels
                    valid_mask = labels[axis] != -1
                    if valid_mask.sum() > 0:
                        axis_loss = self.criterion(
                            bias_predictions[axis][valid_mask],
                            labels[axis][valid_mask]
                        )
                        total_batch_loss += axis_loss
                        valid_losses += 1

            if valid_losses > 0:
                avg_loss = total_batch_loss / valid_losses
                avg_loss.backward()
                self.optimizer.step()

                total_loss += avg_loss.item()
                num_batches += 1

        return total_loss / num_batches if num_batches > 0 else 0

    def evaluate(self):
        """Evaluate model on validation set"""
        if not self.val_loader:
            return {}

        self.model.eval()
        all_predictions = {axis: [] for axis in self.model.bias_axes}
        all_labels = {axis: [] for axis in self.model.bias_axes}

        with torch.no_grad():
            for batch in self.val_loader:
                labels = {axis: labels.to(self.device)
                         for axis, labels in batch['labels'].items()}

                bias_predictions, _ = self.model(batch['text_a'], batch['text_b'])

                for axis in self.model.bias_axes:
                    if axis in bias_predictions and axis in labels:
                        valid_mask = labels[axis] != -1
                        if valid_mask.sum() > 0:
                            preds = torch.argmax(bias_predictions[axis][valid_mask], dim=1)
                            all_predictions[axis].extend(preds.cpu().numpy())
                            all_labels[axis].extend(labels[axis][valid_mask].cpu().numpy())

        # Calculate metrics
        metrics = {}
        for axis in self.model.bias_axes:
            if all_predictions[axis] and all_labels[axis]:
                accuracy = accuracy_score(all_labels[axis], all_predictions[axis])
                metrics[axis] = accuracy

        return metrics

    def train(self, num_epochs: int = 10, unfreeze_after: int = 5):
        """Full training loop"""
        print(f"Training for {num_epochs} epochs on {self.device}")

        best_metrics = {}

        for epoch in range(num_epochs):
            # Unfreeze encoder for fine-tuning after initial training
            if epoch == unfreeze_after:
                print("Unfreezing encoder for fine-tuning...")
                self.model.unfreeze_encoder()
                # Reduce learning rate for fine-tuning
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] *= 0.1

            # Train
            train_loss = self.train_epoch()

            # Evaluate
            val_metrics = self.evaluate()

            print(f"Epoch {epoch+1}/{num_epochs}")
            print(f"  Train Loss: {train_loss:.4f}")

            if val_metrics:
                print("  Validation Accuracy:")
                for axis, acc in val_metrics.items():
                    print(f"    {axis}: {acc:.3f}")
                    if axis not in best_metrics or acc > best_metrics[axis]:
                        best_metrics[axis] = acc

        return best_metrics

def create_training_pipeline(
    articles_csv: str = "all_articles.csv",
    pairs_file: str = "artifacts/pairs_judged.jsonl",
    model_save_path: str = "artifacts/transformer_bias_model.pt"
):
    """Complete training pipeline"""

    print("Loading data...")
    articles_df = pd.read_csv(articles_csv)

    # Load dataset
    dataset = BiasDataset(pairs_file, articles_df)

    # Split into train/val
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size

    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )

    print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}")

    # Create model
    model = ContrastiveBiasModel()

    # Create trainer
    trainer = BiasTrainer(model, train_dataset, val_dataset)

    # Train
    best_metrics = trainer.train(num_epochs=15, unfreeze_after=8)

    # Save model
    torch.save(model.state_dict(), model_save_path)
    print(f"Model saved to {model_save_path}")

    print("\nBest validation accuracies:")
    for axis, acc in best_metrics.items():
        print(f"  {axis}: {acc:.3f}")

    return model, best_metrics

if __name__ == "__main__":
    # Run training pipeline
    model, metrics = create_training_pipeline()