"""Training utilities for clinical trial matching models."""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from typing import Dict, List, Optional, Tuple, Any
import logging
from tqdm import tqdm
import numpy as np
from pathlib import Path

logger = logging.getLogger(__name__)


class ClinicalTrialDataset(Dataset):
    """PyTorch Dataset for clinical trial matching."""
    
    def __init__(
        self,
        patient_texts: List[str],
        trial_texts: List[str],
        labels: List[int],
        tokenizer: Any,
        max_length: int = 512
    ):
        """Initialize dataset.
        
        Args:
            patient_texts: List of patient text descriptions.
            trial_texts: List of trial text descriptions.
            labels: Binary labels for patient-trial pairs.
            tokenizer: Tokenizer for text processing.
            max_length: Maximum sequence length.
        """
        self.patient_texts = patient_texts
        self.trial_texts = trial_texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self) -> int:
        """Get dataset length."""
        return len(self.patient_texts)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get item by index.
        
        Args:
            idx: Item index.
            
        Returns:
            Dictionary containing patient text, trial text, and label.
        """
        patient_text = self.patient_texts[idx]
        trial_text = self.trial_texts[idx]
        label = self.labels[idx]
        
        return {
            'patient_text': patient_text,
            'trial_text': trial_text,
            'label': torch.tensor(label, dtype=torch.float)
        }


class Trainer:
    """Trainer class for clinical trial matching models."""
    
    def __init__(
        self,
        model: nn.Module,
        train_dataloader: DataLoader,
        val_dataloader: Optional[DataLoader] = None,
        device: Optional[torch.device] = None,
        learning_rate: float = 2e-5,
        num_epochs: int = 3,
        save_dir: Optional[str] = None
    ):
        """Initialize trainer.
        
        Args:
            model: Model to train.
            train_dataloader: Training data loader.
            val_dataloader: Validation data loader.
            device: Device to run training on.
            learning_rate: Learning rate for optimizer.
            num_epochs: Number of training epochs.
            save_dir: Directory to save checkpoints.
        """
        self.model = model
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.num_epochs = num_epochs
        self.save_dir = Path(save_dir) if save_dir else None
        
        # Move model to device
        self.model.to(self.device)
        
        # Initialize optimizer and loss
        self.optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
        self.criterion = nn.BCEWithLogitsLoss()
        
        # Training history
        self.train_losses = []
        self.val_losses = []
        self.val_metrics = []
        
        logger.info(f"Initialized trainer on {self.device}")
    
    def train_epoch(self) -> float:
        """Train for one epoch.
        
        Returns:
            Average training loss.
        """
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        progress_bar = tqdm(self.train_dataloader, desc="Training")
        
        for batch in progress_bar:
            # Move batch to device
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                    for k, v in batch.items()}
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Forward pass
            try:
                # For dual encoder model
                if hasattr(self.model, 'dual_encoder'):
                    patient_embeddings = self.model.dual_encoder.encode_patients([batch['patient_text']])
                    trial_embeddings = self.model.dual_encoder.encode_trials([batch['trial_text']])
                    similarity = self.model.dual_encoder.compute_similarity(patient_embeddings, trial_embeddings)
                    logits = similarity.squeeze()
                else:
                    # For cross-encoder model
                    logits = self.model([batch['patient_text']], [batch['trial_text']])
                
                # Compute loss
                loss = self.criterion(logits, batch['label'])
                
                # Backward pass
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                # Update parameters
                self.optimizer.step()
                
                # Update metrics
                total_loss += loss.item()
                num_batches += 1
                
                # Update progress bar
                progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})
                
            except Exception as e:
                logger.error(f"Error in training batch: {e}")
                continue
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        return avg_loss
    
    def validate(self) -> Tuple[float, Dict[str, float]]:
        """Validate model.
        
        Returns:
            Tuple of (average validation loss, validation metrics).
        """
        if self.val_dataloader is None:
            return 0.0, {}
        
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        all_predictions = []
        all_labels = []
        all_scores = []
        
        with torch.no_grad():
            for batch in tqdm(self.val_dataloader, desc="Validation"):
                # Move batch to device
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                        for k, v in batch.items()}
                
                try:
                    # Forward pass
                    if hasattr(self.model, 'dual_encoder'):
                        patient_embeddings = self.model.dual_encoder.encode_patients([batch['patient_text']])
                        trial_embeddings = self.model.dual_encoder.encode_trials([batch['trial_text']])
                        similarity = self.model.dual_encoder.compute_similarity(patient_embeddings, trial_embeddings)
                        logits = similarity.squeeze()
                    else:
                        logits = self.model([batch['patient_text']], [batch['trial_text']])
                    
                    # Compute loss
                    loss = self.criterion(logits, batch['label'])
                    total_loss += loss.item()
                    num_batches += 1
                    
                    # Store predictions for metrics
                    predictions = (torch.sigmoid(logits) > 0.5).float()
                    all_predictions.append(predictions)
                    all_labels.append(batch['label'])
                    all_scores.append(torch.sigmoid(logits))
                    
                except Exception as e:
                    logger.error(f"Error in validation batch: {e}")
                    continue
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        
        # Compute metrics
        if all_predictions:
            all_predictions = torch.cat(all_predictions)
            all_labels = torch.cat(all_labels)
            all_scores = torch.cat(all_scores)
            
            metrics = self._compute_metrics(all_predictions, all_labels, all_scores)
        else:
            metrics = {}
        
        return avg_loss, metrics
    
    def _compute_metrics(
        self, 
        predictions: torch.Tensor, 
        labels: torch.Tensor, 
        scores: torch.Tensor
    ) -> Dict[str, float]:
        """Compute validation metrics.
        
        Args:
            predictions: Predicted labels.
            labels: True labels.
            scores: Prediction scores.
            
        Returns:
            Dictionary of metrics.
        """
        from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
        
        metrics = {}
        
        # Convert to numpy
        pred_np = predictions.cpu().numpy()
        label_np = labels.cpu().numpy()
        score_np = scores.cpu().numpy()
        
        # Classification metrics
        metrics['precision'] = precision_score(label_np, pred_np, zero_division=0)
        metrics['recall'] = recall_score(label_np, pred_np, zero_division=0)
        metrics['f1'] = f1_score(label_np, pred_np, zero_division=0)
        
        # Ranking metrics
        if len(np.unique(label_np)) > 1:  # Check if we have both classes
            metrics['auc_roc'] = roc_auc_score(label_np, score_np)
        
        return metrics
    
    def train(self) -> Dict[str, List[float]]:
        """Train the model.
        
        Returns:
            Training history.
        """
        logger.info(f"Starting training for {self.num_epochs} epochs")
        
        best_val_loss = float('inf')
        
        for epoch in range(self.num_epochs):
            logger.info(f"Epoch {epoch + 1}/{self.num_epochs}")
            
            # Train
            train_loss = self.train_epoch()
            self.train_losses.append(train_loss)
            
            # Validate
            val_loss, val_metrics = self.validate()
            self.val_losses.append(val_loss)
            self.val_metrics.append(val_metrics)
            
            # Log metrics
            logger.info(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
            if val_metrics:
                logger.info(f"Val Metrics: {val_metrics}")
            
            # Save checkpoint if validation loss improved
            if val_loss < best_val_loss and self.save_dir:
                best_val_loss = val_loss
                self.save_checkpoint(epoch, is_best=True)
            
            # Save regular checkpoint
            if self.save_dir:
                self.save_checkpoint(epoch, is_best=False)
        
        logger.info("Training completed")
        
        return {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'val_metrics': self.val_metrics
        }
    
    def save_checkpoint(self, epoch: int, is_best: bool = False):
        """Save model checkpoint.
        
        Args:
            epoch: Current epoch number.
            is_best: Whether this is the best model so far.
        """
        if not self.save_dir:
            return
        
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'val_metrics': self.val_metrics
        }
        
        # Save regular checkpoint
        checkpoint_path = self.save_dir / f'checkpoint_epoch_{epoch}.pt'
        torch.save(checkpoint, checkpoint_path)
        
        # Save best model
        if is_best:
            best_path = self.save_dir / 'best_model.pt'
            torch.save(checkpoint, best_path)
            logger.info(f"Saved best model to {best_path}")
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load model checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint file.
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.train_losses = checkpoint.get('train_losses', [])
        self.val_losses = checkpoint.get('val_losses', [])
        self.val_metrics = checkpoint.get('val_metrics', [])
        
        logger.info(f"Loaded checkpoint from {checkpoint_path}")


class TrainingConfig:
    """Configuration class for training."""
    
    def __init__(
        self,
        batch_size: int = 32,
        learning_rate: float = 2e-5,
        num_epochs: int = 3,
        max_length: int = 512,
        warmup_steps: int = 100,
        weight_decay: float = 0.01,
        gradient_clip_norm: float = 1.0,
        save_steps: int = 500,
        eval_steps: int = 500,
        logging_steps: int = 100
    ):
        """Initialize training configuration.
        
        Args:
            batch_size: Training batch size.
            learning_rate: Learning rate.
            num_epochs: Number of training epochs.
            max_length: Maximum sequence length.
            warmup_steps: Number of warmup steps.
            weight_decay: Weight decay for regularization.
            gradient_clip_norm: Gradient clipping norm.
            save_steps: Steps between saving checkpoints.
            eval_steps: Steps between evaluations.
            logging_steps: Steps between logging.
        """
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.max_length = max_length
        self.warmup_steps = warmup_steps
        self.weight_decay = weight_decay
        self.gradient_clip_norm = gradient_clip_norm
        self.save_steps = save_steps
        self.eval_steps = eval_steps
        self.logging_steps = logging_steps
