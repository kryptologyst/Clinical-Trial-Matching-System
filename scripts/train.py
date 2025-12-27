#!/usr/bin/env python3
"""Training script for clinical trial matching system."""

import argparse
import logging
import yaml
from pathlib import Path
import torch
from torch.utils.data import DataLoader, random_split

from src.utils import set_seed, get_device, setup_logging, Config
from src.data import ClinicalTrialDataset, DataProcessor
from src.models import HybridRetrievalModel
from src.train import Trainer, TrainingConfig, ClinicalTrialDataset as PyTorchDataset
from src.metrics import ClinicalTrialEvaluator
from src.eval import ModelEvaluator

logger = logging.getLogger(__name__)


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file.
    
    Args:
        config_path: Path to configuration file.
        
    Returns:
        Configuration dictionary.
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def create_data_loaders(config: dict) -> tuple:
    """Create data loaders for training and validation.
    
    Args:
        config: Configuration dictionary.
        
    Returns:
        Tuple of (train_loader, val_loader, test_loader).
    """
    # Load dataset
    dataset = ClinicalTrialDataset()
    
    # Create retrieval pairs
    processor = DataProcessor(max_length=config['model']['max_length'])
    pairs = processor.create_retrieval_pairs(dataset)
    
    # Split data
    train_size = int(config['data']['train_split'] * len(pairs))
    val_size = int(config['data']['val_split'] * len(pairs))
    test_size = len(pairs) - train_size - val_size
    
    train_pairs, val_pairs, test_pairs = random_split(
        pairs, [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(config['data']['random_seed'])
    )
    
    # Create PyTorch datasets
    train_dataset = PyTorchDataset(
        [p['patient_text'] for p in train_pairs],
        [p['trial_text'] for p in train_pairs],
        [p['label'] for p in train_pairs],
        None,  # Tokenizer will be handled by model
        config['model']['max_length']
    )
    
    val_dataset = PyTorchDataset(
        [p['patient_text'] for p in val_pairs],
        [p['trial_text'] for p in val_pairs],
        [p['label'] for p in val_pairs],
        None,
        config['model']['max_length']
    )
    
    test_dataset = PyTorchDataset(
        [p['patient_text'] for p in test_pairs],
        [p['trial_text'] for p in test_pairs],
        [p['label'] for p in test_pairs],
        None,
        config['model']['max_length']
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=config['system']['num_workers'],
        pin_memory=config['system']['pin_memory']
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=config['system']['num_workers'],
        pin_memory=config['system']['pin_memory']
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=config['system']['num_workers'],
        pin_memory=config['system']['pin_memory']
    )
    
    return train_loader, val_loader, test_loader


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description='Train clinical trial matching model')
    parser.add_argument('--config', type=str, default='configs/default.yaml',
                       help='Path to configuration file')
    parser.add_argument('--output_dir', type=str, default='outputs',
                       help='Output directory for models and logs')
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint to resume from')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Set random seed
    set_seed(args.seed)
    
    # Setup logging
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    log_file = output_dir / 'training.log'
    setup_logging(config['logging']['level'], str(log_file))
    
    logger.info("Starting clinical trial matching training")
    logger.info(f"Configuration: {args.config}")
    logger.info(f"Output directory: {args.output_dir}")
    
    # Get device
    device = get_device()
    logger.info(f"Using device: {device}")
    
    # Create data loaders
    logger.info("Creating data loaders")
    train_loader, val_loader, test_loader = create_data_loaders(config)
    
    # Initialize model
    logger.info("Initializing model")
    model = HybridRetrievalModel(
        model_name=config['model']['name'],
        max_length=config['model']['max_length'],
        dropout=config['model']['dropout']
    )
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        train_dataloader=train_loader,
        val_dataloader=val_loader,
        device=device,
        learning_rate=config['training']['learning_rate'],
        num_epochs=config['training']['num_epochs'],
        save_dir=str(output_dir / 'checkpoints')
    )
    
    # Resume from checkpoint if provided
    if args.resume:
        logger.info(f"Resuming from checkpoint: {args.resume}")
        trainer.load_checkpoint(args.resume)
    
    # Train model
    logger.info("Starting training")
    history = trainer.train()
    
    # Evaluate on test set
    logger.info("Evaluating on test set")
    evaluator = ModelEvaluator(model, device)
    
    # Get test data
    test_patient_texts = []
    test_trial_texts = []
    test_labels = []
    
    for batch in test_loader:
        test_patient_texts.append(batch['patient_text'])
        test_trial_texts.append(batch['trial_text'])
        test_labels.append(batch['label'].item())
    
    # Evaluate
    test_metrics = evaluator.evaluate_retrieval_performance(
        test_patient_texts,
        test_trial_texts,
        test_labels,
        top_k_values=config['evaluation']['top_k_values']
    )
    
    # Generate evaluation report
    report = evaluator.generate_evaluation_report(test_metrics)
    logger.info("Test Results:")
    logger.info(report)
    
    # Save evaluation report
    report_path = output_dir / 'test_evaluation_report.txt'
    with open(report_path, 'w') as f:
        f.write(report)
    
    # Save training history
    history_path = output_dir / 'training_history.json'
    import json
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    
    logger.info("Training completed successfully")
    logger.info(f"Results saved to: {output_dir}")


if __name__ == '__main__':
    main()
