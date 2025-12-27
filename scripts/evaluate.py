#!/usr/bin/env python3
"""Evaluation script for clinical trial matching system."""

import argparse
import logging
import yaml
from pathlib import Path
import torch
import json

from src.utils import set_seed, get_device, setup_logging
from src.data import ClinicalTrialDataset, DataProcessor
from src.models import HybridRetrievalModel, ClinicalTrialMatcher
from src.eval import ModelEvaluator, BenchmarkEvaluator

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


def evaluate_model(model_path: str, config: dict) -> dict:
    """Evaluate a trained model.
    
    Args:
        model_path: Path to trained model checkpoint.
        config: Configuration dictionary.
        
    Returns:
        Dictionary of evaluation metrics.
    """
    # Get device
    device = get_device()
    
    # Load model
    logger.info(f"Loading model from {model_path}")
    checkpoint = torch.load(model_path, map_location=device)
    
    model = HybridRetrievalModel(
        model_name=config['model']['name'],
        max_length=config['model']['max_length'],
        dropout=config['model']['dropout']
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    
    # Load dataset
    dataset = ClinicalTrialDataset()
    processor = DataProcessor(max_length=config['model']['max_length'])
    pairs = processor.create_retrieval_pairs(dataset)
    
    # Split data for evaluation
    test_size = int(config['data']['test_split'] * len(pairs))
    test_pairs = pairs[-test_size:]  # Use last portion as test set
    
    # Prepare test data
    test_patient_texts = [p['patient_text'] for p in test_pairs]
    test_trial_texts = [p['trial_text'] for p in test_pairs]
    test_labels = [p['label'] for p in test_pairs]
    
    # Initialize evaluator
    evaluator = ModelEvaluator(model, device)
    
    # Evaluate model
    metrics = evaluator.evaluate_retrieval_performance(
        test_patient_texts,
        test_trial_texts,
        test_labels,
        top_k_values=config['evaluation']['top_k_values']
    )
    
    return metrics


def run_demo_matching(config: dict):
    """Run demo clinical trial matching.
    
    Args:
        config: Configuration dictionary.
    """
    logger.info("Running demo clinical trial matching")
    
    # Initialize matcher
    matcher = ClinicalTrialMatcher(
        model_name=config['model']['name'],
        device=get_device()
    )
    
    # Load dataset
    dataset = ClinicalTrialDataset()
    
    # Demo patient profile
    patient_profile = "65-year-old female with stage II HER2-positive breast cancer seeking targeted therapy"
    
    # Get trial texts
    trial_texts = [dataset.get_trial_text(trial['id']) for trial in dataset.trials]
    
    # Find matches
    matches = matcher.match_trials(patient_profile, trial_texts, top_k=3)
    
    # Display results
    logger.info("Demo Clinical Trial Matching Results:")
    logger.info("=" * 50)
    logger.info(f"Patient Profile: {patient_profile}")
    logger.info("\nTop Trial Matches:")
    
    for i, (trial_text, score) in enumerate(matches, 1):
        logger.info(f"{i}. Score: {score:.3f}")
        logger.info(f"   Trial: {trial_text[:100]}...")
        logger.info("")


def main():
    """Main evaluation function."""
    parser = argparse.ArgumentParser(description='Evaluate clinical trial matching model')
    parser.add_argument('--config', type=str, default='configs/default.yaml',
                       help='Path to configuration file')
    parser.add_argument('--model_path', type=str, default=None,
                       help='Path to trained model checkpoint')
    parser.add_argument('--output_dir', type=str, default='outputs',
                       help='Output directory for results')
    parser.add_argument('--demo', action='store_true',
                       help='Run demo matching')
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
    
    log_file = output_dir / 'evaluation.log'
    setup_logging(config['logging']['level'], str(log_file))
    
    logger.info("Starting clinical trial matching evaluation")
    logger.info(f"Configuration: {args.config}")
    logger.info(f"Output directory: {args.output_dir}")
    
    if args.demo:
        # Run demo
        run_demo_matching(config)
    
    if args.model_path:
        # Evaluate model
        logger.info(f"Evaluating model: {args.model_path}")
        
        metrics = evaluate_model(args.model_path, config)
        
        # Generate evaluation report
        evaluator = ModelEvaluator(None, get_device())  # Dummy evaluator for report generation
        report = evaluator.generate_evaluation_report(metrics)
        
        logger.info("Evaluation Results:")
        logger.info(report)
        
        # Save results
        results_path = output_dir / 'evaluation_results.json'
        with open(results_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        
        report_path = output_dir / 'evaluation_report.txt'
        with open(report_path, 'w') as f:
            f.write(report)
        
        logger.info(f"Results saved to: {output_dir}")
    
    logger.info("Evaluation completed")


if __name__ == '__main__':
    main()
