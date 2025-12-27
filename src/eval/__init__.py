"""Evaluation utilities for clinical trial matching."""

import torch
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
import logging
from pathlib import Path
import json
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, precision_recall_curve

logger = logging.getLogger(__name__)


class ModelEvaluator:
    """Comprehensive model evaluator."""
    
    def __init__(self, model: torch.nn.Module, device: torch.device):
        """Initialize evaluator.
        
        Args:
            model: Model to evaluate.
            device: Device to run evaluation on.
        """
        self.model = model
        self.device = device
        self.model.eval()
    
    def evaluate_retrieval_performance(
        self,
        patient_texts: List[str],
        trial_texts: List[str],
        labels: List[int],
        top_k_values: List[int] = [1, 3, 5, 10]
    ) -> Dict[str, float]:
        """Evaluate retrieval performance.
        
        Args:
            patient_texts: List of patient texts.
            trial_texts: List of trial texts.
            labels: Binary relevance labels.
            top_k_values: List of k values for top-k evaluation.
            
        Returns:
            Dictionary of evaluation metrics.
        """
        metrics = {}
        
        # Get similarity scores
        with torch.no_grad():
            if hasattr(self.model, 'dual_encoder'):
                patient_embeddings = self.model.dual_encoder.encode_patients(patient_texts)
                trial_embeddings = self.model.dual_encoder.encode_trials(trial_texts)
                similarities = self.model.dual_encoder.compute_similarity(patient_embeddings, trial_embeddings)
            else:
                similarities = self.model(patient_texts, trial_texts)
            
            similarities = similarities.cpu().numpy()
        
        # Compute ranking metrics
        metrics.update(self._compute_ranking_metrics(similarities, labels, top_k_values))
        
        # Compute classification metrics
        predictions = (similarities > 0.5).astype(int)
        metrics.update(self._compute_classification_metrics(predictions, labels))
        
        return metrics
    
    def _compute_ranking_metrics(
        self, 
        similarities: np.ndarray, 
        labels: List[int], 
        top_k_values: List[int]
    ) -> Dict[str, float]:
        """Compute ranking metrics.
        
        Args:
            similarities: Similarity scores.
            labels: Relevance labels.
            top_k_values: List of k values.
            
        Returns:
            Dictionary of ranking metrics.
        """
        metrics = {}
        
        # Sort by similarity scores
        sorted_indices = np.argsort(similarities)[::-1]
        sorted_labels = np.array(labels)[sorted_indices]
        
        # Compute MRR
        mrr = 0.0
        for i, label in enumerate(sorted_labels):
            if label == 1:
                mrr = 1.0 / (i + 1)
                break
        metrics['mrr'] = mrr
        
        # Compute Hit Rate at different k values
        for k in top_k_values:
            hit_rate = np.sum(sorted_labels[:k]) / min(k, len(sorted_labels))
            metrics[f'hit_rate_at_{k}'] = hit_rate
        
        # Compute NDCG at different k values
        for k in top_k_values:
            ndcg = self._compute_ndcg(sorted_labels[:k])
            metrics[f'ndcg_at_{k}'] = ndcg
        
        return metrics
    
    def _compute_ndcg(self, labels: np.ndarray) -> float:
        """Compute Normalized Discounted Cumulative Gain.
        
        Args:
            labels: Relevance labels.
            
        Returns:
            NDCG score.
        """
        # Compute DCG
        dcg = 0.0
        for i, label in enumerate(labels):
            if label == 1:
                dcg += 1.0 / np.log2(i + 2)
        
        # Compute IDCG
        num_relevant = np.sum(labels)
        idcg = 0.0
        for i in range(min(len(labels), int(num_relevant))):
            idcg += 1.0 / np.log2(i + 2)
        
        # Compute NDCG
        ndcg = dcg / idcg if idcg > 0 else 0.0
        return ndcg
    
    def _compute_classification_metrics(
        self, 
        predictions: np.ndarray, 
        labels: List[int]
    ) -> Dict[str, float]:
        """Compute classification metrics.
        
        Args:
            predictions: Predicted labels.
            labels: True labels.
            
        Returns:
            Dictionary of classification metrics.
        """
        from sklearn.metrics import (
            precision_score, recall_score, f1_score, 
            roc_auc_score, average_precision_score
        )
        
        metrics = {}
        
        # Basic classification metrics
        metrics['precision'] = precision_score(labels, predictions, zero_division=0)
        metrics['recall'] = recall_score(labels, predictions, zero_division=0)
        metrics['f1'] = f1_score(labels, predictions, zero_division=0)
        
        # Ranking metrics
        if len(np.unique(labels)) > 1:
            # Get similarity scores for AUC computation
            similarities = self._get_similarity_scores()
            metrics['auc_roc'] = roc_auc_score(labels, similarities)
            metrics['auc_pr'] = average_precision_score(labels, similarities)
        
        return metrics
    
    def _get_similarity_scores(self) -> np.ndarray:
        """Get similarity scores for current evaluation.
        
        Returns:
            Similarity scores.
        """
        # This would need to be implemented based on the specific model
        # For now, return dummy scores
        return np.random.random(100)  # Placeholder
    
    def generate_evaluation_report(
        self,
        metrics: Dict[str, float],
        save_path: Optional[str] = None
    ) -> str:
        """Generate comprehensive evaluation report.
        
        Args:
            metrics: Dictionary of evaluation metrics.
            save_path: Optional path to save report.
            
        Returns:
            Formatted evaluation report.
        """
        report = "Clinical Trial Matching - Evaluation Report\n"
        report += "=" * 50 + "\n\n"
        
        # Classification metrics
        report += "Classification Performance:\n"
        report += f"  Precision: {metrics.get('precision', 0.0):.3f}\n"
        report += f"  Recall:    {metrics.get('recall', 0.0):.3f}\n"
        report += f"  F1 Score:  {metrics.get('f1', 0.0):.3f}\n\n"
        
        # Ranking metrics
        report += "Ranking Performance:\n"
        report += f"  MRR:       {metrics.get('mrr', 0.0):.3f}\n"
        report += f"  AUC-ROC:   {metrics.get('auc_roc', 0.0):.3f}\n"
        report += f"  AUC-PR:    {metrics.get('auc_pr', 0.0):.3f}\n\n"
        
        # Top-k metrics
        report += "Top-K Performance:\n"
        for k in [1, 3, 5, 10]:
            hit_rate = metrics.get(f'hit_rate_at_{k}', 0.0)
            ndcg = metrics.get(f'ndcg_at_{k}', 0.0)
            report += f"  Top-{k}: Hit Rate = {hit_rate:.3f}, NDCG = {ndcg:.3f}\n"
        
        # Save report if path provided
        if save_path:
            with open(save_path, 'w') as f:
                f.write(report)
            logger.info(f"Evaluation report saved to {save_path}")
        
        return report
    
    def plot_evaluation_curves(
        self,
        similarities: np.ndarray,
        labels: List[int],
        save_dir: Optional[str] = None
    ):
        """Plot evaluation curves.
        
        Args:
            similarities: Similarity scores.
            labels: True labels.
            save_dir: Directory to save plots.
        """
        if save_dir:
            save_path = Path(save_dir)
            save_path.mkdir(parents=True, exist_ok=True)
        
        # ROC Curve
        if len(np.unique(labels)) > 1:
            fpr, tpr, _ = roc_curve(labels, similarities)
            auc_score = np.trapz(tpr, fpr)
            
            plt.figure(figsize=(8, 6))
            plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {auc_score:.3f})')
            plt.plot([0, 1], [0, 1], 'k--', label='Random')
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('ROC Curve')
            plt.legend()
            plt.grid(True)
            
            if save_dir:
                plt.savefig(save_path / 'roc_curve.png', dpi=300, bbox_inches='tight')
            plt.close()
        
        # Precision-Recall Curve
        precision, recall, _ = precision_recall_curve(labels, similarities)
        ap_score = np.trapz(precision, recall)
        
        plt.figure(figsize=(8, 6))
        plt.plot(recall, precision, label=f'PR Curve (AP = {ap_score:.3f})')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.legend()
        plt.grid(True)
        
        if save_dir:
            plt.savefig(save_path / 'pr_curve.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Confusion Matrix
        predictions = (similarities > 0.5).astype(int)
        cm = confusion_matrix(labels, predictions)
        
        plt.figure(figsize=(6, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title('Confusion Matrix')
        
        if save_dir:
            plt.savefig(save_path / 'confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info("Evaluation plots generated")


class BenchmarkEvaluator:
    """Benchmark evaluator for comparing different models."""
    
    def __init__(self):
        """Initialize benchmark evaluator."""
        self.results = {}
    
    def add_model_results(self, model_name: str, metrics: Dict[str, float]):
        """Add results for a model.
        
        Args:
            model_name: Name of the model.
            metrics: Evaluation metrics.
        """
        self.results[model_name] = metrics
    
    def generate_leaderboard(self) -> str:
        """Generate model comparison leaderboard.
        
        Returns:
            Formatted leaderboard.
        """
        if not self.results:
            return "No results available for comparison."
        
        leaderboard = "Clinical Trial Matching - Model Comparison\n"
        leaderboard += "=" * 60 + "\n\n"
        
        # Sort models by F1 score
        sorted_models = sorted(
            self.results.items(), 
            key=lambda x: x[1].get('f1', 0.0), 
            reverse=True
        )
        
        # Header
        leaderboard += f"{'Model':<20} {'F1':<8} {'Precision':<10} {'Recall':<8} {'MRR':<8} {'AUC-ROC':<8}\n"
        leaderboard += "-" * 60 + "\n"
        
        # Results
        for model_name, metrics in sorted_models:
            f1 = metrics.get('f1', 0.0)
            precision = metrics.get('precision', 0.0)
            recall = metrics.get('recall', 0.0)
            mrr = metrics.get('mrr', 0.0)
            auc_roc = metrics.get('auc_roc', 0.0)
            
            leaderboard += f"{model_name:<20} {f1:<8.3f} {precision:<10.3f} {recall:<8.3f} {mrr:<8.3f} {auc_roc:<8.3f}\n"
        
        return leaderboard
    
    def save_results(self, filepath: str):
        """Save results to JSON file.
        
        Args:
            filepath: Path to save results.
        """
        with open(filepath, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        logger.info(f"Results saved to {filepath}")
    
    def load_results(self, filepath: str):
        """Load results from JSON file.
        
        Args:
            filepath: Path to load results from.
        """
        with open(filepath, 'r') as f:
            self.results = json.load(f)
        
        logger.info(f"Results loaded from {filepath}")
