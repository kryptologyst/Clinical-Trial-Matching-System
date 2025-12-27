"""Evaluation metrics for clinical trial matching."""

import torch
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from sklearn.metrics import (
    precision_recall_curve, 
    roc_auc_score, 
    average_precision_score,
    precision_score,
    recall_score,
    f1_score
)
import logging

logger = logging.getLogger(__name__)


class RetrievalMetrics:
    """Metrics for retrieval-based clinical trial matching."""
    
    def __init__(self):
        """Initialize retrieval metrics."""
        self.reset()
    
    def reset(self):
        """Reset all metrics."""
        self.predictions = []
        self.labels = []
        self.similarity_scores = []
    
    def update(self, predictions: torch.Tensor, labels: torch.Tensor, scores: torch.Tensor):
        """Update metrics with new batch.
        
        Args:
            predictions: Predicted labels.
            labels: True labels.
            scores: Similarity scores.
        """
        self.predictions.extend(predictions.cpu().numpy())
        self.labels.extend(labels.cpu().numpy())
        self.similarity_scores.extend(scores.cpu().numpy())
    
    def compute_metrics(self) -> Dict[str, float]:
        """Compute all metrics.
        
        Returns:
            Dictionary of metric names and values.
        """
        if not self.predictions:
            return {}
        
        predictions = np.array(self.predictions)
        labels = np.array(self.labels)
        scores = np.array(self.similarity_scores)
        
        metrics = {}
        
        # Classification metrics
        metrics['precision'] = precision_score(labels, predictions, zero_division=0)
        metrics['recall'] = recall_score(labels, predictions, zero_division=0)
        metrics['f1'] = f1_score(labels, predictions, zero_division=0)
        
        # Ranking metrics
        metrics['auc_roc'] = roc_auc_score(labels, scores)
        metrics['auc_pr'] = average_precision_score(labels, scores)
        
        # Retrieval-specific metrics
        metrics.update(self._compute_retrieval_metrics(labels, scores))
        
        return metrics
    
    def _compute_retrieval_metrics(self, labels: np.ndarray, scores: np.ndarray) -> Dict[str, float]:
        """Compute retrieval-specific metrics.
        
        Args:
            labels: True labels.
            scores: Similarity scores.
            
        Returns:
            Dictionary of retrieval metrics.
        """
        metrics = {}
        
        # Precision at different thresholds
        thresholds = [0.1, 0.3, 0.5, 0.7, 0.9]
        for threshold in thresholds:
            pred_at_thresh = (scores >= threshold).astype(int)
            precision = precision_score(labels, pred_at_thresh, zero_division=0)
            metrics[f'precision_at_{threshold}'] = precision
        
        # Recall at different thresholds
        for threshold in thresholds:
            pred_at_thresh = (scores >= threshold).astype(int)
            recall = recall_score(labels, pred_at_thresh, zero_division=0)
            metrics[f'recall_at_{threshold}'] = recall
        
        return metrics


class RankingMetrics:
    """Metrics for ranking-based evaluation."""
    
    def __init__(self, k_values: List[int] = [1, 3, 5, 10]):
        """Initialize ranking metrics.
        
        Args:
            k_values: List of k values for top-k metrics.
        """
        self.k_values = k_values
        self.reset()
    
    def reset(self):
        """Reset all metrics."""
        self.rankings = []
        self.relevance_labels = []
    
    def update(self, rankings: List[List[int]], relevance_labels: List[List[int]]):
        """Update metrics with new rankings.
        
        Args:
            rankings: List of ranked item indices.
            relevance_labels: List of relevance labels for each ranking.
        """
        self.rankings.extend(rankings)
        self.relevance_labels.extend(relevance_labels)
    
    def compute_metrics(self) -> Dict[str, float]:
        """Compute ranking metrics.
        
        Returns:
            Dictionary of metric names and values.
        """
        if not self.rankings:
            return {}
        
        metrics = {}
        
        # Mean Reciprocal Rank (MRR)
        mrr_scores = []
        for ranking, labels in zip(self.rankings, self.relevance_labels):
            mrr = self._compute_mrr(ranking, labels)
            mrr_scores.append(mrr)
        metrics['mrr'] = np.mean(mrr_scores)
        
        # Hit Rate at different k values
        for k in self.k_values:
            hit_rate = self._compute_hit_rate(k)
            metrics[f'hit_rate_at_{k}'] = hit_rate
        
        # Normalized Discounted Cumulative Gain (NDCG)
        for k in self.k_values:
            ndcg = self._compute_ndcg(k)
            metrics[f'ndcg_at_{k}'] = ndcg
        
        return metrics
    
    def _compute_mrr(self, ranking: List[int], labels: List[int]) -> float:
        """Compute Mean Reciprocal Rank.
        
        Args:
            ranking: Ranked list of item indices.
            labels: Relevance labels.
            
        Returns:
            MRR score.
        """
        for i, item_idx in enumerate(ranking):
            if labels[item_idx] == 1:
                return 1.0 / (i + 1)
        return 0.0
    
    def _compute_hit_rate(self, k: int) -> float:
        """Compute Hit Rate at k.
        
        Args:
            k: Number of top items to consider.
            
        Returns:
            Hit rate at k.
        """
        hits = 0
        total = len(self.rankings)
        
        for ranking, labels in zip(self.rankings, self.relevance_labels):
            top_k_items = ranking[:k]
            if any(labels[item_idx] == 1 for item_idx in top_k_items):
                hits += 1
        
        return hits / total if total > 0 else 0.0
    
    def _compute_ndcg(self, k: int) -> float:
        """Compute Normalized Discounted Cumulative Gain at k.
        
        Args:
            k: Number of top items to consider.
            
        Returns:
            NDCG at k.
        """
        ndcg_scores = []
        
        for ranking, labels in zip(self.rankings, self.relevance_labels):
            # Compute DCG
            dcg = 0.0
            for i, item_idx in enumerate(ranking[:k]):
                if labels[item_idx] == 1:
                    dcg += 1.0 / np.log2(i + 2)
            
            # Compute IDCG (ideal DCG)
            idcg = 0.0
            num_relevant = sum(labels)
            for i in range(min(k, num_relevant)):
                idcg += 1.0 / np.log2(i + 2)
            
            # Compute NDCG
            ndcg = dcg / idcg if idcg > 0 else 0.0
            ndcg_scores.append(ndcg)
        
        return np.mean(ndcg_scores)


class ClinicalTrialEvaluator:
    """Comprehensive evaluator for clinical trial matching."""
    
    def __init__(self):
        """Initialize evaluator."""
        self.retrieval_metrics = RetrievalMetrics()
        self.ranking_metrics = RankingMetrics()
    
    def evaluate_batch(
        self,
        predictions: torch.Tensor,
        labels: torch.Tensor,
        scores: torch.Tensor,
        rankings: Optional[List[List[int]]] = None,
        relevance_labels: Optional[List[List[int]]] = None
    ) -> Dict[str, float]:
        """Evaluate a batch of predictions.
        
        Args:
            predictions: Predicted labels.
            labels: True labels.
            scores: Similarity scores.
            rankings: Optional rankings for each query.
            relevance_labels: Optional relevance labels for rankings.
            
        Returns:
            Dictionary of computed metrics.
        """
        # Update retrieval metrics
        self.retrieval_metrics.update(predictions, labels, scores)
        
        # Update ranking metrics if provided
        if rankings is not None and relevance_labels is not None:
            self.ranking_metrics.update(rankings, relevance_labels)
        
        # Compute metrics
        metrics = {}
        metrics.update(self.retrieval_metrics.compute_metrics())
        metrics.update(self.ranking_metrics.compute_metrics())
        
        return metrics
    
    def evaluate_model(
        self,
        model: torch.nn.Module,
        dataloader: torch.utils.data.DataLoader,
        device: torch.device
    ) -> Dict[str, float]:
        """Evaluate model on entire dataset.
        
        Args:
            model: Model to evaluate.
            dataloader: Data loader for evaluation.
            device: Device to run evaluation on.
            
        Returns:
            Dictionary of evaluation metrics.
        """
        model.eval()
        
        all_predictions = []
        all_labels = []
        all_scores = []
        
        with torch.no_grad():
            for batch in dataloader:
                # Move batch to device
                batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                        for k, v in batch.items()}
                
                # Get model outputs
                outputs = model(**batch)
                scores = outputs['scores']
                predictions = (scores > 0.5).float()
                
                all_predictions.append(predictions)
                all_labels.append(batch['labels'])
                all_scores.append(scores)
        
        # Concatenate all results
        all_predictions = torch.cat(all_predictions)
        all_labels = torch.cat(all_labels)
        all_scores = torch.cat(all_scores)
        
        # Compute final metrics
        metrics = self.evaluate_batch(all_predictions, all_labels, all_scores)
        
        return metrics
    
    def reset(self):
        """Reset all metrics."""
        self.retrieval_metrics.reset()
        self.ranking_metrics.reset()
    
    def get_leaderboard(self, metrics: Dict[str, float]) -> str:
        """Format metrics as a leaderboard.
        
        Args:
            metrics: Dictionary of metrics.
            
        Returns:
            Formatted leaderboard string.
        """
        leaderboard = "Clinical Trial Matching - Evaluation Results\n"
        leaderboard += "=" * 50 + "\n\n"
        
        # Classification metrics
        leaderboard += "Classification Metrics:\n"
        leaderboard += f"  Precision: {metrics.get('precision', 0.0):.3f}\n"
        leaderboard += f"  Recall:    {metrics.get('recall', 0.0):.3f}\n"
        leaderboard += f"  F1 Score:  {metrics.get('f1', 0.0):.3f}\n\n"
        
        # Ranking metrics
        leaderboard += "Ranking Metrics:\n"
        leaderboard += f"  AUC-ROC:   {metrics.get('auc_roc', 0.0):.3f}\n"
        leaderboard += f"  AUC-PR:    {metrics.get('auc_pr', 0.0):.3f}\n"
        leaderboard += f"  MRR:       {metrics.get('mrr', 0.0):.3f}\n\n"
        
        # Top-k metrics
        leaderboard += "Top-K Hit Rates:\n"
        for k in [1, 3, 5, 10]:
            hit_rate = metrics.get(f'hit_rate_at_{k}', 0.0)
            leaderboard += f"  Hit@{k}:   {hit_rate:.3f}\n"
        
        return leaderboard
