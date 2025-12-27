"""Advanced NLP models for clinical trial matching."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import (
    AutoTokenizer, 
    AutoModel, 
    AutoConfig,
    BertModel,
    BertTokenizer
)
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import logging

logger = logging.getLogger(__name__)


class ClinicalBERTEncoder(nn.Module):
    """Clinical BERT encoder for text representation."""
    
    def __init__(
        self, 
        model_name: str = "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract",
        max_length: int = 512,
        dropout: float = 0.1
    ):
        """Initialize Clinical BERT encoder.
        
        Args:
            model_name: Pre-trained model name.
            max_length: Maximum sequence length.
            dropout: Dropout rate.
        """
        super().__init__()
        
        self.model_name = model_name
        self.max_length = max_length
        
        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.config = AutoConfig.from_pretrained(model_name)
        self.bert = AutoModel.from_pretrained(model_name)
        
        # Add dropout for regularization
        self.dropout = nn.Dropout(dropout)
        
        # Projection layer for embedding dimension
        self.projection = nn.Linear(self.config.hidden_size, 768)
        
        logger.info(f"Initialized ClinicalBERT encoder: {model_name}")
    
    def encode_text(self, texts: List[str]) -> torch.Tensor:
        """Encode texts to embeddings.
        
        Args:
            texts: List of input texts.
            
        Returns:
            Text embeddings tensor.
        """
        # Tokenize texts
        encoded = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        # Move to device
        device = next(self.parameters()).device
        encoded = {k: v.to(device) for k, v in encoded.items()}
        
        # Get BERT outputs
        with torch.no_grad():
            outputs = self.bert(**encoded)
        
        # Use [CLS] token representation
        embeddings = outputs.last_hidden_state[:, 0, :]  # [batch_size, hidden_size]
        
        # Apply projection and dropout
        embeddings = self.projection(embeddings)
        embeddings = self.dropout(embeddings)
        
        # Normalize embeddings
        embeddings = F.normalize(embeddings, p=2, dim=1)
        
        return embeddings
    
    def forward(self, texts: List[str]) -> torch.Tensor:
        """Forward pass.
        
        Args:
            texts: List of input texts.
            
        Returns:
            Text embeddings.
        """
        return self.encode_text(texts)


class DualEncoderRetrieval(nn.Module):
    """Dual encoder model for clinical trial retrieval."""
    
    def __init__(
        self,
        model_name: str = "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract",
        max_length: int = 512,
        dropout: float = 0.1
    ):
        """Initialize dual encoder model.
        
        Args:
            model_name: Pre-trained model name.
            max_length: Maximum sequence length.
            dropout: Dropout rate.
        """
        super().__init__()
        
        # Shared encoder for both patients and trials
        self.encoder = ClinicalBERTEncoder(model_name, max_length, dropout)
        
        # Separate projection layers for patients and trials
        self.patient_projection = nn.Linear(768, 256)
        self.trial_projection = nn.Linear(768, 256)
        
        logger.info("Initialized dual encoder retrieval model")
    
    def encode_patients(self, patient_texts: List[str]) -> torch.Tensor:
        """Encode patient texts.
        
        Args:
            patient_texts: List of patient text descriptions.
            
        Returns:
            Patient embeddings.
        """
        embeddings = self.encoder(patient_texts)
        embeddings = self.patient_projection(embeddings)
        return F.normalize(embeddings, p=2, dim=1)
    
    def encode_trials(self, trial_texts: List[str]) -> torch.Tensor:
        """Encode trial texts.
        
        Args:
            trial_texts: List of trial text descriptions.
            
        Returns:
            Trial embeddings.
        """
        embeddings = self.encoder(trial_texts)
        embeddings = self.trial_projection(embeddings)
        return F.normalize(embeddings, p=2, dim=1)
    
    def compute_similarity(self, patient_embeddings: torch.Tensor, trial_embeddings: torch.Tensor) -> torch.Tensor:
        """Compute similarity scores between patients and trials.
        
        Args:
            patient_embeddings: Patient embeddings.
            trial_embeddings: Trial embeddings.
            
        Returns:
            Similarity scores.
        """
        # Compute cosine similarity
        similarity = torch.mm(patient_embeddings, trial_embeddings.t())
        return similarity
    
    def forward(self, patient_texts: List[str], trial_texts: List[str]) -> torch.Tensor:
        """Forward pass.
        
        Args:
            patient_texts: List of patient texts.
            trial_texts: List of trial texts.
            
        Returns:
            Similarity scores.
        """
        patient_embeddings = self.encode_patients(patient_texts)
        trial_embeddings = self.encode_trials(trial_texts)
        
        return self.compute_similarity(patient_embeddings, trial_embeddings)


class CrossEncoderRanking(nn.Module):
    """Cross-encoder model for fine-grained ranking."""
    
    def __init__(
        self,
        model_name: str = "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract",
        max_length: int = 512,
        dropout: float = 0.1
    ):
        """Initialize cross-encoder model.
        
        Args:
            model_name: Pre-trained model name.
            max_length: Maximum sequence length.
            dropout: Dropout rate.
        """
        super().__init__()
        
        self.model_name = model_name
        self.max_length = max_length
        
        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.config = AutoConfig.from_pretrained(model_name)
        self.bert = AutoModel.from_pretrained(model_name)
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(self.config.hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 1)
        )
        
        logger.info("Initialized cross-encoder ranking model")
    
    def encode_pair(self, patient_text: str, trial_text: str) -> torch.Tensor:
        """Encode patient-trial pair.
        
        Args:
            patient_text: Patient text description.
            trial_text: Trial text description.
            
        Returns:
            Pair embedding.
        """
        # Combine texts with separator
        combined_text = f"{patient_text} [SEP] {trial_text}"
        
        # Tokenize
        encoded = self.tokenizer(
            combined_text,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        # Move to device
        device = next(self.parameters()).device
        encoded = {k: v.to(device) for k, v in encoded.items()}
        
        # Get BERT outputs
        outputs = self.bert(**encoded)
        
        # Use [CLS] token representation
        embedding = outputs.last_hidden_state[:, 0, :]  # [batch_size, hidden_size]
        
        return embedding
    
    def forward(self, patient_texts: List[str], trial_texts: List[str]) -> torch.Tensor:
        """Forward pass.
        
        Args:
            patient_texts: List of patient texts.
            trial_texts: List of trial texts.
            
        Returns:
            Relevance scores.
        """
        batch_size = len(patient_texts)
        scores = []
        
        for i in range(batch_size):
            embedding = self.encode_pair(patient_texts[i], trial_texts[i])
            score = self.classifier(embedding)
            scores.append(score)
        
        return torch.cat(scores, dim=0)


class HybridRetrievalModel(nn.Module):
    """Hybrid model combining dual encoder and cross-encoder."""
    
    def __init__(
        self,
        model_name: str = "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract",
        max_length: int = 512,
        dropout: float = 0.1
    ):
        """Initialize hybrid retrieval model.
        
        Args:
            model_name: Pre-trained model name.
            max_length: Maximum sequence length.
            dropout: Dropout rate.
        """
        super().__init__()
        
        self.dual_encoder = DualEncoderRetrieval(model_name, max_length, dropout)
        self.cross_encoder = CrossEncoderRanking(model_name, max_length, dropout)
        
        # Fusion layer
        self.fusion = nn.Linear(2, 1)
        
        logger.info("Initialized hybrid retrieval model")
    
    def forward(
        self, 
        patient_texts: List[str], 
        trial_texts: List[str],
        use_cross_encoder: bool = True
    ) -> torch.Tensor:
        """Forward pass.
        
        Args:
            patient_texts: List of patient texts.
            trial_texts: List of trial texts.
            use_cross_encoder: Whether to use cross-encoder refinement.
            
        Returns:
            Final similarity scores.
        """
        # Get dual encoder scores
        dual_scores = self.dual_encoder(patient_texts, trial_texts)
        
        if not use_cross_encoder:
            return dual_scores
        
        # Get cross-encoder scores for refinement
        cross_scores = []
        for i, patient_text in enumerate(patient_texts):
            batch_cross_scores = []
            for trial_text in trial_texts:
                score = self.cross_encoder([patient_text], [trial_text])
                batch_cross_scores.append(score)
            cross_scores.append(torch.cat(batch_cross_scores))
        
        cross_scores = torch.stack(cross_scores)
        
        # Fuse scores
        combined_scores = torch.stack([dual_scores, cross_scores], dim=-1)
        final_scores = self.fusion(combined_scores).squeeze(-1)
        
        return final_scores


class ClinicalTrialMatcher:
    """Main clinical trial matching system."""
    
    def __init__(
        self,
        model_name: str = "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract",
        device: Optional[torch.device] = None
    ):
        """Initialize clinical trial matcher.
        
        Args:
            model_name: Pre-trained model name.
            device: Device to run model on.
        """
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_name = model_name
        
        # Initialize model
        self.model = HybridRetrievalModel(model_name).to(self.device)
        
        logger.info(f"Initialized ClinicalTrialMatcher on {self.device}")
    
    def match_trials(
        self, 
        patient_text: str, 
        trial_texts: List[str],
        top_k: int = 5
    ) -> List[Tuple[str, float]]:
        """Match patient to top-k trials.
        
        Args:
            patient_text: Patient description.
            trial_texts: List of trial descriptions.
            top_k: Number of top matches to return.
            
        Returns:
            List of (trial_text, similarity_score) tuples.
        """
        self.model.eval()
        
        with torch.no_grad():
            # Get similarity scores
            scores = self.model([patient_text], trial_texts)
            scores = scores.squeeze(0)  # Remove batch dimension
            
            # Get top-k indices
            top_indices = torch.topk(scores, min(top_k, len(trial_texts))).indices
            
            # Return results
            results = []
            for idx in top_indices:
                trial_text = trial_texts[idx.item()]
                score = scores[idx].item()
                results.append((trial_text, score))
            
            return results
    
    def get_embeddings(self, texts: List[str]) -> torch.Tensor:
        """Get embeddings for texts.
        
        Args:
            texts: List of input texts.
            
        Returns:
            Text embeddings.
        """
        self.model.eval()
        
        with torch.no_grad():
            # Use dual encoder for embeddings
            embeddings = self.model.dual_encoder.encode_patients(texts)
            return embeddings
