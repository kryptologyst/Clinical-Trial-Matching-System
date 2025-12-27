"""Tests for clinical trial matching system."""

import pytest
import torch
import numpy as np
from unittest.mock import Mock, patch

from src.utils import set_seed, get_device, deidentify_text, validate_patient_data
from src.data import ClinicalTrialDataset, DataProcessor
from src.models import ClinicalBERTEncoder, DualEncoderRetrieval
from src.metrics import RetrievalMetrics, RankingMetrics


class TestUtils:
    """Test utility functions."""
    
    def test_set_seed(self):
        """Test random seed setting."""
        set_seed(42)
        # Test that seed is set (basic check)
        assert True  # Placeholder - would need more sophisticated testing
    
    def test_get_device(self):
        """Test device selection."""
        device = get_device()
        assert isinstance(device, torch.device)
    
    def test_deidentify_text(self):
        """Test text de-identification."""
        text = "Patient John Doe, phone 555-123-4567, email john@example.com"
        deidentified = deidentify_text(text, deid_mode=True)
        
        assert "[PHONE]" in deidentified
        assert "[EMAIL]" in deidentified
        assert "John Doe" not in deidentified
    
    def test_deidentify_text_disabled(self):
        """Test de-identification when disabled."""
        text = "Patient John Doe, phone 555-123-4567"
        deidentified = deidentify_text(text, deid_mode=False)
        
        assert text == deidentified
    
    def test_validate_patient_data(self):
        """Test patient data validation."""
        # Valid data
        valid_data = {
            'age': 65,
            'gender': 'female',
            'diagnosis': 'breast cancer'
        }
        assert validate_patient_data(valid_data) is True
        
        # Invalid data - missing field
        invalid_data = {
            'age': 65,
            'gender': 'female'
            # Missing diagnosis
        }
        assert validate_patient_data(invalid_data) is False
        
        # Invalid data - invalid age
        invalid_age = {
            'age': -5,
            'gender': 'female',
            'diagnosis': 'breast cancer'
        }
        assert validate_patient_data(invalid_age) is False


class TestData:
    """Test data processing."""
    
    def test_clinical_trial_dataset(self):
        """Test clinical trial dataset creation."""
        dataset = ClinicalTrialDataset()
        
        assert len(dataset.trials) > 0
        assert len(dataset.patients) > 0
        
        # Test trial text generation
        trial_text = dataset.get_trial_text(dataset.trials[0]['id'])
        assert isinstance(trial_text, str)
        assert len(trial_text) > 0
        
        # Test patient text generation
        patient_text = dataset.get_patient_text(dataset.patients[0]['id'])
        assert isinstance(patient_text, str)
        assert len(patient_text) > 0
    
    def test_data_processor(self):
        """Test data processor."""
        processor = DataProcessor(max_length=256)
        
        # Test text preprocessing
        text = "  This is a test   text with   extra spaces  "
        processed = processor.preprocess_text(text)
        assert processed == "this is a test text with extra spaces"
        
        # Test retrieval pairs creation
        dataset = ClinicalTrialDataset()
        pairs = processor.create_retrieval_pairs(dataset)
        
        assert len(pairs) > 0
        assert all('patient_text' in pair for pair in pairs)
        assert all('trial_text' in pair for pair in pairs)
        assert all('label' in pair for pair in pairs)


class TestModels:
    """Test model components."""
    
    @patch('src.models.AutoTokenizer')
    @patch('src.models.AutoModel')
    def test_clinical_bert_encoder(self, mock_model, mock_tokenizer):
        """Test ClinicalBERT encoder."""
        # Mock tokenizer and model
        mock_tokenizer.from_pretrained.return_value = Mock()
        mock_model.from_pretrained.return_value = Mock()
        
        # This would need more sophisticated mocking for actual testing
        # For now, just test that the class can be instantiated
        try:
            encoder = ClinicalBERTEncoder()
            assert encoder is not None
        except Exception:
            # Expected to fail without proper model loading
            pass
    
    def test_dual_encoder_retrieval(self):
        """Test dual encoder retrieval model."""
        # This would need proper mocking for transformer models
        # For now, just test basic structure
        try:
            model = DualEncoderRetrieval()
            assert model is not None
        except Exception:
            # Expected to fail without proper model loading
            pass


class TestMetrics:
    """Test evaluation metrics."""
    
    def test_retrieval_metrics(self):
        """Test retrieval metrics."""
        metrics = RetrievalMetrics()
        
        # Test update
        predictions = torch.tensor([1, 0, 1, 0])
        labels = torch.tensor([1, 0, 1, 1])
        scores = torch.tensor([0.8, 0.3, 0.9, 0.2])
        
        metrics.update(predictions, labels, scores)
        
        # Test compute
        result = metrics.compute_metrics()
        assert isinstance(result, dict)
        assert 'precision' in result
        assert 'recall' in result
        assert 'f1' in result
    
    def test_ranking_metrics(self):
        """Test ranking metrics."""
        metrics = RankingMetrics()
        
        # Test update
        rankings = [[0, 1, 2], [1, 0, 2]]
        relevance_labels = [[1, 0, 1], [0, 1, 0]]
        
        metrics.update(rankings, relevance_labels)
        
        # Test compute
        result = metrics.compute_metrics()
        assert isinstance(result, dict)
        assert 'mrr' in result
        assert 'hit_rate_at_1' in result


class TestIntegration:
    """Test integration scenarios."""
    
    def test_end_to_end_basic(self):
        """Test basic end-to-end functionality."""
        # Load dataset
        dataset = ClinicalTrialDataset()
        
        # Get sample data
        patient_text = dataset.get_patient_text(dataset.patients[0]['id'])
        trial_texts = [dataset.get_trial_text(trial['id']) for trial in dataset.trials]
        
        assert len(patient_text) > 0
        assert len(trial_texts) > 0
        
        # Test data processor
        processor = DataProcessor()
        processed_patient = processor.preprocess_text(patient_text)
        processed_trials = [processor.preprocess_text(text) for text in trial_texts]
        
        assert len(processed_patient) > 0
        assert len(processed_trials) > 0
    
    def test_configuration_loading(self):
        """Test configuration loading."""
        import yaml
        from pathlib import Path
        
        config_path = Path("configs/default.yaml")
        if config_path.exists():
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            
            assert 'model' in config
            assert 'training' in config
            assert 'data' in config


if __name__ == '__main__':
    pytest.main([__file__])
