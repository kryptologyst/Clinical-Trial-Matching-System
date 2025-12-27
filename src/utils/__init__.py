"""Utility functions for clinical trial matching system."""

import random
import numpy as np
import torch
from typing import Any, Dict, List, Optional, Tuple, Union
import logging
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

logger = logging.getLogger(__name__)


def set_seed(seed: int = 42) -> None:
    """Set random seeds for reproducibility.
    
    Args:
        seed: Random seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # For deterministic behavior
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    logger.info(f"Random seed set to {seed}")


def get_device() -> torch.device:
    """Get the best available device (CUDA -> MPS -> CPU).
    
    Returns:
        torch.device: The selected device.
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
        logger.info(f"Using CUDA device: {torch.cuda.get_device_name()}")
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = torch.device("mps")
        logger.info("Using Apple Silicon MPS device")
    else:
        device = torch.device("cpu")
        logger.info("Using CPU device")
    
    return device


def setup_logging(level: str = "INFO", log_file: Optional[str] = None) -> None:
    """Setup logging configuration.
    
    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR).
        log_file: Optional log file path.
    """
    log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    handlers = [logging.StreamHandler()]
    if log_file:
        handlers.append(logging.FileHandler(log_file))
    
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format=log_format,
        handlers=handlers
    )


def deidentify_text(text: str, deid_mode: bool = True) -> str:
    """Basic de-identification for demonstration purposes.
    
    WARNING: This is a simple regex-based approach for demo only.
    For production use, consider using Presidio or similar tools.
    
    Args:
        text: Input text to de-identify.
        deid_mode: Whether to apply de-identification.
        
    Returns:
        De-identified text.
    """
    if not deid_mode:
        return text
    
    import re
    
    # Simple patterns for demonstration
    patterns = [
        (r'\b\d{1,3}-\d{1,3}-\d{1,3}-\d{1,3}\b', '[PHONE]'),  # Phone numbers
        (r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '[EMAIL]'),  # Email
        (r'\b\d{3}-\d{2}-\d{4}\b', '[SSN]'),  # SSN
        (r'\b\d{1,2}/\d{1,2}/\d{4}\b', '[DATE]'),  # Dates
    ]
    
    deidentified_text = text
    for pattern, replacement in patterns:
        deidentified_text = re.sub(pattern, replacement, deidentified_text)
    
    return deidentified_text


def validate_patient_data(patient_data: Dict[str, Any]) -> bool:
    """Validate patient data structure.
    
    Args:
        patient_data: Dictionary containing patient information.
        
    Returns:
        True if valid, False otherwise.
    """
    required_fields = ['age', 'gender', 'diagnosis']
    
    for field in required_fields:
        if field not in patient_data:
            logger.warning(f"Missing required field: {field}")
            return False
    
    # Additional validation
    if not isinstance(patient_data['age'], (int, float)) or patient_data['age'] < 0:
        logger.warning("Invalid age value")
        return False
    
    if patient_data['gender'].lower() not in ['male', 'female', 'other']:
        logger.warning("Invalid gender value")
        return False
    
    return True


def format_similarity_score(score: float) -> str:
    """Format similarity score for display.
    
    Args:
        score: Similarity score between 0 and 1.
        
    Returns:
        Formatted score string.
    """
    return f"{score:.3f}"


def create_safety_banner() -> str:
    """Create safety disclaimer banner.
    
    Returns:
        Safety disclaimer text.
    """
    return """
    ⚠️  RESEARCH DEMONSTRATION ONLY ⚠️
    
    This system is for educational and research purposes only.
    NOT FOR CLINICAL USE. NOT MEDICAL ADVICE.
    
    Always consult qualified healthcare professionals for medical decisions.
    """


class Config:
    """Configuration class for the clinical trial matching system."""
    
    def __init__(self, config_dict: Optional[Dict[str, Any]] = None):
        """Initialize configuration.
        
        Args:
            config_dict: Optional configuration dictionary.
        """
        # Default configuration
        self.seed = 42
        self.device = get_device()
        self.batch_size = 32
        self.max_length = 512
        self.learning_rate = 2e-5
        self.num_epochs = 3
        self.model_name = "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract"
        self.deid_mode = True
        self.similarity_threshold = 0.3
        
        # Update with provided config
        if config_dict:
            for key, value in config_dict.items():
                if hasattr(self, key):
                    setattr(self, key, value)
                else:
                    logger.warning(f"Unknown config key: {key}")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary.
        
        Returns:
            Configuration as dictionary.
        """
        return {key: value for key, value in self.__dict__.items() 
                if not key.startswith('_')}
