# Clinical Trial Matching System

A research demonstration system that uses advanced NLP techniques to match patients with relevant clinical trials based on text similarity and semantic understanding.

## ⚠️ IMPORTANT DISCLAIMER

**THIS IS A RESEARCH DEMONSTRATION PROJECT ONLY**

- **NOT FOR CLINICAL USE**
- **NOT MEDICAL ADVICE**
- **NOT INTENDED FOR DIAGNOSTIC PURPOSES**

This system is designed for educational and research purposes only. Always consult qualified healthcare professionals for medical decisions and clinical trial enrollment.

## Overview

The Clinical Trial Matching System automatically matches patients to relevant clinical trials using:

- **Dual-Encoder Architecture**: Separate encoders for patients and trials
- **Cross-Encoder Refinement**: Fine-grained ranking for improved accuracy
- **Clinical BERT**: Pre-trained on biomedical literature
- **Comprehensive Evaluation**: Multiple metrics for thorough assessment
- **Privacy Protection**: Built-in de-identification capabilities

## Features

### Core Functionality
- Patient-trial matching based on semantic similarity
- Support for multiple pre-trained clinical language models
- Configurable similarity thresholds
- Top-k result ranking
- Comprehensive evaluation metrics

### Advanced Capabilities
- Hybrid retrieval combining dual-encoder and cross-encoder approaches
- De-identification for privacy protection
- Interactive web interface
- Model performance visualization
- Synthetic dataset generation for demonstration

### Evaluation Metrics
- **Classification**: Precision, Recall, F1-Score
- **Ranking**: Mean Reciprocal Rank (MRR), Hit Rate@K
- **Information Retrieval**: NDCG, AUC-ROC, AUC-PR
- **Clinical Relevance**: Condition overlap analysis

## Installation

### Prerequisites
- Python 3.10+
- PyTorch 2.0+
- CUDA/MPS support (optional but recommended)

### Setup

1. **Clone the repository**:
```bash
git clone https://github.com/kryptologyst/Clinical-Trial-Matching-System.git
cd Clinical-Trial-Matching-System
```

2. **Install dependencies**:
```bash
pip install -r requirements.txt
```

3. **Verify installation**:
```bash
python -c "import torch; print(f'PyTorch version: {torch.__version__}')"
```

## Quick Start

### 1. Run the Interactive Demo

```bash
streamlit run demo/app.py
```

This launches a web interface where you can:
- Enter patient descriptions
- Select trials to match against
- View similarity scores and detailed trial information
- Explore the dataset and model performance

### 2. Train a Model

```bash
python scripts/train.py --config configs/default.yaml --output_dir outputs/training
```

### 3. Evaluate a Model

```bash
python scripts/evaluate.py --model_path outputs/training/checkpoints/best_model.pt --demo
```

## Usage

### Basic Usage

```python
from src.models import ClinicalTrialMatcher
from src.data import ClinicalTrialDataset

# Initialize matcher
matcher = ClinicalTrialMatcher()

# Load dataset
dataset = ClinicalTrialDataset()

# Patient description
patient_text = "65-year-old female with stage II HER2-positive breast cancer"

# Get trial texts
trial_texts = [dataset.get_trial_text(trial['id']) for trial in dataset.trials]

# Find matches
matches = matcher.match_trials(patient_text, trial_texts, top_k=5)

# Display results
for trial_text, score in matches:
    print(f"Score: {score:.3f} - {trial_text[:100]}...")
```

### Advanced Usage

```python
from src.models import HybridRetrievalModel
from src.data import ClinicalTrialDataset, DataProcessor
from src.train import Trainer

# Initialize components
dataset = ClinicalTrialDataset()
processor = DataProcessor()
model = HybridRetrievalModel()

# Create training data
pairs = processor.create_retrieval_pairs(dataset)

# Train model
trainer = Trainer(model, train_dataloader, val_dataloader)
history = trainer.train()
```

## Configuration

The system uses YAML configuration files located in `configs/`:

### Model Configuration
```yaml
model:
  name: "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract"
  max_length: 512
  dropout: 0.1
  embedding_dim: 256
```

### Training Configuration
```yaml
training:
  batch_size: 32
  learning_rate: 2e-5
  num_epochs: 3
  warmup_steps: 100
```

### Evaluation Configuration
```yaml
evaluation:
  top_k_values: [1, 3, 5, 10]
  similarity_threshold: 0.5
  metrics: ["precision", "recall", "f1", "mrr", "auc_roc"]
```

## Dataset

The system includes a synthetic dataset for demonstration purposes:

### Clinical Trials
- 5 diverse clinical trials covering different conditions
- Detailed eligibility criteria and intervention descriptions
- Phase, status, and location information
- Structured metadata for filtering

### Patient Profiles
- 5 representative patient profiles
- Demographics, medical history, and preferences
- Biomarker information where relevant
- Location and treatment preferences

### Data Format
```json
{
  "trials": [
    {
      "id": "NCT001",
      "title": "Lung Cancer Immunotherapy Trial",
      "criteria": "Adults with advanced non-small cell lung cancer...",
      "phase": "Phase II",
      "status": "Recruiting",
      "conditions": ["Non-small cell lung cancer"],
      "interventions": ["Immunotherapy"],
      "locations": ["United States", "Canada"]
    }
  ],
  "patients": [
    {
      "id": "P001",
      "age": 65,
      "gender": "female",
      "diagnosis": "Stage II HER2-positive breast cancer",
      "medical_history": "Hypertension, diabetes type 2",
      "biomarkers": {"HER2": "positive"}
    }
  ]
}
```

## Model Architecture

### Dual-Encoder Model
- **Patient Encoder**: Encodes patient descriptions into embeddings
- **Trial Encoder**: Encodes trial descriptions into embeddings
- **Similarity Computation**: Cosine similarity between embeddings
- **Efficient Retrieval**: Fast similarity search across large trial databases

### Cross-Encoder Model
- **Pair Encoding**: Jointly encodes patient-trial pairs
- **Fine-grained Ranking**: Detailed relevance scoring
- **Higher Accuracy**: Better performance for ranking tasks
- **Computational Cost**: More expensive than dual-encoder

### Hybrid Model
- **Two-stage Process**: Dual-encoder for retrieval, cross-encoder for ranking
- **Balanced Performance**: Good accuracy with reasonable speed
- **Configurable**: Can disable cross-encoder for faster inference

## Evaluation

### Metrics

#### Classification Metrics
- **Precision**: True positives / (True positives + False positives)
- **Recall**: True positives / (True positives + False negatives)
- **F1-Score**: Harmonic mean of precision and recall

#### Ranking Metrics
- **MRR**: Mean Reciprocal Rank across queries
- **Hit Rate@K**: Percentage of queries with relevant results in top-K
- **NDCG@K**: Normalized Discounted Cumulative Gain at rank K

#### Information Retrieval Metrics
- **AUC-ROC**: Area under ROC curve
- **AUC-PR**: Area under Precision-Recall curve

### Evaluation Process

1. **Data Splitting**: Patient-level splits to prevent data leakage
2. **Cross-Validation**: Multiple folds for robust evaluation
3. **Threshold Analysis**: Performance at different similarity thresholds
4. **Ablation Studies**: Component-wise performance analysis

## Privacy and Safety

### De-identification
- **Automatic Detection**: Identifies PHI/PII patterns
- **Pattern Replacement**: Replaces sensitive information with placeholders
- **Configurable**: Can be enabled/disabled as needed
- **Demo Only**: Simple regex-based approach for demonstration

### Safety Measures
- **Clear Disclaimers**: Prominent warnings about research-only use
- **No Clinical Claims**: Explicitly states non-diagnostic purpose
- **Transparency**: Open about limitations and intended use
- **Professional Guidance**: Recommends consulting healthcare professionals

## Development

### Project Structure
```
clinical-trial-matching-system/
├── src/                    # Source code
│   ├── models/             # Model implementations
│   ├── data/               # Data processing
│   ├── train/              # Training utilities
│   ├── eval/               # Evaluation tools
│   ├── metrics/            # Evaluation metrics
│   └── utils/              # Utility functions
├── configs/                # Configuration files
├── scripts/                # Training/evaluation scripts
├── demo/                   # Streamlit demo application
├── tests/                  # Unit tests
├── data/                   # Data directory
├── outputs/                # Model outputs and results
└── assets/                 # Static assets
```

### Adding New Models

1. **Implement Model Class**:
```python
class NewModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        # Model implementation
    
    def forward(self, patient_texts, trial_texts):
        # Forward pass implementation
```

2. **Add to Model Registry**:
```python
MODEL_REGISTRY = {
    'hybrid': HybridRetrievalModel,
    'dual_encoder': DualEncoderRetrieval,
    'cross_encoder': CrossEncoderRanking,
    'new_model': NewModel
}
```

3. **Update Configuration**:
```yaml
model:
  type: "new_model"
  config:
    # Model-specific configuration
```

### Adding New Metrics

1. **Implement Metric Class**:
```python
class NewMetric:
    def __init__(self):
        self.reset()
    
    def update(self, predictions, labels):
        # Update metric state
    
    def compute(self):
        # Compute final metric value
```

2. **Register Metric**:
```python
METRIC_REGISTRY = {
    'precision': PrecisionMetric,
    'recall': RecallMetric,
    'new_metric': NewMetric
}
```

## Testing

### Run Tests
```bash
pytest tests/ -v
```

### Test Coverage
```bash
pytest tests/ --cov=src --cov-report=html
```

### Linting
```bash
black src/ tests/
ruff check src/ tests/
```

## Contributing

1. **Fork the repository**
2. **Create a feature branch**: `git checkout -b feature/new-feature`
3. **Make changes**: Follow coding standards and add tests
4. **Run tests**: Ensure all tests pass
5. **Submit pull request**: Include description of changes

### Coding Standards
- **Type Hints**: Use type annotations for all functions
- **Docstrings**: Google-style docstrings for all classes and functions
- **Formatting**: Black for code formatting, Ruff for linting
- **Testing**: Unit tests for all new functionality

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Citation

If you use this system in your research, please cite:

```bibtex
@software{clinical_trial_matching,
  title={Clinical Trial Matching System},
  author={Kryptologyst},
  year={2025},
  url={https://github.com/kryptologyst/Clinical-Trial-Matching-System}
}
```

## Acknowledgments

- Microsoft for the BiomedNLP-BiomedBERT model
- Hugging Face for the Transformers library
- Streamlit for the web interface framework
- The clinical NLP research community

## Contact

For questions or support, please open an issue on GitHub or contact [your-email@domain.com].

---

**Remember: This is a research demonstration system only. Not for clinical use.**
# Clinical-Trial-Matching-System
