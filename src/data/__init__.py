"""Data processing and loading utilities for clinical trial matching."""

import json
import pandas as pd
from typing import Any, Dict, List, Optional, Tuple, Union
from pathlib import Path
import logging
from sklearn.model_selection import train_test_split
import numpy as np

logger = logging.getLogger(__name__)


class ClinicalTrialDataset:
    """Dataset class for clinical trials and patient data."""
    
    def __init__(self, data_path: Optional[str] = None):
        """Initialize dataset.
        
        Args:
            data_path: Path to data file. If None, uses synthetic data.
        """
        self.trials: List[Dict[str, Any]] = []
        self.patients: List[Dict[str, Any]] = []
        
        if data_path and Path(data_path).exists():
            self.load_from_file(data_path)
        else:
            self._create_synthetic_data()
    
    def _create_synthetic_data(self) -> None:
        """Create synthetic clinical trial and patient data for demonstration."""
        logger.info("Creating synthetic clinical trial dataset")
        
        # Synthetic clinical trials
        self.trials = [
            {
                "id": "NCT001",
                "title": "Lung Cancer Immunotherapy Trial",
                "criteria": "Adults with advanced non-small cell lung cancer and no prior immunotherapy treatment. Age 18-80, ECOG 0-1.",
                "phase": "Phase II",
                "status": "Recruiting",
                "conditions": ["Non-small cell lung cancer", "Advanced cancer"],
                "interventions": ["Immunotherapy", "Checkpoint inhibitor"],
                "locations": ["United States", "Canada"],
                "sponsor": "Academic Medical Center"
            },
            {
                "id": "NCT002",
                "title": "Diabetes Type 2 Medication Study",
                "criteria": "Patients diagnosed with type 2 diabetes and HbA1c above 7.0%. Age 18-75, BMI 18.5-40.",
                "phase": "Phase III",
                "status": "Recruiting",
                "conditions": ["Type 2 diabetes", "Hyperglycemia"],
                "interventions": ["Metformin", "GLP-1 agonist"],
                "locations": ["United States", "Europe"],
                "sponsor": "Pharmaceutical Company"
            },
            {
                "id": "NCT003",
                "title": "Breast Cancer HER2+ Targeted Therapy",
                "criteria": "Female patients with HER2-positive breast cancer, stage II or III. Age 18-70, adequate organ function.",
                "phase": "Phase II",
                "status": "Recruiting",
                "conditions": ["Breast cancer", "HER2 positive"],
                "interventions": ["Trastuzumab", "Pertuzumab"],
                "locations": ["United States", "Europe", "Asia"],
                "sponsor": "Biotech Company"
            },
            {
                "id": "NCT004",
                "title": "COVID-19 Vaccine Booster Evaluation",
                "criteria": "Adults previously vaccinated with two doses and no recent infections. Age 18-65, healthy volunteers.",
                "phase": "Phase III",
                "status": "Recruiting",
                "conditions": ["COVID-19", "Vaccination"],
                "interventions": ["mRNA vaccine", "Booster dose"],
                "locations": ["United States", "Canada", "Europe"],
                "sponsor": "Government Agency"
            },
            {
                "id": "NCT005",
                "title": "Alzheimer's Disease Cognitive Assessment",
                "criteria": "Patients with mild to moderate Alzheimer's disease. Age 50-85, MMSE 12-26, stable medications.",
                "phase": "Phase II",
                "status": "Recruiting",
                "conditions": ["Alzheimer's disease", "Dementia"],
                "interventions": ["Cognitive training", "Memory enhancement"],
                "locations": ["United States", "Europe"],
                "sponsor": "Research Institute"
            }
        ]
        
        # Synthetic patient profiles
        self.patients = [
            {
                "id": "P001",
                "age": 65,
                "gender": "female",
                "diagnosis": "Stage II HER2-positive breast cancer",
                "medical_history": "Hypertension, diabetes type 2",
                "medications": ["Metformin", "Lisinopril"],
                "biomarkers": {"HER2": "positive", "ER": "negative", "PR": "negative"},
                "location": "United States",
                "preferences": "Targeted therapy, clinical trials"
            },
            {
                "id": "P002",
                "age": 58,
                "gender": "male",
                "diagnosis": "Advanced non-small cell lung cancer",
                "medical_history": "Smoking history, COPD",
                "medications": ["Albuterol", "Prednisone"],
                "biomarkers": {"PD-L1": "high", "EGFR": "wild-type"},
                "location": "Canada",
                "preferences": "Immunotherapy, novel treatments"
            },
            {
                "id": "P003",
                "age": 72,
                "gender": "male",
                "diagnosis": "Type 2 diabetes with poor glycemic control",
                "medical_history": "Obesity, hypertension, diabetic nephropathy",
                "medications": ["Metformin", "Insulin", "ACE inhibitor"],
                "biomarkers": {"HbA1c": 8.5, "eGFR": 45},
                "location": "United States",
                "preferences": "New diabetes medications, lifestyle interventions"
            },
            {
                "id": "P004",
                "age": 45,
                "gender": "female",
                "diagnosis": "Mild Alzheimer's disease",
                "medical_history": "Family history of dementia, depression",
                "medications": ["Donepezil", "Sertraline"],
                "biomarkers": {"MMSE": 22, "APOE": "e4/e4"},
                "location": "Europe",
                "preferences": "Cognitive interventions, memory training"
            },
            {
                "id": "P005",
                "age": 35,
                "gender": "female",
                "diagnosis": "Healthy volunteer for COVID-19 vaccine study",
                "medical_history": "No significant medical history",
                "medications": ["Multivitamin"],
                "biomarkers": {"Previous vaccination": "2 doses mRNA"},
                "location": "United States",
                "preferences": "Vaccine research, preventive medicine"
            }
        ]
    
    def load_from_file(self, data_path: str) -> None:
        """Load data from JSON file.
        
        Args:
            data_path: Path to JSON data file.
        """
        with open(data_path, 'r') as f:
            data = json.load(f)
        
        self.trials = data.get('trials', [])
        self.patients = data.get('patients', [])
        
        logger.info(f"Loaded {len(self.trials)} trials and {len(self.patients)} patients")
    
    def save_to_file(self, data_path: str) -> None:
        """Save data to JSON file.
        
        Args:
            data_path: Path to save JSON data file.
        """
        data = {
            'trials': self.trials,
            'patients': self.patients
        }
        
        with open(data_path, 'w') as f:
            json.dump(data, f, indent=2)
        
        logger.info(f"Saved data to {data_path}")
    
    def get_trial_text(self, trial_id: str) -> str:
        """Get combined text representation of a trial.
        
        Args:
            trial_id: Trial identifier.
            
        Returns:
            Combined text representation.
        """
        trial = next((t for t in self.trials if t['id'] == trial_id), None)
        if not trial:
            return ""
        
        text_parts = [
            trial['title'],
            trial['criteria'],
            ' '.join(trial['conditions']),
            ' '.join(trial['interventions'])
        ]
        
        return ' '.join(text_parts)
    
    def get_patient_text(self, patient_id: str) -> str:
        """Get combined text representation of a patient.
        
        Args:
            patient_id: Patient identifier.
            
        Returns:
            Combined text representation.
        """
        patient = next((p for p in self.patients if p['id'] == patient_id), None)
        if not patient:
            return ""
        
        text_parts = [
            f"{patient['age']} year old {patient['gender']}",
            patient['diagnosis'],
            patient['medical_history'],
            ' '.join(patient['medications']),
            patient['location']
        ]
        
        return ' '.join(text_parts)
    
    def create_train_test_split(self, test_size: float = 0.2, random_state: int = 42) -> Tuple[List[str], List[str]]:
        """Create train/test split for patient-level evaluation.
        
        Args:
            test_size: Proportion of data for testing.
            random_state: Random seed for reproducibility.
            
        Returns:
            Tuple of (train_patient_ids, test_patient_ids).
        """
        patient_ids = [p['id'] for p in self.patients]
        train_ids, test_ids = train_test_split(
            patient_ids, 
            test_size=test_size, 
            random_state=random_state
        )
        
        logger.info(f"Created train/test split: {len(train_ids)} train, {len(test_ids)} test")
        return train_ids, test_ids


class DataProcessor:
    """Data processing utilities for clinical trial matching."""
    
    def __init__(self, max_length: int = 512):
        """Initialize data processor.
        
        Args:
            max_length: Maximum sequence length for text processing.
        """
        self.max_length = max_length
    
    def preprocess_text(self, text: str) -> str:
        """Preprocess text for model input.
        
        Args:
            text: Input text.
            
        Returns:
            Preprocessed text.
        """
        # Basic text cleaning
        text = text.lower().strip()
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        return text
    
    def create_retrieval_pairs(self, dataset: ClinicalTrialDataset) -> List[Dict[str, Any]]:
        """Create positive and negative retrieval pairs for training.
        
        Args:
            dataset: Clinical trial dataset.
            
        Returns:
            List of retrieval pairs with labels.
        """
        pairs = []
        
        # Create positive pairs (patient-trial matches)
        for patient in dataset.patients:
            patient_text = dataset.get_patient_text(patient['id'])
            
            # Simple heuristic for positive matches based on conditions
            for trial in dataset.trials:
                trial_text = dataset.get_trial_text(trial['id'])
                
                # Check for condition overlap
                is_positive = self._check_condition_overlap(patient, trial)
                
                pairs.append({
                    'patient_id': patient['id'],
                    'trial_id': trial['id'],
                    'patient_text': patient_text,
                    'trial_text': trial_text,
                    'label': 1 if is_positive else 0,
                    'similarity_score': 0.0  # Will be computed by model
                })
        
        logger.info(f"Created {len(pairs)} retrieval pairs")
        return pairs
    
    def _check_condition_overlap(self, patient: Dict[str, Any], trial: Dict[str, Any]) -> bool:
        """Check if patient conditions overlap with trial conditions.
        
        Args:
            patient: Patient data.
            trial: Trial data.
            
        Returns:
            True if there's condition overlap.
        """
        patient_diagnosis = patient['diagnosis'].lower()
        trial_conditions = [c.lower() for c in trial['conditions']]
        
        # Simple keyword matching
        for condition in trial_conditions:
            if any(keyword in patient_diagnosis for keyword in condition.split()):
                return True
        
        return False
