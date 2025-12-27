#!/usr/bin/env python3
"""
Clinical Trial Matching System - Modernized Implementation

This is a research demonstration system that matches patients to relevant clinical trials
using advanced NLP techniques. This script provides a simple example of the core functionality.

DISCLAIMER: This is for research and educational purposes only. NOT FOR CLINICAL USE.
"""

import logging
import warnings
from typing import List, Dict, Tuple, Any
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore", category=UserWarning)

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def create_safety_banner() -> str:
    """Create safety disclaimer banner."""
    return """
    ⚠️  RESEARCH DEMONSTRATION ONLY ⚠️
    
    This system is for educational and research purposes only.
    NOT FOR CLINICAL USE. NOT MEDICAL ADVICE.
    
    Always consult qualified healthcare professionals for medical decisions.
    """


def load_sample_data() -> Tuple[List[Dict[str, str]], str]:
    """Load sample clinical trials and patient profile.
    
    Returns:
        Tuple of (trials_list, patient_profile).
    """
    trials = [
        {
            "id": "NCT001",
            "title": "Lung Cancer Immunotherapy Trial",
            "criteria": "Adults with advanced non-small cell lung cancer and no prior immunotherapy treatment. Age 18-80, ECOG 0-1.",
            "phase": "Phase II",
            "status": "Recruiting",
            "conditions": ["Non-small cell lung cancer", "Advanced cancer"],
            "interventions": ["Immunotherapy", "Checkpoint inhibitor"],
            "locations": ["United States", "Canada"]
        },
        {
            "id": "NCT002", 
            "title": "Diabetes Type 2 Medication Study",
            "criteria": "Patients diagnosed with type 2 diabetes and HbA1c above 7.0%. Age 18-75, BMI 18.5-40.",
            "phase": "Phase III",
            "status": "Recruiting",
            "conditions": ["Type 2 diabetes", "Hyperglycemia"],
            "interventions": ["Metformin", "GLP-1 agonist"],
            "locations": ["United States", "Europe"]
        },
        {
            "id": "NCT003",
            "title": "Breast Cancer HER2+ Targeted Therapy", 
            "criteria": "Female patients with HER2-positive breast cancer, stage II or III. Age 18-70, adequate organ function.",
            "phase": "Phase II",
            "status": "Recruiting",
            "conditions": ["Breast cancer", "HER2 positive"],
            "interventions": ["Trastuzumab", "Pertuzumab"],
            "locations": ["United States", "Europe", "Asia"]
        },
        {
            "id": "NCT004",
            "title": "COVID-19 Vaccine Booster Evaluation",
            "criteria": "Adults previously vaccinated with two doses and no recent infections. Age 18-65, healthy volunteers.",
            "phase": "Phase III", 
            "status": "Recruiting",
            "conditions": ["COVID-19", "Vaccination"],
            "interventions": ["mRNA vaccine", "Booster dose"],
            "locations": ["United States", "Canada", "Europe"]
        }
    ]
    
    patient_profile = "65-year-old female with stage II HER2-positive breast cancer seeking targeted therapy"
    
    return trials, patient_profile


def compute_tfidf_similarity(patient_text: str, trial_texts: List[str]) -> List[float]:
    """Compute TF-IDF based similarity scores.
    
    Args:
        patient_text: Patient description text.
        trial_texts: List of trial description texts.
        
    Returns:
        List of similarity scores.
    """
    # Combine patient profile with trial criteria
    corpus = [patient_text] + trial_texts
    
    # Compute TF-IDF embeddings
    vectorizer = TfidfVectorizer(
        max_features=1000,
        stop_words='english',
        ngram_range=(1, 2)
    )
    X = vectorizer.fit_transform(corpus)
    
    # Compute cosine similarity between patient and each trial
    similarities = cosine_similarity(X[0:1], X[1:]).flatten()
    
    return similarities.tolist()


def rank_trial_matches(trials: List[Dict[str, Any]], similarities: List[float], 
                      top_k: int = 3) -> List[Tuple[Dict[str, Any], float]]:
    """Rank trial matches by similarity score.
    
    Args:
        trials: List of trial dictionaries.
        similarities: List of similarity scores.
        top_k: Number of top matches to return.
        
    Returns:
        List of (trial, score) tuples sorted by score.
    """
    # Create trial-score pairs
    trial_scores = list(zip(trials, similarities))
    
    # Sort by similarity score (descending)
    trial_scores.sort(key=lambda x: x[1], reverse=True)
    
    # Return top-k matches
    return trial_scores[:top_k]


def display_results(matches: List[Tuple[Dict[str, Any], float]]) -> None:
    """Display trial matching results.
    
    Args:
        matches: List of (trial, score) tuples.
    """
    print("Top Clinical Trial Matches:")
    print("=" * 50)
    
    for i, (trial, score) in enumerate(matches, 1):
        print(f"\n{i}. {trial['title']}")
        print(f"   Similarity Score: {score:.3f}")
        print(f"   Trial ID: {trial['id']}")
        print(f"   Phase: {trial['phase']}")
        print(f"   Status: {trial['status']}")
        print(f"   Conditions: {', '.join(trial['conditions'])}")
        print(f"   Interventions: {', '.join(trial['interventions'])}")
        print(f"   Locations: {', '.join(trial['locations'])}")
        print(f"   Criteria: {trial['criteria'][:100]}...")


def main():
    """Main function demonstrating clinical trial matching."""
    # Display safety banner
    print(create_safety_banner())
    
    logger.info("Loading sample clinical trial data")
    
    # Load sample data
    trials, patient_profile = load_sample_data()
    
    logger.info(f"Loaded {len(trials)} clinical trials")
    logger.info(f"Patient profile: {patient_profile}")
    
    # Extract trial criteria texts
    trial_texts = [trial["criteria"] for trial in trials]
    
    logger.info("Computing TF-IDF similarity scores")
    
    # Compute similarity scores
    similarities = compute_tfidf_similarity(patient_profile, trial_texts)
    
    logger.info("Ranking trial matches")
    
    # Rank and get top matches
    top_matches = rank_trial_matches(trials, similarities, top_k=3)
    
    # Display results
    display_results(top_matches)
    
    logger.info("Clinical trial matching completed")
    
    print("\n" + "=" * 50)
    print("Note: This is a simple TF-IDF based approach.")
    print("For better results, use the full system with ClinicalBERT embeddings.")
    print("See README.md for advanced usage instructions.")


if __name__ == "__main__":
    main()