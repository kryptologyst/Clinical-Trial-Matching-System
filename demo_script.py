#!/usr/bin/env python3
"""
Quick Demo Script for Clinical Trial Matching System

This script demonstrates the basic functionality of the clinical trial matching system
using the modernized implementation.

DISCLAIMER: This is for research and educational purposes only. NOT FOR CLINICAL USE.
"""

import sys
from pathlib import Path

# Add src to path for imports
sys.path.append(str(Path(__file__).parent / "src"))

from src.utils import set_seed, get_device, create_safety_banner
from src.data import ClinicalTrialDataset
from src.models import ClinicalTrialMatcher


def run_basic_demo():
    """Run basic demo with TF-IDF approach."""
    print("=" * 60)
    print("BASIC DEMO - TF-IDF Based Clinical Trial Matching")
    print("=" * 60)
    
    # Import and run the modernized original script
    import importlib.util
    spec = importlib.util.spec_from_file_location("original_demo", "0458.py")
    original_demo = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(original_demo)
    original_demo.main()


def run_advanced_demo():
    """Run advanced demo with ClinicalBERT."""
    print("=" * 60)
    print("ADVANCED DEMO - ClinicalBERT Based Clinical Trial Matching")
    print("=" * 60)
    
    print(create_safety_banner())
    
    try:
        # Set random seed for reproducibility
        set_seed(42)
        
        # Get device
        device = get_device()
        print(f"Using device: {device}")
        
        # Load dataset
        print("\nLoading clinical trial dataset...")
        dataset = ClinicalTrialDataset()
        print(f"Loaded {len(dataset.trials)} trials and {len(dataset.patients)} patients")
        
        # Initialize matcher
        print("\nInitializing ClinicalBERT matcher...")
        matcher = ClinicalTrialMatcher(device=device)
        
        # Demo patient profiles
        demo_patients = [
            "65-year-old female with stage II HER2-positive breast cancer seeking targeted therapy",
            "58-year-old male with advanced non-small cell lung cancer, no prior immunotherapy",
            "72-year-old male with type 2 diabetes, HbA1c 8.5%, seeking new medication options"
        ]
        
        # Get trial texts
        trial_texts = [dataset.get_trial_text(trial['id']) for trial in dataset.trials]
        trial_titles = [trial['title'] for trial in dataset.trials]
        
        # Run matching for each patient
        for i, patient_text in enumerate(demo_patients, 1):
            print(f"\n{'='*50}")
            print(f"PATIENT {i}: {patient_text}")
            print(f"{'='*50}")
            
            try:
                # Find matches
                matches = matcher.match_trials(patient_text, trial_texts, top_k=3)
                
                # Display results
                for j, (trial_text, score) in enumerate(matches, 1):
                    trial_idx = trial_texts.index(trial_text)
                    trial_title = trial_titles[trial_idx]
                    
                    print(f"\n{j}. {trial_title}")
                    print(f"   Similarity Score: {score:.3f}")
                    print(f"   Trial ID: {dataset.trials[trial_idx]['id']}")
                    print(f"   Phase: {dataset.trials[trial_idx]['phase']}")
                    print(f"   Status: {dataset.trials[trial_idx]['status']}")
                    
            except Exception as e:
                print(f"Error during matching: {e}")
                print("Falling back to basic TF-IDF approach...")
                
                # Fallback to basic approach
                import importlib.util
                spec = importlib.util.spec_from_file_location("original_demo", "0458.py")
                original_demo = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(original_demo)
                
                compute_tfidf_similarity = original_demo.compute_tfidf_similarity
                rank_trial_matches = original_demo.rank_trial_matches
                
                similarities = compute_tfidf_similarity(patient_text, trial_texts)
                matches = rank_trial_matches(dataset.trials, similarities, top_k=3)
                
                for j, (trial, score) in enumerate(matches, 1):
                    print(f"\n{j}. {trial['title']}")
                    print(f"   Similarity Score: {score:.3f}")
                    print(f"   Trial ID: {trial['id']}")
                    print(f"   Phase: {trial['phase']}")
        
        print(f"\n{'='*60}")
        print("Advanced demo completed!")
        print("For interactive demo, run: streamlit run demo/app.py")
        
    except Exception as e:
        print(f"Error in advanced demo: {e}")
        print("This is expected if ClinicalBERT model is not available.")
        print("Falling back to basic demo...")


def main():
    """Main demo function."""
    print("Clinical Trial Matching System - Demo")
    print("=" * 60)
    
    # Check if advanced models are available
    try:
        import transformers
        print("✓ Transformers library available - running advanced demo")
        run_advanced_demo()
    except ImportError:
        print("⚠ Transformers library not available - running basic demo")
        run_basic_demo()
    
    print("\n" + "=" * 60)
    print("Demo completed!")
    print("\nNext steps:")
    print("1. Install full dependencies: pip install -r requirements.txt")
    print("2. Run interactive demo: streamlit run demo/app.py")
    print("3. Train a model: python scripts/train.py")
    print("4. Read documentation: README.md")
    print("\nRemember: This is for research purposes only!")


if __name__ == "__main__":
    main()
