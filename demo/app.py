"""Streamlit demo for clinical trial matching system."""

import streamlit as st
import torch
import yaml
from pathlib import Path
import logging
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from typing import List, Dict, Any

from src.utils import set_seed, get_device, deidentify_text, create_safety_banner
from src.data import ClinicalTrialDataset
from src.models import ClinicalTrialMatcher

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="Clinical Trial Matching System",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Safety banner
st.markdown(create_safety_banner())

# Title and description
st.title("🏥 Clinical Trial Matching System")
st.markdown("""
This system matches patients to relevant clinical trials using advanced NLP techniques.
**This is a research demonstration only - NOT FOR CLINICAL USE.**
""")

# Sidebar configuration
st.sidebar.header("Configuration")

# Load configuration
@st.cache_resource
def load_config():
    """Load configuration with caching."""
    config_path = Path("configs/default.yaml")
    if config_path.exists():
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    return {
        'model': {'name': 'microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract'},
        'data': {'deid_mode': True},
        'evaluation': {'similarity_threshold': 0.3}
    }

config = load_config()

# Model selection
model_name = st.sidebar.selectbox(
    "Model",
    [
        "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract",
        "dmis-lab/biobert-base-cased-v1.1",
        "allenai/scibert_scivocab_uncased"
    ],
    index=0
)

# De-identification toggle
deid_mode = st.sidebar.checkbox("Enable De-identification", value=config['data']['deid_mode'])

# Similarity threshold
similarity_threshold = st.sidebar.slider(
    "Similarity Threshold",
    min_value=0.0,
    max_value=1.0,
    value=config['evaluation']['similarity_threshold'],
    step=0.05
)

# Number of results
top_k = st.sidebar.slider("Number of Results", min_value=1, max_value=10, value=5)

# Initialize model (cached)
@st.cache_resource
def load_model(_model_name: str):
    """Load model with caching."""
    try:
        matcher = ClinicalTrialMatcher(model_name=_model_name, device=get_device())
        return matcher
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

matcher = load_model(model_name)

# Load dataset
@st.cache_data
def load_dataset():
    """Load dataset with caching."""
    return ClinicalTrialDataset()

dataset = load_dataset()

# Main interface
tab1, tab2, tab3, tab4 = st.tabs(["🔍 Patient Matching", "📊 Dataset Overview", "📈 Model Performance", "ℹ️ About"])

with tab1:
    st.header("Patient-Trial Matching")
    
    # Patient input
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Patient Profile")
        patient_text = st.text_area(
            "Enter patient description:",
            value="65-year-old female with stage II HER2-positive breast cancer seeking targeted therapy",
            height=100,
            help="Describe the patient's condition, demographics, and treatment preferences"
        )
    
    with col2:
        st.subheader("Patient Demographics")
        age = st.number_input("Age", min_value=0, max_value=120, value=65)
        gender = st.selectbox("Gender", ["Male", "Female", "Other"])
        location = st.selectbox("Location", ["United States", "Canada", "Europe", "Asia"])
    
    # De-identify text if enabled
    if deid_mode:
        patient_text_deid = deidentify_text(patient_text, deid_mode=True)
        if patient_text_deid != patient_text:
            st.info("🔒 Text has been de-identified for privacy")
            st.text_area("De-identified text:", value=patient_text_deid, height=100, disabled=True)
    
    # Trial selection
    st.subheader("Available Trials")
    trial_options = [f"{trial['id']}: {trial['title']}" for trial in dataset.trials]
    selected_trials = st.multiselect(
        "Select trials to match against (leave empty for all trials):",
        trial_options,
        default=[]
    )
    
    # Get trial texts
    if selected_trials:
        trial_ids = [option.split(':')[0] for option in selected_trials]
        trial_texts = [dataset.get_trial_text(trial_id) for trial_id in trial_ids]
        trial_titles = [dataset.trials[i]['title'] for i, trial in enumerate(dataset.trials) if trial['id'] in trial_ids]
    else:
        trial_texts = [dataset.get_trial_text(trial['id']) for trial in dataset.trials]
        trial_titles = [trial['title'] for trial in dataset.trials]
    
    # Match button
    if st.button("🔍 Find Matching Trials", type="primary"):
        if matcher is None:
            st.error("Model not loaded. Please check the model configuration.")
        elif not patient_text.strip():
            st.warning("Please enter a patient description.")
        else:
            with st.spinner("Finding matching trials..."):
                try:
                    # Get matches
                    matches = matcher.match_trials(patient_text, trial_texts, top_k=top_k)
                    
                    # Display results
                    st.subheader("🎯 Matching Results")
                    
                    if not matches:
                        st.warning("No matching trials found.")
                    else:
                        # Create results dataframe
                        results_data = []
                        for i, (trial_text, score) in enumerate(matches):
                            if score >= similarity_threshold:
                                trial_idx = trial_texts.index(trial_text)
                                results_data.append({
                                    'Rank': i + 1,
                                    'Trial': trial_titles[trial_idx],
                                    'Similarity Score': score,
                                    'Match': '✅' if score >= similarity_threshold else '❌'
                                })
                        
                        if results_data:
                            results_df = pd.DataFrame(results_data)
                            
                            # Display table
                            st.dataframe(results_df, use_container_width=True)
                            
                            # Display detailed results
                            for i, (trial_text, score) in enumerate(matches):
                                if score >= similarity_threshold:
                                    trial_idx = trial_texts.index(trial_text)
                                    trial_info = dataset.trials[trial_idx]
                                    
                                    with st.expander(f"Rank {i+1}: {trial_info['title']} (Score: {score:.3f})"):
                                        col1, col2 = st.columns(2)
                                        
                                        with col1:
                                            st.write("**Trial Details:**")
                                            st.write(f"- **ID:** {trial_info['id']}")
                                            st.write(f"- **Phase:** {trial_info['phase']}")
                                            st.write(f"- **Status:** {trial_info['status']}")
                                            st.write(f"- **Sponsor:** {trial_info['sponsor']}")
                                        
                                        with col2:
                                            st.write("**Eligibility Criteria:**")
                                            st.write(trial_info['criteria'])
                                        
                                        st.write("**Conditions:**")
                                        st.write(", ".join(trial_info['conditions']))
                                        
                                        st.write("**Interventions:**")
                                        st.write(", ".join(trial_info['interventions']))
                                        
                                        st.write("**Locations:**")
                                        st.write(", ".join(trial_info['locations']))
                        else:
                            st.warning(f"No trials meet the similarity threshold of {similarity_threshold}")
                    
                except Exception as e:
                    st.error(f"Error during matching: {e}")
                    logger.error(f"Matching error: {e}")

with tab2:
    st.header("Dataset Overview")
    
    # Trial statistics
    st.subheader("Clinical Trials")
    
    trials_df = pd.DataFrame(dataset.trials)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Trial Phases:**")
        phase_counts = trials_df['phase'].value_counts()
        fig_phases = px.pie(values=phase_counts.values, names=phase_counts.index, title="Trial Phases")
        st.plotly_chart(fig_phases, use_container_width=True)
    
    with col2:
        st.write("**Trial Status:**")
        status_counts = trials_df['status'].value_counts()
        fig_status = px.bar(x=status_counts.index, y=status_counts.values, title="Trial Status")
        st.plotly_chart(fig_status, use_container_width=True)
    
    # Trial details table
    st.subheader("All Trials")
    display_trials_df = trials_df[['id', 'title', 'phase', 'status', 'sponsor']].copy()
    st.dataframe(display_trials_df, use_container_width=True)
    
    # Patient statistics
    st.subheader("Patient Profiles")
    
    patients_df = pd.DataFrame(dataset.patients)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Age Distribution:**")
        fig_age = px.histogram(patients_df, x='age', nbins=10, title="Patient Age Distribution")
        st.plotly_chart(fig_age, use_container_width=True)
    
    with col2:
        st.write("**Gender Distribution:**")
        gender_counts = patients_df['gender'].value_counts()
        fig_gender = px.pie(values=gender_counts.values, names=gender_counts.index, title="Patient Gender")
        st.plotly_chart(fig_gender, use_container_width=True)
    
    # Patient details table
    st.subheader("All Patient Profiles")
    display_patients_df = patients_df[['id', 'age', 'gender', 'diagnosis', 'location']].copy()
    st.dataframe(display_patients_df, use_container_width=True)

with tab3:
    st.header("Model Performance")
    
    st.info("This section would display model performance metrics and evaluation results.")
    
    # Placeholder for performance metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Precision", "0.85", "0.02")
    
    with col2:
        st.metric("Recall", "0.78", "-0.01")
    
    with col3:
        st.metric("F1 Score", "0.81", "0.01")
    
    with col4:
        st.metric("MRR", "0.72", "0.03")
    
    # Performance chart placeholder
    st.subheader("Performance Over Time")
    
    # Generate sample data
    epochs = list(range(1, 11))
    train_loss = [0.8 - 0.05 * i + 0.01 * (i % 3) for i in epochs]
    val_loss = [0.75 - 0.04 * i + 0.02 * (i % 2) for i in epochs]
    
    fig_performance = go.Figure()
    fig_performance.add_trace(go.Scatter(x=epochs, y=train_loss, name='Training Loss', line=dict(color='blue')))
    fig_performance.add_trace(go.Scatter(x=epochs, y=val_loss, name='Validation Loss', line=dict(color='red')))
    
    fig_performance.update_layout(
        title="Training Progress",
        xaxis_title="Epoch",
        yaxis_title="Loss",
        hovermode='x unified'
    )
    
    st.plotly_chart(fig_performance, use_container_width=True)

with tab4:
    st.header("About This System")
    
    st.markdown("""
    ## Clinical Trial Matching System
    
    This system uses advanced natural language processing techniques to match patients with relevant clinical trials.
    
    ### Features:
    - **Dual-Encoder Architecture**: Efficient retrieval using separate encoders for patients and trials
    - **Cross-Encoder Refinement**: Fine-grained ranking for improved accuracy
    - **Clinical BERT**: Pre-trained on biomedical literature for better understanding
    - **De-identification**: Privacy-preserving text processing
    - **Comprehensive Evaluation**: Multiple metrics for thorough assessment
    
    ### Model Architecture:
    1. **Text Encoding**: Clinical BERT encodes patient and trial descriptions
    2. **Similarity Computation**: Cosine similarity between embeddings
    3. **Ranking**: Cross-encoder refines similarity scores
    4. **Threshold Filtering**: Only returns trials above similarity threshold
    
    ### Evaluation Metrics:
    - **Precision/Recall/F1**: Classification performance
    - **MRR**: Mean Reciprocal Rank for ranking quality
    - **Hit Rate@K**: Success rate at different ranks
    - **NDCG**: Normalized Discounted Cumulative Gain
    
    ### Safety and Privacy:
    - **De-identification**: Automatic removal of PHI/PII
    - **Research Only**: Not intended for clinical use
    - **Transparency**: Open about limitations and intended use
    
    ### Technical Details:
    - **Framework**: PyTorch with Transformers
    - **Model**: Microsoft BiomedNLP-BiomedBERT
    - **Interface**: Streamlit web application
    - **Evaluation**: Comprehensive metrics suite
    """)
    
    st.subheader("Disclaimer")
    st.warning("""
    **IMPORTANT DISCLAIMER:**
    
    This system is for educational and research purposes only. It is NOT intended for clinical use and should NOT be used as medical advice.
    
    - Always consult qualified healthcare professionals for medical decisions
    - Clinical trial enrollment should be done through proper medical channels
    - This system may have limitations and biases
    - Results should be interpreted with caution
    """)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray;'>
    Clinical Trial Matching System - Research Demo Only
</div>
""", unsafe_allow_html=True)
