import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler
from joblib import load as joblib_load
import json
import os

# Configuration
MODEL_DIRS = {
    "Deep Learning": "Deep Learning/Feedforward Neural Network/models/",
    "Machine Learning": "Machine Learning/Suprivised ML/SML/"  # Fixed path for ML models
}
MODEL_INFO_FILE = "Website/model_info.json"

# Load model information
@st.cache_data
def load_model_info():
    try:
        with open(MODEL_INFO_FILE) as f:
            return json.load(f)
    except FileNotFoundError:
        st.warning(f"Model information file not found: {MODEL_INFO_FILE}")
        return {}

# Fault type mapping with detailed descriptions
FAULT_MAP = {
    0: {
        "name": "No Fault",
        "description": "The system is operating normally with balanced currents and voltages.",
        "symptoms": [
            "Balanced three-phase currents",
            "Nominal voltage levels",
            "No protective device operation"
        ],
        "actions": [
            "Continue normal operation",
            "Monitor system parameters"
        ]
    },
    1: {
        "name": "LG (Line to Ground)",
        "description": "A single line-to-ground fault occurs when one conductor comes into contact with ground.",
        "symptoms": [
            "High current in faulted phase",
            "Voltage collapse in faulted phase",
            "Ground fault protection activation"
        ],
        "actions": [
            "Isolate the affected line",
            "Check insulation of faulted phase",
            "Inspect grounding systems"
        ]
    },
    2: {
        "name": "LL (Line to Line)",
        "description": "A line-to-line fault occurs when two conductors come into contact with each other.",
        "symptoms": [
            "High currents in both faulted phases",
            "Voltage drop in both phases",
            "Phase protection activation"
        ],
        "actions": [
            "Isolate the affected circuit",
            "Check for physical conductor damage",
            "Inspect insulation between phases"
        ]
    },
    3: {
        "name": "LLG (Line-Line to Ground)",
        "description": "A double line-to-ground fault occurs when two conductors contact both each other and ground.",
        "symptoms": [
            "High currents in both faulted phases",
            "Voltage collapse in faulted phases",
            "Both phase and ground protection activation"
        ],
        "actions": [
            "Immediate isolation required",
            "Check for multiple insulation failures",
            "Inspect nearby lightning arresters"
        ]
    },
    4: {
        "name": "LLL (Three-phase)",
        "description": "A three-phase fault occurs when all three conductors come into contact with each other.",
        "symptoms": [
            "Extremely high currents in all phases",
            "Complete voltage collapse",
            "Instantaneous protection operation"
        ],
        "actions": [
            "Emergency shutdown required",
            "Check for catastrophic insulation failure",
            "Inspect for physical damage to busbars"
        ]
    },
    5: {
        "name": "LLLG (Three-phase to Ground)",
        "description": "A three-phase-to-ground fault is the most severe fault where all phases contact each other and ground.",
        "symptoms": [
            "Maximum possible fault currents",
            "Complete voltage collapse",
            "All protection systems activated"
        ],
        "actions": [
            "Emergency shutdown and isolation",
            "Full system inspection required",
            "Check for explosion or fire damage"
        ]
    }
}

# Available models
@st.cache_data
def get_available_models():
    models = {}
    model_info = load_model_info()
    
    for model_type, model_dir in MODEL_DIRS.items():
        if os.path.exists(model_dir):
            for file in os.listdir(model_dir):
                if file.endswith('.h5') or file.endswith('.joblib'):
                    model_name = file.split('.')[0]
                    models[model_name] = {
                        "path": os.path.join(model_dir, file),
                        "type": model_type,
                        "info": model_info.get(model_name, {})
                    }
        else:
            st.sidebar.warning(f"Directory not found: {model_dir}")
    
    return models

# Load the trained model
@st.cache_resource
def load_trained_model(model_path):
    try:
        if model_path.endswith('.h5'):
            return load_model(model_path)
        elif model_path.endswith('.joblib'):
            return joblib_load(model_path)
        else:
            raise ValueError(f"Unsupported model format: {model_path}. Only .h5 and .joblib files are supported.")
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

# Create and fit a scaler instance with typical power system measurements
@st.cache_resource
def get_fitted_scaler():
    scaler = StandardScaler()
    # Fit the scaler with typical ranges of power system measurements
    typical_measurements = np.array([
        # Normal operating conditions
        [1.0, 1.0, 1.0, 230.0, 230.0, 230.0],
        # Various fault conditions
        [5.0, 1.0, 1.0, 180.0, 230.0, 230.0],  # LG fault
        [4.5, 4.5, 1.0, 120.0, 120.0, 230.0],  # LL fault
        [8.0, 8.0, 1.0, 50.0, 50.0, 230.0],    # LLG fault
        [9.0, 9.0, 9.0, 50.0, 50.0, 50.0],     # LLL fault
        [10.0, 10.0, 10.0, 0.0, 0.0, 0.0]      # LLLG fault
    ])
    scaler.fit(typical_measurements)
    return scaler

def preprocess_input(input_data):
    """Preprocess user input for prediction using pre-fitted scaler"""
    scaler = get_fitted_scaler()
    features = np.array(input_data).reshape(1, -1)
    scaled_features = scaler.transform(features)
    return scaled_features

def predict_fault(model, input_data):
    """Make prediction using the model"""
    if model is None:
        st.error("Model not loaded properly!")
        return 0, 0, np.zeros(6)
        
    processed_data = preprocess_input(input_data)
    
    try:
        prediction = model.predict(processed_data)
        # Handle different prediction formats for different model types
        if len(prediction.shape) > 1 and prediction.shape[1] > 1:
            # For models that return probabilities for each class
            predicted_class = np.argmax(prediction)
            confidence = np.max(prediction)
            probabilities = prediction[0]
        else:
            # For models that return class index directly
            predicted_class = int(np.round(prediction[0]))
            confidence = 1.0  # We don't have confidence for these models
            # Create a one-hot encoded array for probabilities
            probabilities = np.zeros(len(FAULT_MAP))
            probabilities[predicted_class] = 1.0
            
        return predicted_class, confidence, probabilities
    except Exception as e:
        st.error(f"Error making prediction: {str(e)}")
        return 0, 0, np.zeros(len(FAULT_MAP))

def display_fault_info(fault_class):
    """Display detailed information about the detected fault"""
    fault_info = FAULT_MAP[fault_class]
    
    st.subheader(f"Fault Information: {fault_info['name']}")
    st.markdown(f"**Description:** {fault_info['description']}")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Typical Symptoms:**")
        for symptom in fault_info['symptoms']:
            st.markdown(f"- {symptom}")
    
    with col2:
        st.markdown("**Recommended Actions:**")
        for action in fault_info['actions']:
            st.markdown(f"- {action}")
    
    if fault_class != 0:  # If not "No Fault"
        st.warning("⚠️ Immediate attention required!")

def display_model_info(model_name, model_info):
    """Display information about the selected model"""
    st.subheader("Model Information")
    
    if model_info:
        st.markdown(f"**Name:** {model_name}")
        st.markdown(f"**Type:** {model_info.get('type', 'N/A')}")
        st.markdown(f"**Architecture:** {model_info.get('architecture', 'N/A')}")
        st.markdown(f"**Training Date:** {model_info.get('training_date', 'N/A')}")
        st.markdown(f"**Accuracy:** {model_info.get('accuracy', 'N/A')}")
        st.markdown(f"**Description:** {model_info.get('description', 'No description available.')}")
    else:
        st.warning("No information available for this model.")

def create_probability_chart(probabilities):
    """Create and return a customized probability distribution chart"""
    prob_df = pd.DataFrame({
        "Fault Type": [FAULT_MAP[i]['name'] for i in range(len(FAULT_MAP))],
        "Probability": probabilities
    })
    
    # Set up custom style
    plt.style.use('default')
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Customize the style
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.size': 10,
        'axes.grid': True,
        'grid.alpha': 0.3,
        'grid.linestyle': '--'
    })
    
    # Create gradient colors
    colors = plt.cm.viridis(np.linspace(0, 0.8, len(prob_df)))
    
    # Sort by probability for better visualization
    prob_df = prob_df.sort_values('Probability', ascending=True)
    
    # Plot horizontal bars with gradient colors
    bars = ax.barh(prob_df["Fault Type"], prob_df["Probability"], color=colors)
    
    # Customize the plot
    ax.set_xlim(0, 1)
    ax.set_xlabel("Probability", fontsize=12, fontweight='bold')
    ax.set_title("Fault Probability Distribution", fontsize=14, fontweight='bold', pad=20)
    
    # Add value labels on the bars
    for i, bar in enumerate(bars):
        width = bar.get_width()
        if width > 0.01:  # Only show labels for significant probabilities
            ax.text(max(width + 0.01, 0.05), bar.get_y() + bar.get_height()/2,
                    f'{width:.2%}', ha='left', va='center',
                    fontweight='bold', fontsize=10)
    
    # Customize grid and spines
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Adjust layout
    plt.tight_layout()
    return fig

def create_confusion_matrix_plot(confusion_matrix):
    """Create and return a customized confusion matrix plot"""
    plt.style.use('default')
    fig, ax = plt.subplots(figsize=(12, 8))
    
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.size': 10,
        'axes.grid': True,
        'grid.alpha': 0.3,
        'grid.linestyle': '--'
    })
    
    # Create heatmap with custom colormap and styling
    sns.heatmap(
        confusion_matrix, 
        annot=True, 
        fmt='d', 
        cmap='viridis',
        xticklabels=[FAULT_MAP[i]['name'] for i in range(len(FAULT_MAP))],
        yticklabels=[FAULT_MAP[i]['name'] for i in range(len(FAULT_MAP))],
        ax=ax,
        annot_kws={'size': 12, 'weight': 'bold'},
        cbar_kws={'label': 'Number of Samples'}
    )
    
    # Customize the plot
    ax.set_title('Confusion Matrix', fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel('Predicted Label', fontsize=12, fontweight='bold')
    ax.set_ylabel('True Label', fontsize=12, fontweight='bold')
    
    # Rotate labels for better readability
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    
    # Adjust layout
    plt.tight_layout()
    return fig

# Initialize session state for input values if not present
def init_session_state():
    if 'ia' not in st.session_state:
        st.session_state.ia = -64.59840133
    if 'ib' not in st.session_state:
        st.session_state.ib = 34.48079878
    if 'ic' not in st.session_state:
        st.session_state.ic = 27.25006492
    if 'va' not in st.session_state:
        st.session_state.va = 0.131668579
    if 'vb' not in st.session_state:
        st.session_state.vb = -0.563834635
    if 'vc' not in st.session_state:
        st.session_state.vc = 0.432166056

# Streamlit app
def main():
    st.set_page_config(
        page_title="Power System Fault Analyzer", 
        page_icon="⚡",
        layout="wide"
    )
    
    # Initialize session state
    init_session_state()
    
    # Load available models
    available_models = get_available_models()
    
    # Sidebar configuration
    st.sidebar.title("⚙️ Configuration")
    
    # Model type and selection
    if available_models:
        # Group models by type
        dl_models = {name: info for name, info in available_models.items() if info['type'] == 'Deep Learning'}
        ml_models = {name: info for name, info in available_models.items() if info['type'] == 'Machine Learning'}
        
        # Model type selection
        model_type = st.sidebar.radio(
            "Select Model Type",
            ["Deep Learning", "Machine Learning"],
            index=0
        )
        
        # Filter models based on type
        model_dict = dl_models if model_type == "Deep Learning" else ml_models
        available_model_names = list(model_dict.keys())
        
        if available_model_names:
            selected_model = st.sidebar.selectbox(
                "Select Model",
                available_model_names,
                index=0
            )
            model_path = available_models[selected_model]["path"]
            model = load_trained_model(model_path)
            model_info = available_models[selected_model]["info"]
        else:
            st.sidebar.warning(f"No {model_type} models available.")
            st.stop()
    else:
        st.error("No models found in the configured directories!")
        st.stop()
    
    # Main content
    st.title("⚡ Power System Fault Analyzer")
    st.markdown("""
    This advanced tool detects and classifies faults in electrical power systems using machine learning models.
    Enter the current and voltage measurements to analyze potential faults.
    """)
    
    # Create tabs
    tab1, tab2, tab3 = st.tabs(["Fault Detection", "Model Information", "Fault Reference"])
    
    with tab1:
        st.header("Real-time Fault Detection")
        
        # Input section
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Current Measurements (A)")
            ia = st.number_input("Ia (Phase A Current)", 
                                value=st.session_state.ia, 
                                min_value=-1000.0, 
                                max_value=1000.0,
                                step=0.1,
                                format="%.8f",
                                key="ia_input")
            ib = st.number_input("Ib (Phase B Current)", 
                                value=st.session_state.ib, 
                                min_value=-1000.0, 
                                max_value=1000.0,
                                step=0.1,
                                format="%.8f",
                                key="ib_input")
            ic = st.number_input("Ic (Phase C Current)", 
                                value=st.session_state.ic, 
                                min_value=-1000.0, 
                                max_value=1000.0,
                                step=0.1,
                                format="%.8f",
                                key="ic_input")
        
        with col2:
            st.subheader("Voltage Measurements (V)")
            va = st.number_input("Va (Phase A Voltage)", 
                                value=st.session_state.va, 
                                min_value=-1000.0, 
                                max_value=1000.0,
                                step=0.1,
                                format="%.9f",
                                key="va_input")
            vb = st.number_input("Vb (Phase B Voltage)", 
                                value=st.session_state.vb, 
                                min_value=-1000.0, 
                                max_value=1000.0,
                                step=0.1,
                                format="%.9f",
                                key="vb_input")
            vc = st.number_input("Vc (Phase C Voltage)", 
                                value=st.session_state.vc, 
                                min_value=-1000.0, 
                                max_value=1000.0,
                                step=0.1,
                                format="%.9f",
                                key="vc_input")
        
        # Update session state with current values
        st.session_state.ia = ia
        st.session_state.ib = ib
        st.session_state.ic = ic
        st.session_state.va = va
        st.session_state.vb = vb
        st.session_state.vc = vc
        
        # Create input array
        input_data = [ia, ib, ic, va, vb, vc]
        
        # Prediction button
        if st.button("⚡ Analyze System", type="primary", use_container_width=True):
            with st.spinner("Analyzing measurements..."):
                if model is not None:
                    predicted_class, confidence, probabilities = predict_fault(model, input_data)
                    
                    # Display results
                    result_col1, result_col2 = st.columns([3, 1])
                    
                    with result_col1:
                        st.success(f"🔍 **Detection Result:** {FAULT_MAP[predicted_class]['name']}")
                    
                    with result_col2:
                        st.metric("Confidence Level", f"{confidence*100:.2f}%")
                    
                    # Probability distribution
                    st.subheader("Fault Probability Distribution")
                    fig = create_probability_chart(probabilities)
                    st.pyplot(fig)
                    
                    # Detailed fault information
                    display_fault_info(predicted_class)
                else:
                    st.error("Unable to perform analysis. Please check model selection.")
    
    with tab2:
        display_model_info(selected_model, model_info)
        
        # Additional model statistics
        st.subheader("Performance Metrics")
        if model_info and model_info.get("metrics"):
            metrics = model_info["metrics"]
            col1, col2, col3 = st.columns(3)
            col1.metric("Accuracy", f"{metrics.get('accuracy', 'N/A')}")
            col2.metric("Precision", f"{metrics.get('precision', 'N/A')}")
            col3.metric("Recall", f"{metrics.get('recall', 'N/A')}")
            
            # Confusion matrix if available
            if "confusion_matrix" in model_info:
                st.subheader("Confusion Matrix")
                cm = np.array(model_info["confusion_matrix"])
                fig = create_confusion_matrix_plot(cm)
                st.pyplot(fig)
        else:
            st.info("No performance metrics available for this model. Upload model metrics to the model_info.json file to see detailed performance data.")
    
    with tab3:
        st.header("Fault Type Reference Guide")
        
        selected_fault = st.selectbox(
            "Select a fault type to learn more:",
            [f"{i} - {FAULT_MAP[i]['name']}" for i in range(len(FAULT_MAP))]
        )
        fault_class = int(selected_fault.split(" - ")[0])
        
        display_fault_info(fault_class)
        
        # Add fault visualization placeholder
        st.subheader("Typical Waveform Pattern")
        st.info("Waveform visualizations will be available in future updates. Upload fault waveform images to enable this feature.")
        
        # Fault statistics (placeholder data - can be replaced with actual statistics)
        st.subheader("Industry Statistics")
        if fault_class == 0:
            st.markdown("""
            - **Normal Operation:** Represents ideal system conditions
            - **Occurrence:** Expected during >99% of system operation time
            - **Impact:** No damage or interruption to service
            """)
        elif fault_class == 1:  # LG
            st.markdown("""
            - **Frequency:** ~70% of all power system faults
            - **Clearance Time:** Typically 3-5 cycles
            - **Protection:** Usually cleared by ground overcurrent relays
            """)
        elif fault_class == 2:  # LL
            st.markdown("""
            - **Frequency:** ~15% of all power system faults
            - **Clearance Time:** Typically 3-6 cycles
            - **Protection:** Usually cleared by phase overcurrent or distance relays
            """)
        elif fault_class == 3:  # LLG
            st.markdown("""
            - **Frequency:** ~10% of all power system faults
            - **Clearance Time:** Typically 4-7 cycles
            - **Protection:** Usually cleared by both ground and phase protection
            """)
        elif fault_class == 4:  # LLL
            st.markdown("""
            - **Frequency:** ~3% of all power system faults
            - **Clearance Time:** Typically 2-4 cycles
            - **Protection:** Usually cleared by differential or distance protection
            """)
        elif fault_class == 5:  # LLLG
            st.markdown("""
            - **Frequency:** ~2% of all power system faults
            - **Clearance Time:** Typically 2-3 cycles
            - **Protection:** Usually cleared by main differential protection
            """)
    
    # Footer with sample test data
    st.sidebar.markdown("---")
    st.sidebar.markdown("### Quick Test Samples")
    
    def set_sample_values(values):
        st.session_state.ia = values[0]
        st.session_state.ib = values[1]
        st.session_state.ic = values[2]
        st.session_state.va = values[3]
        st.session_state.vb = values[4]
        st.session_state.vc = values[5]
    
    # Sample test data for different fault types
    samples = {
        "No Fault": [-64.59840133, 34.48079878, 27.25006492, 0.131668579, -0.563834635, 0.432166056],
        "LG Fault": [232.5, 1.32, 1.21, 12.3, 229.5, 230.1],
        "LL Fault": [4.8, 217.1, 1.0, 120.5, 15.3, 229.8],
        "LLG Fault": [325.3, 298.7, 1.2, 10.5, 8.9, 230.1],
        "Three-Phase Fault": [389.7, 412.5, 378.2, 5.2, 4.7, 6.1],
        "LLLG Fault": [485.2, 502.7, 493.5, 0.7, 0.5, 0.6]
    }
    
    for name, values in samples.items():
        if st.sidebar.button(name, key=f"sample_{name}"):
            set_sample_values(values)
            st.rerun()
    
    st.sidebar.markdown("---")
    st.sidebar.info("""
    **Instructions:**
    1. Select a model from the dropdown
    2. Enter current and voltage measurements or use a quick test sample
    3. Click 'Analyze System'
    4. Review results and recommendations
    
    **Note:** For more accurate results, ensure model paths and directories are correctly configured.
    """)
    
    # Version information
    st.sidebar.markdown("---")
    st.sidebar.caption("Power System Fault Analyzer v1.0.1")
    st.sidebar.caption("© 2025 Power Systems Engineering")

if __name__ == "__main__":
    main()